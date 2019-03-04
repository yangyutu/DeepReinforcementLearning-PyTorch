import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math
from utils.utils import torchvector
class ActorCriticVanilla(object):
    def __init__(self, actorNet, criticNet, env, optimizers, netLossFunc, nbAction, config, stateProcessor = None):
        self.config = config
        self.readConfig()
        self.actorNet = actorNet
        self.criticNet = criticNet
        self.env = env[0]
        self.testEnv = env[1]

        self.numActions = nbAction
        self.actorNetOptim = optimizers[0]
        self.criticNetOptim = optimizers[1]

        self.stateProcessor = stateProcessor
        self.netLossFunc = netLossFunc



        self.actorNet.to(self.device)
        self.criticNet.to(self.device)

    def readConfig(self):
        self.trainStep = self.config['trainStep']

        self.device = 'cpu'
        if 'device' in self.config and torch.cuda.is_available():
            self.device = self.config['device']

        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']

        self.numSteps = 30
        if 'numStepsPerSweep' in self.config:
            self.numSteps = self.config['numStepsPerSweep']

        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        np.random.seed(self.randomSeed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.randomSeed)

        self.testFlag = True
        self.testFrequency = 50
        self.testEpisode = 100

        if 'testFlag' in self.config:
            self.testFlag = self.config['testFlag']
        if 'testFrequency' in self.config:
            self.testFrequency = self.config['testFrequency']
        if 'testEpisode' in self.config:
            self.testEpisode = self.config['testEpisode']


        self.gamma = self.config['gamma']

    def discountReward(self):

        # given a series of STEP rewards stored in self.rewards, we want to estimate PROCESS reward

        discounted_r = np.zeros_like(self.rewards)
        running_add = self.final_r
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def stateProcess(self, state):
        if self.stateProcessor is not None:
            return self.stateProcessor([state])
        else:
            return torchvector(state[np.newaxis, :]).to(self.device)

    def collectSample(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.final_r = 0
        done = False
        state = self.previousState
        for j in range(self.numSteps):
            self.states.append(state)
            logSoftmaxAction = self.actorNet(self.stateProcess(self.previousState))
            softmaxAction = torch.exp(logSoftmaxAction)
            action = np.random.choice(self.numActions, p=softmaxAction.cpu().data.numpy()[0])

            one_hot_action = [int(k == action) for k in range(self.numActions)]

            nextState, reward, done, info = self.env.step(action)
            self.actions.append(one_hot_action)
            self.rewards.append(reward)
            state = nextState
            self.previousState = state
            if done:
                state = self.env.reset()
                break
            # if the process is not finished, then we use current value function to calculate continuing reward sum
            # if it is zero, final_r will take default value of zero since the final reward is stored in self.rewards
        if not done:
            self.final_r = self.criticNet(self.stateProcess(state)).cpu().item()

        # transform the states to batch form for network forwarding
        if self.stateProcessor is not None:
            self.states = self.stateProcessor(self.states)
        else:
            self.states = torch.tensor(self.states, device=self.device, dtype=torch.float32)
        # actions is a batch of one-hot actions
        self.actions = torch.tensor(self.actions, device=self.device, dtype=torch.float32)


    def train(self):


        self.previousState = self.env.reset()
        for self.step in range(self.trainStep):

            # run and collect samples
            self.collectSample()

            self.actorNetOptim.zero_grad()
            self.criticNetOptim.zero_grad()

            # input a batch of states, output a batch of log(pi(a|s))
            logSoftmaxActions = self.actorNet(self.states)


            values = self.criticNet(self.states).detach()

            # calculate Qs, Qvalues is numpy array, and should not calculate grad
            # since it is at cpu, we need to move to the right device
            Qvalues = self.discountReward()
            Qvalues = torch.from_numpy(Qvalues).to(self.device).float()

            advantages = Qvalues - values

            # logSoftmaxActions*actions is elementwise product
            actorNetLoss = -torch.mean(torch.sum(logSoftmaxActions*self.actions, 1)*advantages)
            actorNetLoss.backward()

            # Qvalues is numpy array
            targetValues = Qvalues
            values = self.criticNet(self.states)
            criticNetLoss = self.netLossFunc(values, targetValues)
            criticNetLoss.backward()

            if self.netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), self.netGradClip)
                torch.nn.utils.clip_grad_norm_(self.criticNet.parameters(), self.netGradClip)

            self.actorNetOptim.step()
            self.criticNetOptim.step()

            if self.testFlag and self.step % self.testFrequency == 0:
                self.test(self.testEpisode)

    def test(self, numEpisode):

        averageReward = 0
        for epIdx in range(numEpisode):
            #print("episode index:" + str(epIdx))
            rewardSum = 0.0
            state = self.testEnv.reset()
            done = False
            stepCount = 0
            while not done:
                softmaxAction = torch.exp(self.actorNet(self.stateProcess(state)).detach())
                action = np.argmax(softmaxAction.cpu().numpy()[0])
                next_state, reward, done, info = self.testEnv.step(action)
                rewardSum += reward
                state = next_state
                if done:
            #        print("done in step count: {}".format(stepCount))
                    break
                stepCount += 1
            #print("reward sum = " + str(rewardSum))
            averageReward += rewardSum

        averageReward /= numEpisode
        print("step:",self.step,"test result:",averageReward)