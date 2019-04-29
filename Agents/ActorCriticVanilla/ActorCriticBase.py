import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math
from utils.utils import torchvector

class ActorCriticBase(object):
    def __init__(self, config, env, netLossFunc, nbAction, stateProcessor = None):
        self.config = config
        self.readConfig()
        self.numActions = nbAction
        self.env = env[0]
        self.testEnv = env[1]
        self.stateProcessor = stateProcessor
        self.netLossFunc = netLossFunc


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

    def calProcessReward(self):

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
        pass
    def train(self):
        pass

    def getStateOptimalAction(self, state):
        pass


    def test(self, numEpisode):

        averageReward = 0
        for epIdx in range(numEpisode):
            #print("episode index:" + str(epIdx))
            rewardSum = 0.0
            state = self.testEnv.reset()
            done = False
            stepCount = 0
            while not done:
                action = self.getStateOptimalAction(state)
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


