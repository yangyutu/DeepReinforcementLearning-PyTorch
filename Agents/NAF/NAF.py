
from Agents.DQN.DQN import DQNAgent
from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.ReplayMemoryReward import ReplayMemoryReward
from Agents.Core.PrioritizedReplayMemory import PrioritizedReplayMemory
import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math
import pickle



class NAFAgent(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, experienceProcessor=None):
        super(NAFAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor, experienceProcessor)

        self.init_memory()


    def init_memory(self):

        if self.memoryOption != 'natural':
            raise NotImplementedError

        self.memory = ReplayMemory(self.memoryCapacity)

    def read_config(self):
        super(NAFAgent, self).read_config()
        # read additional parameters
        self.tau = self.config['tau']

    def work_At_Episode_Begin(self):
        pass

    def work_before_step(self, state=None):
        self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

    def train_one_episode(self):

        print("episode index:" + str(self.epIdx))
        state = self.env.reset()
        done = False
        rewardSum = 0

        # any work to be done at episode begin
        self.work_At_Episode_Begin()

        for stepCount in range(self.episodeLength):

            # any work to be done before select actions
            self.work_before_step(state)

            action = self.select_action(self.policyNet, state, noiseFlag=True)

            nextState, reward, done, info = self.env.step(action)

            if stepCount == 0:
                print("at step 0:")
                print(info)

            if done:
                nextState = None

            # learn the transition
            self.update_net(state, action, nextState, reward, info)

            state = nextState
            rewardSum += reward * pow(self.gamma, stepCount)
            self.globalStepCount += 1

            if self.verbose:
                print('action: ' + str(action))
                print('state:')
                print(nextState)
                print('reward:')
                print(reward)
                print('info')
                print(info)

            if done:
                break

        self.runningAvgEpisodeReward = (self.runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
        print("done in step count: {}".format(stepCount))
        print("reward sum = " + str(rewardSum))
        print("running average episode reward sum: {}".format(self.runningAvgEpisodeReward))
        print(info)

        self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, self.runningAvgEpisodeReward])
        if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
            self.save_checkpoint()

        self.epIdx += 1

        return stepCount, rewardSum

    def train(self):

        if len(self.rewards) > 0:
            self.runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):
            self.train_one_episode()
        self.save_all()

    def store_experience(self, state, action, nextState, reward, info):

        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, info)
        transition = Transition(state, action, nextState, reward)
        self.memory.push(transition)

    def update_net(self, state, action, nextState, reward, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, info)

        if self.hindSightER and nextState is not None and self.globalStepCount % self.hindSightERFreq == 0:
            stateNew, actionNew, nextStateNew, rewardNew = self.env.getHindSightExperience(state, action, nextState, info)
            if stateNew is not None:
                self.store_experience(stateNew, actionNew, nextStateNew, rewardNew, info)


        if self.priorityMemoryOption:
            if len(self.memory) < self.config['memoryCapacity']:
                return
        else:
            if len(self.memory) < self.trainBatchSize:
                return


        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience
            for nStep in range(self.netUpdateStep):
                info = {}
                if self.priorityMemoryOption:
                    transitions_raw, b_idx, ISWeights = self.memory.sample(self.trainBatchSize)
                    info['batchIdx'] = b_idx
                    info['ISWeights'] = torch.from_numpy(ISWeights.astype(np.float32)).to(self.device)
                else:
                    transitions_raw= self.memory.sample(self.trainBatchSize)

                loss = self.update_net_on_transitions(transitions_raw, self.netLossFunc, 1, updateOption=self.netUpdateOption, netGradClip=self.netGradClip, info=info)

                if self.globalStepCount % self.lossRecordStep == 0:
                    self.losses.append([self.globalStepCount, self.epIdx, loss])

                if self.learnStepCounter % self.targetNetUpdateStep == 0:
                    self.targetNet.load_state_dict(self.policyNet.state_dict())

                self.learnStepCounter += 1

    def prepare_minibatch(self, transitions_raw):
        # first store memory

        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.float32)  # shape(batch, numActions)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32)  # shape(batch)

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device,
                                        dtype=torch.uint8)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device,
                                             dtype=torch.float32)

        return state, nonFinalMask, nonFinalNextState, action, reward


    def select_action(self, net=None, state=None, noiseFlag = True):

        if net is None:
            net = self.policyNet

        with torch.no_grad():
            # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
            # here state[np.newaxis,:] is to add a batch dimension
            if self.stateProcessor is not None:
                state, _ = self.stateProcessor([state], self.device)
                action = net.select_action(state, noiseFlag)
            else:
                stateTorch = torch.from_numpy(np.array(state[np.newaxis, :], dtype=np.float32))
                action = net.select_action(stateTorch.to(self.device), noiseFlag)

        return action.cpu().data.numpy()[0]

    def update_net_on_transitions(self, transitions_raw, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):

        # order the data
        # convert transition list to torch tensors
        # use trick from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343

        state, nonFinalMask, nonFinalNextState, action, reward = self.prepare_minibatch(transitions_raw)

        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            QValues = self.policyNet.eval_Q_value(state, action).squeeze()

            # Here we detach because we do not want gradient flow from target values to net parameters
            QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
            QNext[nonFinalMask] = self.targetNet.eval_state_value(nonFinalNextState).squeeze().detach()
            targetValues = reward + self.gamma * QNext

            # Compute loss
            loss_single = loss_fun(QValues, targetValues)
            loss = torch.mean(loss_single)

            # Optimize the model
            # zero gradient
            self.optimizer.zero_grad()

            loss.backward()
            if netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(), netGradClip)
            self.optimizer.step()

            return loss.item()
