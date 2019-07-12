from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.DQN.DQN import DQNAgent

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


class StackedDQNAgent(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None, timeIndexMap=None):

        self.policyNets = policyNet
        self.targetNets = targetNet
        self.optimizers = optimizer

        super(StackedDQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction,
                                              stateProcessor, experienceProcessor)
        self.policyNet = None
        self.targetNet = None
        self.optimizer = None
        self.timeIndexMap = timeIndexMap
        self.init_memory()

    def initialization(self):
        # move model to correct device
        for i in range(len(self.policyNets)):
            self.policyNets[i] = self.policyNets[i].to(self.device)

        # in case targetNet is None
        for i in range(len(self.targetNets)):
            if self.targetNets[i] is not None:
                self.targetNets[i] = self.targetNets[i].to(self.device)

        self.dirName = 'Log/'
        if 'dataLogFolder' in self.config:
            self.dirName = self.config['dataLogFolder']
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)

        self.identifier = ''
        self.epIdx = 0
        self.learnStepCounter = 0  # for target net update
        self.globalStepCount = 0
        self.losses = []
        self.rewards = []
        self.nStepBuffer = []

    def init_memory(self):

        self.memories = [ReplayMemory(self.memoryCapacity) for _ in range(len(self.policyNets))]

    def store_experience(self, state, action, nextState, reward, info):

        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, info)
            # caution: using multiple step forward return can increase variance
            # if it is one step

        timeStep = self.timeIndexMap[state['timeStep']]
        transition = Transition(state, action, nextState, reward)
        self.memories[timeStep].push(transition)

    def train(self):

        runningAvgEpisodeReward = 0.0
        if len(self.rewards) > 0:
            runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            state = self.env.reset()

            rewardSum = 0

            for stepCount in range(self.episodeLength):

                timeStep = self.timeIndexMap[state['timeStep']]

                epsThreshold = self.epsilon_by_step(self.globalStepCount * (timeStep + 1) / len(self.policyNets))

                action = self.select_action(self.policyNets[timeStep], state, epsThreshold)

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

            runningAvgEpisodeReward = (runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
            print("done in step count: {}".format(stepCount))
            print("reward sum = " + str(rewardSum))
            print("running average episode reward sum: {}".format(runningAvgEpisodeReward))
            print(info)

            self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                self.save_checkpoint()

            self.epIdx += 1
        self.save_all()

    def update_net(self, state, action, nextState, reward, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, info)


        if self.hindSightER and nextState is not None and self.globalStepCount % self.hindSightERFreq == 0:
            stateNew, actionNew, nextStateNew, rewardNew = self.env.getHindSightExperience(state, action, nextState, info)
            if stateNew is not None:
                self.store_experience(stateNew, actionNew, nextStateNew, rewardNew, info)


        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience

            for i in range(len(self.memories) - 1, -1, -1):
                if len(self.memories[i]) < self.trainBatchSize:
                    continue

                transitions_raw = self.memories[i].sample(self.trainBatchSize)
                self.policyNet = self.policyNets[i]
                self.optimizer = self.optimizers[i]

                if self.netUpdateOption == 'targetNet' or self.netUpdateOption == 'doubleQ':
                    if i < (len(self.memories) - 1):
                        self.targetNet = self.targetNets[i + 1]
                    else:
                        self.targetNet = self.targetNets[i]
                if self.netUpdateOption == 'policyNet':
                    raise NotImplementedError

                loss = self.update_net_on_transitions(transitions_raw, self.netLossFunc, 1,
                                                      updateOption=self.netUpdateOption, netGradClip=self.netGradClip,
                                                      info=info)

                if self.globalStepCount % self.lossRecordStep == 0:
                    self.losses.append([self.globalStepCount, self.epIdx, loss])

                if self.learnStepCounter % self.targetNetUpdateStep == 0:
                    self.targetNet.load_state_dict(self.policyNet.state_dict())

            self.learnStepCounter += 1

    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')

    def save_checkpoint(self):
        prefix = self.dirName + self.identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        # self.loadLosses(prefix + '_loss.txt')
        # self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memories = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        for i in range(len(self.policyNets)):
            self.policyNets[i].load_state_dict(checkpoint['model_state_dict'][i])
            self.targetNets[i].load_state_dict(checkpoint['model_state_dict'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'][i])