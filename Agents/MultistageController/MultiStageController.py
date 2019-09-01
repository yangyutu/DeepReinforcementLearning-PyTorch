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


from collections import namedtuple
# Define a namedtuple with name Transition and attributes of state, action, next_state, reward
ExtendedTransition = namedtuple('ExtendedTransition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExtendedReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(transition)

        # write on the earlier experience
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def fetch_all(self):
        return self.memory

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return str(self.memory)

class MultiStageStackedController:

    def __init__(self, config, agents, env):
        self.config = config
        self.env = env
        self.initialization()


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



    def select_action(self, state):

        # first store memory
        stageID = state['stageID']

        return self.agents[stageID].select_action(state)



    def train(self):

        runningAvgEpisodeReward = 0.0
        if len(self.rewards) > 0:
            runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            state = self.env.reset()

            rewardSum = 0

            for stepCount in range(self.episodeLength):



                action = self.select_action(state)

                nextState, reward, doneDict, info = self.env.step(action)

                done = doneDict['global']

                if stepCount == 0:
                    print("at step 0:")
                    print(info)
                if done:
                    nextState = None

                self.update_net(state, action, nextState, reward, doneDict, info)

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

    def update_net(self, state, action, nextState, reward, done, info):

        # first store memory
        stageID = state['stageID']

        self.agents[stageID].store_experience(state, action, nextState, reward, done, info)


        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience

            for i in range(self.numStages - 1, -1, -1):
                if i < (len(self.numStages) - 1):
                    self.agents[i].update_on_memory_given_target(targetAgent=self.agent[i + 1])
                else:
                    self.agents[i].update_on_memory_given_target(targetAgent=None)

            self.learnStepCounter += 1

    def save_all(self):
        for i in range(self.numStages - 1, -1, -1):
            self.agents[i].save_all()

    def save_checkpoint(self):
        for i in range(self.numStages - 1, -1, -1):
            self.agents[i].save_checkpoint()
    def load_checkpoint(self, prefix):
        for i in range(self.numStages - 1, -1, -1):
            self.agents[i].load_checkpoint(prefix)