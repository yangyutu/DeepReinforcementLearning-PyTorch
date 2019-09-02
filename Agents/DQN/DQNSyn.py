
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
from copy import deepcopy

# DQN agent uses synchronized environment for training


class DQNSynAgent(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, episodeFinalProcessor = None):
        super(DQNSynAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor)
        self.episodeFinalProcessor = episodeFinalProcessor

    def read_config(self):
        super(DQNSynAgent, self).read_config()
        self.numWorkers = self.config['numWorkers']
        self.successRepeat = False
        if 'successRepeat' in self.config:
            self.successRepeat = self.config['successRepeat']
        self.successRepeatTime = 1
        if 'successRepeatTime' in self.config:
            self.successRepeatTime = self.config['successRepeatTime']

    def select_action(self, net, states, epsThreshold):
        # return a list of actions
        # get a random number so that we can do epsilon exploration
        randNum = random.random()
        if randNum > epsThreshold:
            with torch.no_grad():
                if self.stateProcessor is not None:
                    states, _ = self.stateProcessor(states, self.device)
                    QValues = net(states)
                else:
                    stateTorch = torch.tensor(states, dtype=torch.float)
                    QValues = net(stateTorch.to(self.device))
                actions = torch.argmax(QValues, dim=1).cpu().numpy()
        else:
            actions = np.random.randint(0, self.numAction, self.numWorkers)

        return actions


    def train(self):

        runningAvgEpisodeReward = 0.0

        # get a list of reset states
        states = self.env.reset()
        states = states.tolist()
        dummyStates = deepcopy(states)
        rewardSum = 0
        stepCountList = np.zeros(self.numWorkers)
        for trainStepCount in range(self.trainStep):

            self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

            actions = self.select_action(self.policyNet, states, self.epsThreshold)

            nextStates, rewards, dones, infos = self.env.step(actions)

            if self.verbose:
                print('step: ', trainStepCount)
                print(infos)

            nextStates = nextStates.tolist()
            if np.any(dones):
                idx = np.where(dones==True)
#                nextStatesCopy = deepcopy(nextStates)
                for i in np.nditer(idx):
                    # if not ended due to step limit
                    # in baseline vec_env, if done, nextState = reset state
                    if not infos[i]['endBeforeDone']:
                        nextStates[i] = None

            # learn the transition
            self.update_net(states, actions, nextStates, rewards, infos)

            states = nextStates


            self.globalStepCount += self.numWorkers

            rewardSum += np.sum(rewards*np.power(self.gamma, stepCountList))
            stepCountList += 1

            if np.any(dones):
                idx = np.where(dones == True)
                # adjust for states to avoid None in states
                for i in np.nditer(idx):
                    states[i] = dummyStates[i]
                stepCountDone = stepCountList[idx]
                stepCountList[idx] = 0.0
                self.epIdx += len(idx[0])
                runningAvgEpisodeReward = rewardSum / (self.epIdx + 1)
                print('episode', self.epIdx)
                print("done in step count:")
                print(stepCountDone)
                print('rewards')
                print(rewards[idx])
                #print('reward sum: ', rewardSum)
                print("running average episode reward sum: {}".format(runningAvgEpisodeReward))

            self.rewards.append([self.epIdx, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and trainStepCount % (self.config['logFrequency']) == 0:
                self.save_checkpoint()

        self.save_all()

    def store_experience(self, states, actions, nextStates, rewards, infos):

        for i in range(len(states)):
            # if it is ended due to stepLimit, we should not store the experience due to the vec env setup
            if not infos[i]['endBeforeDone']:
                transition = Transition(states[i], actions[i], nextStates[i], rewards[i])
                self.memory.push(transition)
                if self.successRepeat and nextStates[i] is None:
                    for _ in range(self.successRepeatTime):
                        self.memory.push(transition)


