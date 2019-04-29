
from Agents.DQN.DQN import DQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import os
import math
import pickle
from copy import deepcopy

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MADQNAgent(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, episodeFinalProcessor = None):
        super(MADQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor)


    def read_config(self):
        super(MADQNAgent, self).read_config()
        self.numAgents = self.config['N']

    def select_action(self, net, states, epsThreshold):
        # we need to select multiple actions

        randNums = np.random.rand(self.numAgents)
        actions = np.random.randint(0, self.numAction, self.numAgents)

        nonRandIdx = np.where(randNums > epsThreshold)[0]

        if len(nonRandIdx) > 0:
            states_nonRandom = [states[i] for i in nonRandIdx]

            with torch.no_grad():
                if self.stateProcessor is not None:
                    states_nonRandom, _ = self.stateProcessor(states_nonRandom, self.device)
                    QValues = net(states_nonRandom)
                else:
                    stateTorch = torch.tensor(states_nonRandom, dtype=torch.float)
                    QValues = net(stateTorch.to(self.device))
                actions_nonRandom = torch.argmax(QValues, dim=1).cpu().numpy()

            actions[nonRandIdx] = actions_nonRandom

        return actions


    def train(self):

        runningAvgEpisodeReward = 0.0

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            states = self.env.reset()
            rewardSum = 0

            self.work_At_Episode_Begin()

            for stepCount in range(self.episodeLength):

                self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

                actions = self.select_action(self.policyNet, states, self.epsThreshold)

                nextStates, rewards, done, infos = self.env.step(actions)

                if stepCount == 0:
                    print("at step 0:")
                    print(infos)

                if self.verbose:
                    print('step: ', trainStepCount)
                    print(infos)

                if done:
                    nextStates = [None for _ in nextStates]

                # learn the transition
                self.update_net(states, actions, nextStates, rewards, infos)

                states = nextStates

                rewardSum += np.sum(rewards*math.pow(self.gamma, stepCount))
                self.globalStepCount += 1

                if done:
                    break

            self.epIdx += 1
            runningAvgEpisodeReward = (runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
            print('episode', self.epIdx)
            print("done in step count:")
            print(stepCount)
            print('reward sum')
            print(rewardSum)
            print(infos)
            # print('reward sum: ', rewardSum)
            print("running average episode reward sum: {}".format(runningAvgEpisodeReward))

            self.rewards.append([self.epIdx, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                self.save_checkpoint()

        self.save_all()

    def store_experience(self, states, actions, nextStates, rewards, infos):

        for i in range(len(states)):
            transition = Transition(states[i], actions[i], nextStates[i], rewards[i])
            self.memory.push(transition)

