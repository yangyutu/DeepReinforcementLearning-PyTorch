import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

class TwoAgentFreeSpaceMazeTwoD(gym.Env):

    def __init__(self, config):
        super(TwoAgentFreeSpaceMazeTwoD, self).__init__()
        self.config = config

        self.endStep = self.config['episodeLength']
        self.mapHeight = self.config.get('mapHeight', 5)
        self.mapWidth = self.config.get('mapWidth', 6)
        self.timeScale = self.config.get('timeScale', 80.0)
        self.lengthScale = self.config.get('lengthScale', 5.0)

        self.mineVec = np.ones(self.mapHeight)
        self.numAgents = 2
        self.stepCount = 0
        self.agentPositions = 0
        self.nbActions = [ self.mapHeight + 4 for _ in range(self.numAgents)]

        self.stateDim = [ self.mapHeight + 4 for _ in range(self.numAgents)]
        self.epiCount = 0

        random.seed(1)

    def calReward(self):
        done = False
        reward = 0.0
        if self.agentPositions[1][0] == self.mapWidth:
            verticalPos = self.agentPositions[1][1]
            if self.mineVec[verticalPos]:
                self.mineVec[verticalPos] = 0
                reward -= 0.1
                print('agent 1 clear mine', self.agentPosition[1])
        if self.agentPositions[0][0] == self.mapWidth:
            verticalPos = self.agentPositions[1][1]
            reward += 1.0
            done = True
            if self.mineVec[verticalPos]:
                reward -= 2.0
                print('agent 0 finish but on mine', self.agentPosition[1])
            else:
                print('agent 0 finish without mine', self.agentPosition[1])

        return reward, done

    def step(self, action):

        self.stepCount += 1
        for n in range(self.numAgents):
            i = self.agentPositions[n][0]
            j = self.agentPositions[n][1]
            if action == 0 and not self.is_on_obstacle(i - 1, j):
                self.agentPositions[n][0] -= 1
            if action == 1 and not self.is_on_obstacle(i + 1, j):
                self.agentPositions[n][0] += 1
            if action == 2 and not self.is_on_obstacle(i, j - 1):
                self.agentPositions[n][1] -= 1
            if action == 3 and not self.is_on_obstacle(i, j + 1):
                self.agentPositions[n][1] += 1

        # state is a list
        state = []
        state.append(np.concatenate((self.mineVec, self.agentPositions / self.lengthScale), axis = None))
        state.append(np.concatenate((self.mineVec, self.agentPositions / self.lengthScale), axis = None))

        reward, done = self.calReward()

        return state, reward, done, self.agentPositions.copy()

    def reset(self):

        self.epiCount += 1
        self.stepCount = 0

        self.agentPositions = np.array([[0.0, 0.0], []])
        for n in range(self.numAgents):
            yPosition = random.randint(0, self.mapHeight - 1)
            self.agentPositions[n][1] = yPosition

        state = []
        state.append(np.concatenate((self.mineVec, self.agentPositions / self.lengthScale), axis=None))
        state.append(np.concatenate((self.mineVec, self.agentPositions / self.lengthScale), axis=None))

        return state

    def is_on_obstacle(self, i, j):
        return not (0 <= i < self.mapHeight and  0 <= j < self.mapWidth)

    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass