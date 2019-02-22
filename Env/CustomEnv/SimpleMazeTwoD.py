

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class SimpleMazeTwoD(gym.Env):

    def __init__(self, fileName):
        super(SimpleMazeTwoD, self).__init__()

        self.map = np.genfromtxt(fileName)
        self.stepCount = 0
        self.currentState = np.array([8, 1], dtype=np.int32)
        self.targetState = np.array([1, 1], dtype=np.int32)
        self.nbActions = 4
        self.stateDim = 2
        self.endStep = 200
        random.seed(1)

    def step_count(self):
        return self.stepCount

    def step(self, action):
        self.stepCount += 1
        i = self.currentState[0]
        j = self.currentState[1]
        if action == 0 and not self.is_on_obstacle(i - 1, j):
            self.currentState[0] -= 1
        if action == 1 and not self.is_on_obstacle(i + 1, j):
            self.currentState[0] += 1
        if action == 2 and not self.is_on_obstacle(i, j - 1):
            self.currentState[1] -= 1
        if action == 3 and not self.is_on_obstacle(i, j + 1):
            self.currentState[1] += 1

        if np.array_equal(self.currentState, self.targetState):
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        if self.stepCount > self.endStep:
            done = True
            reward = 0.0

        return np.array(self.currentState), reward, done, {}

    def reset(self):

        self.stepCount = 0
        self.currentState = np.array([8, 1], dtype=np.int32)
        return np.array(self.currentState)

    def is_on_obstacle(self, i, j):
        if 0 <= i < self.map.shape[0] and 0 <= j < self.map.shape[1]:
            return self.map[i, j] == 0
        return True
    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass

    def render_traj(self, stateSet, ax):
        idx, idy = np.where(self.map == 0)
        ax.scatter(idx, idy, c='black', marker='s', s=5)
        ax.set_xlim([-1, self.map.shape[0] + 1])
        ax.set_ylim([-1, self.map.shape[1] + 1])
        ax.scatter(self.targetState[0], self.targetState[1], c='green', marker='s', s=10)

        traj = np.array(stateSet)
        ax.plot(traj[:,0], traj[:,1])


    def plot_map(self, ax):
        idx, idy = np.where(self.map == 0)
        ax.scatter(idx, idy, c='black', marker='s', s = 10)
        ax.set_xlim([-1, self.map.shape[0] + 1])
        ax.set_ylim([-1, self.map.shape[1] + 1])
        ax.scatter(self.targetState[0], self.targetState[1], c='green', marker='s', s = 10 )




