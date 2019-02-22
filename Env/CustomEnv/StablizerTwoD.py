

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class StablizerTwoD(gym.Env):

    def __init__(self):
        super(StablizerTwoD, self).__init__()

        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 5
        self.stateDim = 2
        self.endStep = 200

    def step(self, action):
        self.stepCount += 1
        self.currentState += (np.random.rand(2)-0.5)*0.4
        if action == 1: # move to positive
            self.currentState[0] += 0.1
        if action == 2: # move to negative
            self.currentState[0] -= 0.1
        if action == 3: # move to up
            self.currentState[1] += 0.1
        if action == 4: # move to down
            self.currentState[1] -= 0.1

        if np.linalg.norm(self.currentState, ord=np.inf) < 1:
            reward = - np.linalg.norm(self.currentState, ord=np.inf)
            done = False
        else:
            reward = - np.linalg.norm(self.currentState, ord=np.inf) * (self.endStep - self.stepCount)
            done = True

        if self.stepCount > self.endStep:
            done = True


        return np.array(self.currentState), reward, done, {}

    def reset(self):

        self.stepCount = 0
        self.currentState = (np.random.rand(2) - 0.5)-0.1

        return np.array(self.currentState)


    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass

    def render_traj(self, stateSet, ax):
        stateSet = np.array(stateSet)
        ax.plot(stateSet[:,0],stateSet[:,1])




