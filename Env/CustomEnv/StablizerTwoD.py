

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math

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




class StablizerTwoDContinuous(gym.Env):

    def __init__(self, config = None, seed = 1):
        super(StablizerTwoDContinuous, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 2
        self.stateDim = 2
        self.endStep = 200
        self.randomSeed = seed

        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.infoDict = {'reset': False, 'endBeforeDone': False, 'stepCount': 0}

    def step_count(self):
        return self.stepCount

    def step(self, action):
        self.stepCount += 1
        self.infoDict['stepCount'] = self.stepCount
        self.currentState += np.random.randn(2)*0.1

        # action is the movement amount
        self.currentState += action
        reward = 0.0
        done = False
        if np.linalg.norm(self.currentState, ord=np.inf) < 0.5:
            reward = 1
            done = True
            self.infoDict['done_state'] = self.currentState.copy()

        if np.linalg.norm(self.currentState, ord=np.inf) > 5:
            reward = -1

        # if abs(self.currentState) < 5:
        #     reward = - abs(self.currentState[0])
        #     done = False
        # else:
        #     reward = - abs(self.currentState[0]) * (self.endStep - self.stepCount)
        #     done = True
        #     self.infoDict['done_state'] = self.currentState


        return self.currentState.copy(), reward, done, self.infoDict.copy()

    def reset(self):
        #self.infoDict['reset'] = True
        self.infoDict['stepCount'] = 0
        self.stepCount = 0
        self.currentState = (np.random.rand(2) - np.array([0.5, 0.5]))*8
        self.infoDict['initial state'] = self.currentState.copy()
        return self.currentState


    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass


class StablizerTwoDContinuousSP(gym.Env):

    def __init__(self, config = None, seed = 1):
        super(StablizerTwoDContinuousSP, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 1
        self.stateDim = 2
        self.endStep = 200
        self.randomSeed = seed
        self.Dr = 0.161
        self.tau = 1 / self.Dr  # tau about 6.211180124223603
        self.Tc = 0.1 * self.tau  # Tc is control interval
        self.angleStd = math.sqrt(2 * self.Tc * self.Dr)
        self.targetState = np.array([0,0])
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.infoDict = {'reset': False, 'endBeforeDone': False, 'stepCount': 0}
        random.seed(self.randomSeed)
    def step_count(self):
        return self.stepCount

    def step(self, action):
        self.stepCount += 1
        self.infoDict['stepCount'] = self.stepCount
        #self.currentState += np.random.randn(2)*0.1
        self.infoDict['previousState'] = self.currentState.copy()

        # action is the movement amount
        phi = self.currentState[2]
        self.currentState[0] += action * math.cos(phi) * 1
        self.currentState[1] += action * math.sin(phi) * 1

        self.currentState[2] += random.gauss(0, self.angleStd)
        self.currentState[2] = (self.currentState[2] + 2 * np.pi) % (2 * np.pi)
        self.infoDict['currentState'] = self.currentState.copy()
        reward = 0.0
        done = False
        distance = self.targetState - self.currentState[0:2]
        if np.linalg.norm(distance, ord=np.inf) < 0.5:
            reward = 1
            done = True
            self.infoDict['done_state'] = self.currentState.copy()

        #if np.linalg.norm(distance, ord=np.inf) > 5:
        #    reward = -1


        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)
        # if abs(self.currentState) < 5:
        #     reward = - abs(self.currentState[0])
        #     done = False
        # else:
        #     reward = - abs(self.currentState[0]) * (self.endStep - self.stepCount)
        #     done = True
        #     self.infoDict['done_state'] = self.currentState


        return np.array([dx, dy]), reward, done, self.infoDict

    def reset(self):
        #self.infoDict['reset'] = True
        self.infoDict = {}
        self.infoDict['stepCount'] = 0
        self.stepCount = 0
        self.currentState = (np.random.rand(3) - np.array([0.5, 0.5, 0.5]))*8
        self.currentState[2] = (self.currentState[2] + 2 * np.pi) % (2 * np.pi)
        self.infoDict['initial state'] = self.currentState.copy()
        phi = self.currentState[2]
        distance = self.targetState - self.currentState[0:2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)


        return np.array([dx, dy])


    def getHindSightExperience(self, state, action, nextState, info):

        targetNew = self.infoDict['currentState'][0:2]
        phi = self.infoDict['previousState'][2]
        distance = targetNew - self.infoDict['previousState'][0:2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        stateNew = np.array([dx, dy])
        nextStateNew = None
        actionNew = action
        rewardNew = 1
        stateNorm = np.linalg.norm(stateNew, ord=2)
        return stateNew, actionNew, nextStateNew, rewardNew


    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass
