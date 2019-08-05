import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# why incoporating time into state is important/unimportant?

# Suppose two agents are reponsible for a two-stage navigation
# Suppose first agent has a smaller speed and the second agent has a larger speed.
# If there is no time limit, and the discount factor for agent 1 is 1, the first agent
# will try to end at regions most favorable for agent 2 even if agent 1 has a hard time to get there.
# But if the discount factor of agent 1 < 1, then agent 1 will try to seek the largest
# discounted reward.
# if the discount factor for agent 1 is very big (say 0.8), then agent 1 will try to end at regions
# that is short-ranged to agent but can be unfavorable to agent 2.


class CooperativeSimpleMazeTwoD(gym.Env):

    def __init__(self, config=None):
        super(CooperativeSimpleMazeTwoD, self).__init__()

        self.stepCount = 0
        self.currentState = np.array([8, 1], dtype=np.int32)
        self.targetState = np.array([1, 1], dtype=np.int32)
        self.nbActions = 4
        self.stateDim = 2
        self.endStep = 400
        self.discount = 1
        self.mapHeight = 5
        self.mapWidth = 6
        self.timeScale = 80.0
        self.lengthScale = 5.0
        self.epiCount = 0
        self.numStages = 2

        self.multiStage = True


        self.config = {}
        if config is not None:
            self.config = config
        if 'multiStage' in self.config:
            self.multiStage = self.config['multiStage']

        random.seed(1)

    def calReward(self):

        reward = 0.0

        # if pass the time limit and not finish give zero reward
        if self.stepCount > self.endStep:
            self.done['stage'] = [True for _ in range(self.numStages)]
            self.done['global'] = True
            print('not finish ', self.currentState, self.stageID)
            return reward, copy.deepcopy(self.done)


        if self.stepCount <= self.endStep and self.stageID == 1 and self.isTermnate():
            self.done['stage'][self.stageID] = True
            self.done['global'] = True
            reward = self.discount**self.stepCount
            print('finish ', self.currentState, reward, self.stageID)
        return reward, copy.deepcopy(self.done)

    def isTermnate(self):
        dist = self.currentState - self.targetState
        if np.linalg.norm(dist, ord=np.inf) < 0.5:
            return True
        else:
            return False


    def step(self, action):
        if self.stageID == 0:
            self.stepCount += 1
        else:
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

        if self.currentState[1] > self.mapWidth / 2 and self.stageID == 0:
            # pass job to agent 1
            print('job passage', self.currentState)
            self.done['stage'][self.stageID] = True
            self.stageID += 1


        reward, done = self.calReward()
        state = {'stageID': self.stageID,
                 'state': np.array([self.currentState[0] / self.lengthScale, \
                                    self.currentState[1] / self.lengthScale])
                 }


        if done['stage'][0] and done['stage'][1] and not done['global']:
            print('issue!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if done['stage'][0] and state['state'][1] < 1 and self.stageID == 0:
            print('issue!!!!!!!')

        if self.multiStage:
            return state, reward, done, {}
        else:
            return state, reward, done['global'], {'currentState': self.currentState.copy()}
    def reset(self):

        self.done = {'stage': [False for _ in range(self.numStages)], 'global': False}
        self.epiCount += 1

        if self.epiCount < 500:
            if random.random() < 0.5:
                self.stageID = 0
            else:
                self.stageID = 1
                self.done['stage'][0] = True
        else:
            self.stageID = 0
        print('start:', self.stageID)
        self.stepCount = 0

        if self.stageID == 0:
            self.currentState = np.array([0.0, 0.0])
        else:
            self.currentState = np.array([0.0, self.mapWidth / 2 + 1])

        self.targetState = np.array([self.mapHeight - 1.0, self.mapWidth - 1.0] )

        # self.stepCount / self.timeScale
        state = {'stageID': self.stageID,
                 'state': np.array([self.currentState[0] / self.lengthScale, \
                                    self.currentState[1] / self.lengthScale])
                 }
        return state

    def is_on_obstacle(self, i, j):
        return not (0 <= i < self.mapHeight and 0 <= j < self.mapWidth)

    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass