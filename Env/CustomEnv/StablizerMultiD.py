import gym
import numpy as np


class StablizerMultiDContinuous(gym.Env):

    def __init__(self, config = None, seed = 1):
        super(StablizerMultiDContinuous, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.Dim = self.config['Dim']
        self.nbActions = self.Dim
        self.stateDim = self.Dim
        self.randomSeed = seed
        self.endStep = 200
        if 'episodeLength' in self.config:
            self.endStep = self.config['episodeLength']

        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None

        self.finiteHorizon = False
        if 'finiteHorizon' in self.config:
            self.finiteHorizon = self.config['finiteHorizon']

        self.accerlationFlag = False
        if 'accerlationFlag' in self.config:
            self.accerlationFlag = self.config['accerlationFlag']
            self.accerlationTimeWindow = self.config['accerlationTimeWindow']
            self.accerlationFactor = 5

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']


    def updatePosition(self, action):

        if self.stepCount in self.accerlationTimeWindow:
            self.currentState += self.accerlationFactor * action

    def calActionPenalty(self, action):
        penalty = -self.actionPenalty * np.linalg.norm(self.currentState, ord=2)
        return penalty


    def step(self, action):
        self.stepCount += 1
        self.infoDict['timeStep'] = self.stepCount
        self.hindSightInfo['previousState'] = self.hindSightInfo['currentState']

        self.currentState += np.random.randn(self.Dim)*0.05

        # action is the movement amount
        if self.accerlationFlag:
            self.updatePosition(action)
        else:
            self.currentState += action

        reward = 0.0
        done = False

        penalty = self.calActionPenalty(action)

        reward += penalty

        if np.linalg.norm(self.currentState, ord=np.inf) < 0.5:
            reward = 1
            done = True
            self.infoDict['done_state'] = self.currentState.copy()

        if np.linalg.norm(self.currentState, ord=np.inf) > 5:
            reward = -1

        self.hindSightInfo['currentState'] = self.currentState.copy()
        # return state is target - current state (we set target to zero)
        if not self.finiteHorizon:
            return -self.currentState.copy(), reward, done, self.infoDict.copy()
        else:
            if self.stepCount == self.endStep:
                done = True

            combinedState = {'state': -self.currentState.copy(), 'timeStep': self.stepCount}

            return combinedState, reward, done, self.infoDict.copy()

    def reset(self):
        #self.infoDict['reset'] = True
        self.infoDict = {}
        self.hindSightInfo = {}
        self.infoDict['timeStep'] = 0
        self.stepCount = 0
        self.currentState = (np.random.rand(self.Dim) - np.array([0.5 for _ in range(self.Dim)]))*3
        self.infoDict['initial state'] = self.currentState.copy()

        self.hindSightInfo['currentState'] = self.currentState.copy()
        if not self.finiteHorizon:
            return -self.currentState.copy()
        else:
            combinedState = {'state': -self.currentState.copy(), 'timeStep': self.stepCount}
            return combinedState

    def getHindSightExperience(self, state, action, nextState, info):

        targetNew = self.hindSightInfo['currentState']
        distance = targetNew - self.hindSightInfo['previousState']

        stateNew = distance
        nextStateNew = None
        actionNew = action


        rewardNew = 1 + self.calActionPenalty(action)

        if not self.finiteHorizon:
            return stateNew, actionNew, nextStateNew, rewardNew
        else:
            combinedState = {'state': distance, 'timeStep': state['timeStep']}
            return combinedState, actionNew, nextStateNew, rewardNew

    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass

