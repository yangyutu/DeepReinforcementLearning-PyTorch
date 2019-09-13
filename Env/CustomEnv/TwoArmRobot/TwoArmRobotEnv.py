import numpy as np
import random
import matplotlib.pyplot as plt
import math


class TwoArmEnvironmentContinuous():

    def __init__(self, config=None, seed=1):

        self.config = config
        self.read_config()


        self.stepCount = 0

        # current state is the two link angle
        self.currentState = np.array([0.0, 0.0])

        self.targetState = np.array([1.2, 1.2])

        # actions are the two link angles
        self.nbActions = 2

        # observation state can include:
        # first arm endpoint position,
        # second arm endpoint position,
        # target position - first arm end position
        # target position - second arm end position
        self.stateDim = 6

        self.endStep = 200
        self.epiCount = -1
        self.randomSeed = seed

        self.armLength = np.array([1.0, 1.0])


        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.infoDict = {'reset': False, 'endBeforeDone': False, 'stepCount': 0}

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

    def read_config(self):

        self.endStep = 200
        if 'episodeLength' in self.config:
            self.endStep = self.config['episodeLength']

        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000
        self.targetThreshFlag = False

        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.finishThresh = 0.5
        self.distanceScale = 1.0
        self.actionScale = 1.0

        if 'distanceScale' in self.config:
            self.distanceScale = self.config['distanceScale']

        if 'actionScale' in self.config:
            self.actionScale = self.config['actionScale']

        if 'finishThresh' in self.config:
            self.finishThresh = self.config['finishThresh']

    def constructObservation(self):
        firstArmEndPosition = np.array([np.cos(self.currentState[0]), np.sin(self.currentState[0])]) * self.armLength[0]
        secondArmEndPosition = firstArmEndPosition + \
                               np.array([np.cos(np.sum(self.currentState)), np.sin(np.sum(self.currentState))]) * \
                               self.armLength[1]

        firstArmDistToTarget = self.targetState - firstArmEndPosition
        secondArmDistToTarget = (self.targetState - secondArmEndPosition)

        self.effectorPosition = secondArmEndPosition.copy()


        observation = np.concatenate(
            (firstArmEndPosition, secondArmEndPosition, secondArmDistToTarget/self.distanceScale))
        return observation



    def step(self, action):
        self.stepCount += 1
        self.infoDict['stepCount'] = self.stepCount
        self.currentState += action * self.actionScale
        self.currentState %= 2 * np.pi
        observation = self.constructObservation()


        reward = 0.0
        done = False
        dist = self.effectorPosition - self.targetState

        if np.linalg.norm(dist, ord=2) < self.finishThresh:
            reward = 1
            done = True

        self.infoDict['currentState'] = self.currentState.copy()
        self.infoDict['targetState'] = self.targetState.copy()
        self.infoDict['effectorPosition'] = self.effectorPosition.copy()
        return observation, reward, done, self.infoDict.copy()

    def reset(self):
        # self.infoDict['reset'] = True
        self.infoDict['stepCount'] = 0
        self.epiCount += 1
        self.stepCount = 0
        self.currentState = (np.random.rand(2) - np.array([0.5, 0.5])) * 8
        self.resetHelper()
        self.infoDict['initial state'] = self.currentState.copy()

        observation = self.constructObservation()

        return observation

    def resetHelper(self):
        # set target information

        self.targetState = self.config['targetState']
        self.currentState = self.config['currentState']

        if self.config['dynamicTargetFlag']:
            length = random.random() * np.sqrt(np.sum(np.square(self.armLength)))
            angle = random.random() * 2 * np.pi
            x = length * math.cos(angle)
            y = length * math.sin(angle)
            self.targetState = np.array([x, y])

        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * np.sum(self.armLength)
            print('target Thresh', targetThresh)

        if self.config['dynamicInitialStateFlag']:
            while True:
                x = self.targetState[0] + (random.random() - 0.5) * targetThresh
                y = self.targetState[1] + (random.random() - 0.5) * targetThresh

                if (x**2 + y**2) < np.sum(np.square(self.armLength)):
                    theta1, theta2 = self.inverseKM(x, y)
                    break
            self.currentState = np.array([theta1, theta2])

        print('current state at start: ', self.currentState)

    def inverseKM(self, x, y):

        cosTheta2 = (x ** 2 + y ** 2 - np.sum(np.square(self.armLength))) / 2.0 / np.prod(self.armLength)

        rand = random.random()
        # randomly choose the elbow up and elbow down configuration
        if rand < 0.5:
            theta2 = math.atan2(math.sqrt(1.0 - cosTheta2 ** 2), cosTheta2)
        else:
            theta2 = math.atan2(-math.sqrt(1.0 - cosTheta2 ** 2), cosTheta2)
        k1 = self.armLength[0] + self.armLength[1] * math.cos(theta2)
        k2 = self.armLength[1] * math.sin(theta2)

        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        x_forward, y_forward = self.forwardKM(theta1, theta2)

        assert abs(x_forward - x) < 1e-6
        assert abs(y_forward - y) < 1e-6

        return theta1, theta2

    def forwardKM(self, theta1, theta2):
        x = self.armLength[0] * math.cos(theta1) + self.armLength[1] * math.cos(theta1 + theta2)
        y = self.armLength[0] * math.sin(theta1) + self.armLength[1] * math.sin(theta1 + theta2)
        return x, y

    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode='human'):
        pass
