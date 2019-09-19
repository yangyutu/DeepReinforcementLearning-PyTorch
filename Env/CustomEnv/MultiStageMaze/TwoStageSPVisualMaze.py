import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import copy


class TwoStageSPVisualMaze:
    def __init__(self, config, randomSeed=1):
        """
        A model take in a particle configuration and actions and return updated a particle configuration
        """

        self.config = config
        self.randomSeed = randomSeed
        self.read_config()
        self.initilize()

        # self.padding = self.config['']

    def initilize(self):
        if not os.path.exists('Traj'):
            os.makedirs('Traj')
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0

        self.numStages = 2

        self.nbActions = [2, 1]

        self.info = {}

        self.Dr = 0.161
        self.Dt = 2.145e-14
        self.tau = 1 / self.Dr  # tau about 6.211180124223603
        self.a = 1e-6
        self.Tc = 0.1 * self.tau  # Tc is control interval
        self.v = 2 * self.a / self.Tc
        self.angleStd = math.sqrt(2 * self.Tc * self.Dr)
        self.xyStd = math.sqrt(2 * self.Tc * self.Dt) / self.a

        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        self.initObsMat()
        self.constructSensorArrayIndex()
        self.epiCount = -1


    def read_config(self):

        self.receptHalfWidth = self.config['agentReceptHalfWidth']
        self.padding = self.config['obstacleMapPaddingWidth']
        self.receptWidth = 2 * self.receptHalfWidth + 1
        self.targetClipLength = 2 * self.receptHalfWidth
        self.stateDim = (self.receptWidth, self.receptWidth)

        self.sensorArrayWidth = (2 * self.receptHalfWidth + 1)

        self.endStep = 500
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

        self.obstacleFlg = True
        if 'obstacleFlag' in self.config:
            self.obstacleFlg = self.config['obstacleFlag']

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']

        self.scaleFactor = 20.0
        if 'scaleFactor' in self.config:
            self.scaleFactor = self.config['scaleFactor']

        self.obstaclePenalty = 0.0
        if 'obstaclePenalty' in self.config:
            self.obstaclePenalty = self.config['obstaclePenalty']

        self.transitionDistance = 3.0
        if 'transitionDistance' in self.config:
            self.transitionDistance = self.config['transitionDistance']


    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)

    def getHindSightExperience(self, state, action, nextState, done, info):

        # if hit an obstacle or if action is to keep still
        if self.hindSightInfo['obsFlag'] or self.stageID == 0:
            return None, None, None, None, None
        elif self.stageID == 1 and not done:
            targetNew = self.hindSightInfo['currentState'][0:2]

            distance = targetNew - self.hindSightInfo['previousState'][0:2]
            phi = self.hindSightInfo['previousState'][2]

            sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])

            # distance will be changed from lab coordinate to local coordinate
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

            if self.obstacleFlg:
                state = {'sensor': sensorInfoMat,
                         'target': np.array(
                             [dx / self.scaleFactor, dy / self.scaleFactor])}
            else:
                state = np.array([dx / self.scaleFactor, dy / self.scaleFactor])
                nextState = np.array([0.0, 0.0])

            actionNew = action
            # here [0, 0] is dummy input to ensure done is always true
            return state, actionNew, nextState, 1.0, True


    def getSensorInfo(self):
        # sensor information needs to consider orientation information
        # add integer resentation of location
        #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        phi = self.currentState[2]
        #   phi = (self.stepCount)*math.pi/4.0
        # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = np.matrix([[math.cos(phi), -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        self.sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

    def getSensorInfoFromPos(self, position):

        phi = position[2]
        #   phi = (self.stepCount)*math.pi/4.0
        # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = np.matrix([[math.cos(phi), -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(position[0] + 0.5)
        j = math.floor(position[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return np.expand_dims(sensorInfoMat, axis=0)

    def rewardCal(self):
        reward = 0.0

        # if pass the time limit and not finish give zero reward
        if self.stepCount > self.endStep:
            self.done['stage'] = [True for _ in range(self.numStages)]
            self.done['global'] = True
            print('not finish ', self.currentState, self.stageID)
            return reward, copy.deepcopy(self.done)


        distance = self.targetState - self.currentState[0:2]

        if self.is_terminal(distance):
            self.done['stage'][self.stageID] = True
            self.done['global'] = True
            reward = 1.0
            print('finish ', self.currentState, reward, self.stageID)

        return reward, copy.deepcopy(self.done)

    def step(self, action):
        self.hindSightInfo['previousState'] = self.currentState.copy()
        # update step count
        self.stepCount += 1
        # if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        jmRaw = np.array([random.gauss(0, self.xyStd) + 2.0 * action,
                          random.gauss(0, self.xyStd),
                          random.gauss(0, self.angleStd)], dtype=np.float32)





        # converting from local to lab coordinate movement
        phi = self.currentState[2]
        dx = jmRaw[0] * math.cos(phi) - jmRaw[1] * math.sin(phi)
        dy = jmRaw[0] * math.sin(phi) + jmRaw[1] * math.cos(phi)
        # check if collision will occur
        i = math.floor(self.currentState[0] + dx + 0.5) + self.padding
        j = math.floor(self.currentState[1] + dy + 0.5) + self.padding

        if self.obsMap[i, j] == 0:
            jm = np.array([dx, dy, jmRaw[2]], dtype=np.float32)
            self.hindSightInfo['obsFlag'] = False
        else:
            jm = np.array([0.0, 0.0, jmRaw[2]], dtype=np.float32)
            self.hindSightInfo['obsFlag'] = True


        # update current state using modified jump matrix
        self.currentState += jm
        # make sure orientation within 0 to 2pi
        self.currentState[2] = (self.currentState[2] + 2 * np.pi) % (2 * np.pi)
        self.hindSightInfo['currentState'] = self.currentState.copy()

        # transitions of stages
        distance = self.targetState - self.currentState[0:2]

        if self.stageID == 0 and np.linalg.norm(distance, ord=2) < self.transitionDistance:
            print('job passage', self.currentState, 'step', self.stepCount)
            self.done['stage'][self.stageID] = True
            self.stageID += 1



        reward, done = self.rewardCal()

        # distance will be changed from lab coordinate to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        globalTargetX = self.currentState[0]+ dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1]+ dx * math.sin(phi) + dy * math.cos(phi)


        # update sensor information
        if self.obstacleFlg:
            self.getSensorInfo()

        self.info['stageID'] = self.stageID
        self.info['previousTarget'] = self.targetState.copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])

        if self.obstacleFlg:
            state = {'state': {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': np.array(
                         [dx / self.scaleFactor, dy / self.scaleFactor])},
                     'stageID': self.stageID}
        else:
            state = {'stageID': self.stageID,
                     'state': np.array(
                         [dx / self.scaleFactor, dy / self.scaleFactor])}

        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < 0.5

    def reset_helper(self):



        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.float32)
        # set target information
        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)

        if self.config['dynamicInitialStateFlag']:
            while True:
                col = random.randint(0, self.mapMat.shape[1] - 1)
                row = random.randint(0, self.mapMat.shape[0] - 1)
                distanceVec = np.array([row, col], dtype=np.float32) - self.targetState
                distance = np.linalg.norm(distanceVec, ord=np.inf)


                if self.mapMat[row, col] == 0 and distance < targetThresh and not self.is_terminal(distanceVec):
                    break


            # set initial state
            print('target distance', distance)

            self.currentState = np.array([row, col, random.random() * 2 * math.pi],
                                         dtype=np.float32)

        if np.linalg.norm(distanceVec, ord=2) > self.transitionDistance:
            self.stageID = 0
        else:
            self.stageID = 1

        print('initial state, stageID', self.currentState, self.stageID)

    def reset(self):
        self.stepCount = 0
        self.hindSightInfo = {}
        self.info = {}
        self.epiCount += 1
        self.done = {'stage': [False for _ in range(self.numStages)], 'global': False}

        self.reset_helper()
        # update sensor information

        self.info['scaleFactor'] = self.scaleFactor
        self.info['stageID'] = self.stageID

        if self.obstacleFlg:
            self.getSensorInfo()

        distance = self.targetState - self.currentState[0:2]

        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        globalTargetX = self.currentState[0]+ dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1]+ dx * math.sin(phi) + dy * math.cos(phi)


        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])

        # angleDistance = math.atan2(distance[1], distance[0]) - self.currentState[2]
        if self.obstacleFlg:
            state = {'state': {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor])},
                     'stageID': self.stageID}
        else:
            state = {'stageID': self.stageID,
                     'state': np.array([dx / self.scaleFactor, dy / self.scaleFactor])}

        return state

    def initObsMat(self):
        fileName = self.config['mapName']
        self.mapMat = np.genfromtxt(fileName + '.txt')
        self.mapShape = self.mapMat.shape
        padW = self.config['obstacleMapPaddingWidth']
        obsMapSizeOne = self.mapMat.shape[0] + 2 * padW
        obsMapSizeTwo = self.mapMat.shape[1] + 2 * padW
        self.obsMap = np.ones((obsMapSizeOne, obsMapSizeTwo))
        self.obsMap[padW:-padW, padW:-padW] = self.mapMat
        np.savetxt(self.config['mapName'] + 'obsMap.txt', self.obsMap, fmt='%d', delimiter='\t')