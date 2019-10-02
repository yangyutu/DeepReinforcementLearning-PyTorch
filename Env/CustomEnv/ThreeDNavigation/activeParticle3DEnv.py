from Env.CustomEnv.ThreeDNavigation.ActiveParticle3DSimulatorPython import ActiveParticle3DSimulatorPython
import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys


class ActiveParticle3DEnv():
    def __init__(self, configName, randomSeed = 1):

        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.model = ActiveParticle3DSimulatorPython(configName, randomSeed)
        self.read_config()
        self.initilize()

        #self.padding = self.config['']

    def initilize(self):
        if not os.path.exists('Traj'):
            os.makedirs('Traj')
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0

        self.info = {}

        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        self.initObsMat()
        self.constructSensorArrayIndex()
        self.epiCount = -1

    def read_config(self):

        self.receptHalfWidth = self.config['receptHalfWidth']
        self.padding = self.config['obstacleMapPaddingWidth']
        self.receptWidth = 2 * self.receptHalfWidth + 1
        self.targetClipLength = 2 * self.receptHalfWidth
        self.stateDim = (self.receptWidth, self.receptWidth)

        self.sensorArrayWidth = (2*self.receptHalfWidth + 1)


        self.episodeEndStep = 500
        if 'episodeLength' in self.config:
            self.episodeEndStep = self.config['episodeLength']

        self.nbActions = 2

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

        self.obstacleFlag = False
        if 'obstacleFlag' in self.config:
            self.obstacleFlag = self.config['obstacleFlag']

        self.nStep = self.config['modelNStep']

        self.distanceScale = 20
        if 'distanceScale' in self.config:
            self.distanceScale = self.config['distanceScale']

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']

        self.obstaclePenalty = 0.0
        if 'obstaclePenalty' in self.config:
            self.obstaclePenalty = self.config['obstaclePenalty']

        self.finishThresh = 5.0
        if 'finishThresh' in self.config:
            self.finishThresh = self.config['finishThresh']

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)
    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)

    def getSensorInfo(self):
    # sensor information needs to consider orientation information
    # add integer resentation of location
    #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        phi = self.currentState[2]
    #   phi = (self.stepCount)*math.pi/4.0
    # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = np.matrix([[math.cos(phi),  -math.sin(phi)],
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

        rotMatrx = np.matrix([[math.cos(phi),  -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(position[0] + 0.5)
        j = math.floor(position[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return np.expand_dims(sensorInfoMat, axis = 0)

    def getExperienceAugmentation(self, state, action, nextState, reward, info):
        raise NotImplementedError
        if self.timingFlag:
            raise Exception('timing for experience Augmentation case is not implemented!')

        state_Aug, action_Aug, nextState_Aug, reward_Aug = [], [], [], []
        if not self.obstacleFlag:
            if self.particleType == 'FULLCONTROL':
                # state is the position of target in the local frame
                # here uses the mirror relation
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([action[0], -action[1]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
            elif self.particleType == 'SLIDER':
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([-action[0]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
            elif self.particleType == 'VANILLASP':
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([action[0]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
        return state_Aug, action_Aug, nextState_Aug, reward_Aug

    def getHindSightExperience(self, state, action, nextState, info):

        if self.hindSightInfo['obstacle']:
            return None, None, None, None
        else:
            targetNew = self.hindSightInfo['currentState'][0: 3]
            distance = targetNew - self.hindSightInfo['previousState'][0:3]

            distanceLength = np.linalg.norm(distance, ord=2)
            distance = distance / distanceLength * min(self.targetClipLength, distanceLength)
            if self.obstacleFlag:
                sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])

                stateNew = {'sensor': sensorInfoMat,
                         'target': np.concatenate((self.hindSightInfo['previousState'][3:], distance / self.distanceScale))}
            else:
                stateNew = np.concatenate((self.hindSightInfo['previousState'][3:], distance / self.distanceScale))

            rewardNew = 1.0
            actionNew = action
            return stateNew, actionNew, None, rewardNew

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
        # sensormap maps a location (x, y) to to an index. for example (-5, -5) to 0
        # self.sensorMap = {}
        # for i, x in enumerate(x_int):
        #     for j, y in enumerate(y_int):
        #         self.sensorMap[(x, y)] = i * self.receptWidth + j


    def actionPenaltyCal(self, action):
        raise NotImplementedError
        actionNorm = np.linalg.norm(action, ord=2)
        return -self.actionPenalty * actionNorm ** 2

    def obstaclePenaltyCal(self):
        raise NotImplementedError

        if self.obstacleFlag and not self.dynamicObstacleFlag:
            i = math.floor(self.currentState[0] + 0.5)
            j = math.floor(self.currentState[1] + 0.5)

            xIdx = self.padding + i
            yIdx = self.padding + j

            if self.obsMap[xIdx, yIdx] > 0:
                return -self.obstaclePenalty, True
            else:
                return 0, False
        if self.obstacleFlag and self.dynamicObstacleFlag:
            trapFlag = self.model.checkDynamicTrap()
            if trapFlag:
                self.info['dynamicTrap'] += 1
                return -self.obstaclePenalty, True
            else:

                return 0, False

    def step(self, action):
        self.hindSightInfo['obstacle'] = False
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        self.model.step(self.nStep, action)
        self.currentState = self.model.getPositions()
        #self.currentState = self.currentState + 2.0 * np.array([action[0], action[1], 0])

        self.hindSightInfo['currentState'] = self.currentState.copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()
        distance = self.targetState - self.currentState[0:3]

        # update step count
        self.stepCount += 1

        done = False

        if self.is_terminal(distance):
            reward = 1.0
            done = True


        # penalty for actions
#        reward += self.actionPenaltyCal(action)

        # # update sensor information
        # if self.obstacleFlag:
        #     if not self.dynamicObstacleFlag:
        #         self.getSensorInfo()
        #     else:
        #         self.getSequenceSensorInfo()
        #     penalty, flag = self.obstaclePenaltyCal()
        #     reward += penalty
        #     if flag:
        #         self.hindSightInfo['obstacle'] = True
        #         self.currentState = self.hindSightInfo['previousState'].copy()
        #         #if self.dynamicObstacleFlag:
        #         #    done = True


        # distance will be changed from lab coordinate to local coordinate
        distanceLength = np.linalg.norm(distance, ord=2)
        distance = distance / distanceLength * min( self.targetClipLength, distanceLength)
        if self.obstacleFlag:
            state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': np.concatenate((self.currentState[3:], distance / self.distanceScale))}
        else:
            state = np.concatenate((self.currentState[3:], distance / self.distanceScale))
        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < self.finishThresh

    def reset_helper(self):
        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * 100
            print('target thresh', targetThresh)
        self.currentState = np.array(self.config['currentState'], dtype=np.float)
        self.targetState = np.array(self.config['targetState'], dtype=np.float)


        if not self.obstacleFlag:
            if self.config['dynamicTargetFlag']:
                x = random.randint(0, 50)
                y = random.randint(0, 50)
                z = random.randint(0, 50)
                self.targetState = np.array([x, y, z], dtype=np.float)

            if self.config['dynamicInitialStateFlag']:
                while True:
                    x = random.randint(0, 50)
                    y = random.randint(0, 50)
                    z = random.randint(0, 50)

                    distanctVec = np.array([x, y, z],
                                           dtype=np.float32) - self.targetState
                    distance = np.linalg.norm(distanctVec, ord=np.inf)
                    if distance < targetThresh and not self.is_terminal(distanctVec):
                        break
                # set initial state
                print('target distance', distance)
                orientation = np.random.randn(3)
                self.currentState = np.concatenate((np.array([x, y, z], dtype=np.float32), orientation))

    def reset(self):
        self.stepCount = 0

        self.hindSightInfo = {}

        self.info = {}
        self.info['scaleFactor'] = self.distanceScale
        self.epiCount += 1


        self.reset_helper()
        self.model.createInitialState(self.currentState[0], self.currentState[1], self.currentState[2],
                                   self.currentState[3], self.currentState[4], self.currentState[5])

        # distance will be changed from lab coordinate to local coordinate
        distance = self.targetState - self.currentState[0:3]

        distanceLength = np.linalg.norm(distance, ord=2)
        distance = distance / distanceLength * min( self.targetClipLength, distanceLength)
        if self.obstacleFlag:
            state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': np.concatenate((self.currentState[3:], distance / self.distanceScale))}
        else:
            state = np.concatenate((self.currentState[3:], distance / self.distanceScale))
        return state

    def initObsMat(self):
        return
        # fileName = self.config['mapName']
        # self.mapMat = np.genfromtxt(fileName + '.txt')
        # self.mapShape = self.mapMat.shape
        # padW = self.config['obstacleMapPaddingWidth']
        # obsMapSizeOne = self.mapMat.shape[0] + 2*padW
        # obsMapSizeTwo = self.mapMat.shape[1] + 2*padW
        # self.obsMap = np.ones((obsMapSizeOne, obsMapSizeTwo))
        # self.obsMap[padW:-padW, padW:-padW] = self.mapMat
        #
        # self.obsMap -= 0.5
        # self.mapMat -= 0.5
        # np.savetxt(self.config['mapName']+'obsMap.txt', self.obsMap, fmt='%.1f', delimiter='\t')