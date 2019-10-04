from Env.CustomEnv.ThreeDNavigation.ActiveParticle3DSimulatorPython import ActiveParticle3DSimulatorPython
import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys
from scipy.spatial import distance


# notes for gravity
# gravity will have unit of kT/a, then every second, the displacement is given by G D/kT
# given G = 5kT/a, every second the displacement is 5 D/a and is 5 D/a^2 in radius

# a dummy test env
class ActiveParticle3DSimulatorPythonDummy:
    def __init__(self, configName, randomSeed):
        self.currentState = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def getLocalFrame(self):
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def createInitialState(self, x, y, z, ori0, ori1, ori2):
        self.currentState = np.array([x, y, z, ori0, ori1, ori2])


class ThreeDObstacle:

    def __init__(self, center, radius, slope, centralHeight, orientVec):
        self.center = center
        self.centralHeight = centralHeight
        self.radius = radius
        self.slope = slope
        self.orientVec = orientVec

    def isInside(self, pointVec):
        # first convert
        distanceVec = pointVec - self.center
        Heights = abs(np.dot(distanceVec, self.orientVec))
        distance2Axis = np.linalg.norm((distanceVec - np.outer(Heights, self.orientVec)), axis = 1)

        return np.logical_and(distance2Axis < self.radius,(Heights - self.centralHeight - distance2Axis * self.slope) < 0.0)




class ActiveParticle3DEnv():
    def __init__(self, configName, randomSeed = 1, obstacleConstructorCallBack = None):

        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.obstacleConstructorCallBack = obstacleConstructorCallBack
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

        self.sensorPixelSize = 4
        if 'sensorPixelSize' in self.config:
            self.sensorPixelSize = self.config['sensorPixelSize']

        self.receptWidth = 2 * self.receptHalfWidth + 1
        self.targetClipLength = (2 * self.receptHalfWidth) * self.sensorPixelSize
        self.stateDim = (self.receptWidth, self.receptWidth)



        self.sensorArrayWidth = (2*self.receptHalfWidth + 1)


        self.episodeEndStep = 500
        if 'episodeLength' in self.config:
            self.episodeEndStep = self.config['episodeLength']

        self.particleType = self.config['particleType']
        typeList = ['VANILLASP','SLIDER']
        if self.particleType not in typeList:
            sys.exit('particle type not right!')

        if self.particleType == 'SLIDER':
            self.nbActions = 2
        elif self.particleType == 'VANILLASP':
            self.nbActions = 1


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

        if self.obstacleFlag:
            self.constructObstacles()

            self.wallRadius = self.config['wallRadius']
            self.wallHeight = self.config['wallHeight']

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

    def constructObstacles(self):
        self.obstacles, self.obstacleCenters = self.obstacleConstructorCallBack()


    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        z_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X, Z] = np.meshgrid(y_int, x_int, z_int)
        self.sensorIndex = np.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), axis=1)
        self.sensorPos = self.sensorIndex * self.sensorPixelSize
    def getSensorInfo(self):
    # sensor information needs to consider orientation information
    # add integer resentation of location
    #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        localFrame = self.model.getLocalFrame()
    # in local Frame, each row is the vector of the local frame
    # transform from local coordinate to global coordinate is then given by localFrame * localCood or localCood * localFrame
        localFrame.shape = (3, 3)
    # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = localFrame
        sensorGlobalPos = np.matmul(self.sensorPos, rotMatrx.T)

        sensorGlobalPos[:, 0] += self.currentState[0]
        sensorGlobalPos[:, 1] += self.currentState[1]
        sensorGlobalPos[:, 2] += self.currentState[2]

        pDist = euclidean_distances([self.currentState[0:3]], self.obstacleCenters)

        overlapVec = np.zeros(len(self.sensorIndex), dtype=np.uint8)
        for idx, dist in enumerate(pDist[0]):
            if dist < self.targetClipLength:
                overlapVec += self.obstacles[idx].isInside(sensorGlobalPos)

        overlapVec += self.outsideWall(sensorGlobalPos)

    # use augumented obstacle matrix to check collision
        self.sensorInfoMat = np.reshape(overlapVec, (self.receptWidth, self.receptWidth, self.receptWidth))

    def getHindSightExperience(self, state, action, nextState, info):

        if self.hindSightInfo['obstacle']:
            return None, None, None, None
        else:
            targetNew = self.hindSightInfo['currentState'][0: 3]
            distance = targetNew - self.hindSightInfo['previousState'][0:3]

            distanceLength = np.linalg.norm(distance, ord=2)
            distance = distance / distanceLength * min(self.targetClipLength, distanceLength)
            if self.obstacleFlag:

                stateNew = {'sensor': state['sensor'],
                         'target': np.concatenate((self.hindSightInfo['previousState'][3:], distance / self.distanceScale))}
            else:
                stateNew = np.concatenate((self.hindSightInfo['previousState'][3:], distance / self.distanceScale))

            rewardNew = 1.0
            actionNew = action
            return stateNew, actionNew, None, rewardNew


    def actionPenaltyCal(self, action):
        raise NotImplementedError
        actionNorm = np.linalg.norm(action, ord=2)
        return -self.actionPenalty * actionNorm ** 2

    def outsideWall(self, points):

        distance2Axis = np.linalg.norm(points - np.array([self.wallRadius, self.wallRadius, 0]), axis = 1)

        return np.logical_or(distance2Axis > self.wallRadius, np.logical_or(points[:,2] < 0.0, points[:,2] > self.wallHeight))


    def inObstacle(self, point):
        if self.obstacleFlag:
            pDist = euclidean_distances([point], self.obstacleCenters)

            for idx, dist in enumerate(pDist[0]):
                if dist < self.targetClipLength:
                    inObstacle = self.obstacles[idx].isInside([point])

                    if inObstacle[0]:
                        return True

            # check if outside the all
            r = math.sqrt((point[0] - self.wallRadius)**2 + (point[1] - self.wallRadius)**2)
            if r > self.wallRadius:
                return True

            if point[2] > self.wallHeight or point[2] < 0.0:
                return True

        return False



    def step(self, action):
        self.hindSightInfo['obstacle'] = False
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        self.model.step(self.nStep, action)
        self.currentState = self.model.getPositions()
        #self.currentState = self.currentState + 2.0 * np.array([action[0], action[1], 0])

        hitObs = self.inObstacle(self.currentState[0:3])
        if hitObs:
            # if hit obstacle, we move angle but not the position,
            #self.info['trapConfig'].append(self.currentState.copy())
            self.currentState[0:3] = self.hindSightInfo['previousState'][0:3]
            self.model.setInitialState(self.currentState[0], self.currentState[1], self.currentState[2],
                                          self.currentState[3], self.currentState[4], self.currentState[5])

            self.hindSightInfo['obstacle'] = True
            self.info['trapCount'] += 1

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


        # distance will be changed from lab coordinate to local coordinate
        distanceLength = np.linalg.norm(distance, ord=2)
        distance = distance / distanceLength * min( self.targetClipLength, distanceLength)

        self.info['previousTarget'] = self.info['currentTarget'].copy()
        self.info['currentTarget'] = distance.copy()

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


        if self.obstacleFlag:
            if self.config['dynamicTargetFlag']:
                while True:
                    r = random.randint(0, self.wallRadius - 1)
                    angle = random.random() * np.pi * 2
                    x = r * math.cos(angle) + self.wallRadius
                    y = r * math.sin(angle) + self.wallRadius
                    z = random.randint(0, self.wallHeight)
                    if not self.inObstacle(np.array([x, y, z])):
                        break

                self.targetState = np.array([x, y, z], dtype=np.float)

            if self.config['dynamicInitialStateFlag']:
                while True:
                    r = random.randint(0, self.wallRadius - 1)
                    angle = random.random() * np.pi * 2
                    x = r * math.cos(angle) + self.wallRadius
                    y = r * math.sin(angle) + self.wallRadius
                    z = random.randint(0, self.wallHeight)

                    distanctVec = np.array([x, y, z],
                                           dtype=np.float32) - self.targetState
                    distance = np.linalg.norm(distanctVec, ord=np.inf)
                    if distance < targetThresh and \
                            not self.inObstacle(np.array([x, y, z], dtype=np.float32)) \
                            and not self.is_terminal(distanctVec):
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
        self.info['trapCount'] = 0
        self.info['trapConfig'] = []
        self.epiCount += 1


        self.reset_helper()
        self.model.createInitialState(self.currentState[0], self.currentState[1], self.currentState[2],
                                   self.currentState[3], self.currentState[4], self.currentState[5])



        # distance will be changed from lab coordinate to local coordinate
        distance = self.targetState - self.currentState[0:3]

        distanceLength = np.linalg.norm(distance, ord=2)
        distance = distance / distanceLength * min( self.targetClipLength, distanceLength)

        self.info['currentTarget'] = distance.copy()


        if self.obstacleFlag:
            self.getSensorInfo()
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