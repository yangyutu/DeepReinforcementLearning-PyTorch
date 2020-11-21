"""
Created on 8/11/2020

@author: Yuguang Yang yangyutu123@gmail.com
@code :  This program implemets an optoThermoSwimmerEnv
"""
import numpy as np

class OptoThermoSwimmerEnv:
    def __init__(self, config, mapMat, obsMap, seed = 1):
        self.config = config
        # initial and target state
        self.currentState = np.array([0.0, 0.0, 0.0]) # x, y, phi
        self.targetState = np.array(config['targetState'])

        # visual perception component
        self.receptHalfWidth = config['agentReceptHalfWidth']
        self.receptWidth = 2*self.receptHalfWidth + 1
        self.stateDim = (self.receptWidth, self.receptWidth)
        self.padding = config['obstacleMapPaddingWidth']
        self.mapMat = mapMat
        self.obsMap = obsMap
        self.constructSensorArrayIndex()

        # number of actions
        self.nbActions = 4
        # (v = 0, w = 0)
        # (v = 1, w = 0)
        # (v = 0, w = 1)
        # (v = 0, w = -1)

        self.endStep = 500
        self.stepCount = 0


        random.seed(1)

        self.initSensor()

        # one step velocity and rotation
        self.velocity = 1.0
        self.rotationSpeed = np.pi / 10.0

        # Bronwian motion information
        self.tau = 1 / self.Dr  # tau about 6.211180124223603
        self.a = 1e-6 # radius
        self.Tc = 0.1 * self.tau # Tc is control interval
        self.v = 2 * self.a / self.Tc
        self.angleStd = math.sqrt(2*self.Tc*self.Dr)
        self.xyStd = math.sqrt(2 * self.Tc * self.Dt) / self.a

        #self.jumpMat = np.genfromtxt('trajSampleAll.txt')
        self.epiCount = -1


        # curriculum learning parameter
        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']


        self.targetThreshFlag = False
        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        self.scaleFactor = 1.0
        if 'scaleFactor' in self.config:
            self.scaleFactor = self.config['scaleFactor']

        self.targetClipLength = 2*self.receptHalfWidth

        self.stochMoveFlag = False
        if 'stochMoveFlag' in self.config:
            self.stochMoveFlag = self.config['stochMoveFlag']

        self.hindSightER = False
        if 'hindSightER' in self.config:
            self.hindSightER = self.config['hindSightER']
        self.hindSightInfo = {}

        self.wallPenalty = 0.0
        if 'wallPenalty' in self.config:
            self.wallPenalty = self.config['wallPenalty']

        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        np.random.seed(self.randomSeed)
        random.seed(self.randomSeed)
        self.info ={}

    def initSensor(self):
        self.sensorInfoMat = np.zeros((self.receptWidth, self.receptWidth))

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

    def targetClipMap(self, x):
        return min(x, self.targetClipLength)

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

    def getHindSightExperience(self, state, action, nextState, info):
        # if hit an obstacle or if action is to keep still
        if self.hindSightInfo['obsFlag'] or action == 0:
            return None, None, None, None
        else:
            targetNew = self.hindSightInfo['currentState'][0:2]

            distance = targetNew - self.hindSightInfo['previousState'][0:2]
            phi = self.hindSightInfo['previousState'][2]

            sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])

            # distance will be changed from lab coordinate to local coordinate
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)
            timeStep = state['timeStep']
            combinedState = {'sensor': sensorInfoMat,
                             'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor]),
                             'timeStep': timeStep}

            actionNew = action
            rewardNew = 1.0
            return combinedState, actionNew, None, rewardNew

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)


    def step(self, action):
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        self.rewardScale = 1.0
        if action == 1:
            # enforce deterministic
            if not self.stochMoveFlag:
                jmRaw = np.array([2.0, 0, random.gauss(0, self.angleStd)], dtype=np.float32)
            else:
                jmRaw = self.jumpMatEpisode[self.stepCount, :]

                # for symmetric dynamics
                if random.random() < 0.5:
                    jmRaw[1] = -jmRaw[1]
                    jmRaw[2] = -jmRaw[2]



        if action == 0:
            jmRaw = np.array([random.gauss(0, self.xyStd),
                              random.gauss(0, self.xyStd),
                              random.gauss(0, self.angleStd)],dtype=np.float32)

            # enforce deterministic
            if not self.stochMoveFlag:
                jmRaw[0] = 0.0
                jmRaw[1] = 0.0
        # converting from local to lab coordinate movement
        phi = self.currentState[2]
        dx = jmRaw[0]*math.cos(phi) - jmRaw[1]*math.sin(phi)
        dy = jmRaw[0]*math.sin(phi) + jmRaw[1]*math.cos(phi)
        # check if collision will occur
        i = math.floor(self.currentState[0] + dx + 0.5) + self.padding
        j = math.floor(self.currentState[1] + dy + 0.5) + self.padding
        if self.obsMap[i, j] == 0:
            jm = np.array([dx, dy, jmRaw[2]], dtype=np.float32)
            self.hindSightInfo['obsFlag'] = False
        else:
            jm = np.array([0.0, 0.0, jmRaw[2]], dtype=np.float32)
            self.hindSightInfo['obsFlag'] = True
            # penality to hit wall
            reward -= self.wallPenalty / self.rewardScale

        # update current state using modified jump matrix
        self.currentState += jm
        # make sure orientation within 0 to 2pi
        self.currentState[2] = (self.currentState[2] + 2 * np.pi) % (2 * np.pi)
        self.hindSightInfo['currentState'] = self.currentState.copy()

        distance = self.targetState - self.currentState[0:2]

        normDist = np.linalg.norm(distance, ord=2)
        done = False

        # rewards for achieving smaller distance
        if self.minDistSoFar > (normDist + 1):
        #    reward += (self.minDistSoFar - normDist) /rewardScale
            self.minDistSoFar = normDist

        if self.is_terminal(distance):
            reward = 40.0 / self.rewardScale
            done = True
        else:
            done = False

        # update sensor information
        self.getSensorInfo()
        # update step count
        self.stepCount += 1

        # distance will be changed from lab coordinate to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        # recover the global target position after target mapping
        globalTargetX = self.currentState[0] + dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1] + dx * math.sin(phi) + dy * math.cos(phi)

        self.info['previousTarget'] = self.info['currentTarget'].copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()
        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])
        self.info['currentDistance'] = math.sqrt(dx**2 + dy**2)

        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor]),
                         'timeStep': self.stepCount}
        return combinedState, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < 2.0

    def reset_helper(self):

        # set target information
        if self.config['dynamicTargetFlag']:
            while True:
                col = random.randint(0, self.mapMat.shape[1] - 1)
                row = random.randint(0, self.mapMat.shape[0] - 1)
                if self.mapMat[row, col] == 0:
                    break
            self.targetState = np.array([row, col], dtype=np.int32)



        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)


        if self.config['dynamicInitialStateFlag']:
            while True:

                col = random.randint(0, self.mapMat.shape[1] - 1)
                row = random.randint(0, self.mapMat.shape[0] - 1)
                distanctVec = np.array([row, col], dtype=np.float32) - self.targetState
                distance = np.linalg.norm(distanctVec, ord=np.inf)
                if self.mapMat[row, col] == 0 and distance < targetThresh and not self.is_terminal(distanctVec):
                    break
            # set initial state
            self.currentState = np.array([row, col, random.random()*2*math.pi], dtype=np.float32)


    def reset(self):
        self.stepCount = 0
        self.info = {}
        self.info['scaleFactor'] = self.scaleFactor
        self.hindSightInfo = {}
        self.epiCount += 1
        # store random jump for this episode
        if self.stochMoveFlag:
            randomIdx = np.random.choice(self.jumpMat.shape[0], self.endStep + 10)
            self.jumpMatEpisode = self.jumpMat[randomIdx, :]

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.int32)

        self.reset_helper()

        # update sensor information
        self.getSensorInfo()
        distance = self.targetState - self.currentState[0:2]

        self.minDistSoFar = np.linalg.norm(distance, ord=2)
        # distance will be change to local coordinate
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

        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor]),
                         'timeStep': self.stepCount}
        return combinedState
