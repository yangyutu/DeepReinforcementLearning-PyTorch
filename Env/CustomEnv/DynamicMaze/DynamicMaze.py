#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:59:17 2019

@author: yangyutu123
"""

from Env.CustomEnv.DynamicMaze.CircleObs import CircleObs
import numpy as np
import math
import matplotlib.pyplot as plt
import ghalton
import random
from copy import deepcopy
#

np.random.seed(1)

class TrajRecorder:
    def __init__(self):
        self.recorder = []
    def write_to_file(self, fileName):
        output = np.array(self.recorder)
        np.savetxt(fileName, output, fmt='%.5f', delimiter='\t')
    def push(self, epIdx, stepCount, state, action, nextState, reward, info):
        data = []
        data.append(epIdx)
        data.append(stepCount)
        data += info['currentState'].tolist()
        data += info['targetState'].tolist()
        data.append(action)
        data.append(reward)
        self.recorder.append(data)

class DetermAgent:
    def	__init__(self, config, mapMat, obsMap):
        self.config = config
        self.currentState = np.array([0, 0], dtype=np.int32)
        self.targetState = np.array(config['targetState'])
        self.receptHalfWidth = config['agentReceptHalfWidth']
        self.receptWidth = 2*self.receptHalfWidth + 1
        self.nbActions = 4
        self.stateDim = (self.receptWidth, self.receptWidth)
        self.endStep = 500
        self.stepCount = 0
        self.padding = config['obstacleMapPaddingWidth']
        self.mapMat = mapMat
        random.seed(1)
        self.obsMap = obsMap
        self.initSensor()




    def initSensor(self):
        self.sensorInfoMat = np.zeros((self.receptWidth, self.receptWidth))

    def is_on_obstacle(self, i, j):
        return self.sensorInfoMat[i, j] == 1

    def getSensorInfo(self):
    # add integer resentation of location
    #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        xstart = self.padding + self.currentState[0] - self.receptHalfWidth
        xend = xstart + self.receptWidth

        ystart = self.padding + self.currentState[1] - self.receptHalfWidth
        yend = ystart + self.receptWidth


    # use augumented obstacle matrix to check collision
        self.sensorInfoMat = self.obsMap[xstart:xend, ystart: yend]

    def getSensorInfoFromPos(self, position):
        xstart = self.padding + position[0] - self.receptHalfWidth
        xend = xstart + self.receptWidth

        ystart = self.padding + position[1] - self.receptHalfWidth
        yend = ystart + self.receptWidth

        # use augumented obstacle matrix to check collision
        return np.expand_dims(self.obsMap[xstart:xend, ystart: yend], axis = 0)

    def step(self, action):
        self.stepCount += 1
        i = self.currentState[0]
        j = self.currentState[1]
        offset = self.receptHalfWidth
        reward = 0.0
        if action == 0:
            if not self.is_on_obstacle(offset - 1, offset):
                self.currentState[0] -= 1
            else:
                # penality to hit wall
                reward = -0.1
        if action == 1:
            if not self.is_on_obstacle(offset + 1, offset):
                self.currentState[0] += 1
            else:
                reward = -0.1
        if action == 2:
            if not self.is_on_obstacle(offset, offset - 1):
                self.currentState[1] -= 1
            else:
                reward = -0.1
        if action == 3:
            if not self.is_on_obstacle(offset, offset + 1):
                self.currentState[1] += 1
            else:
                reward = -0.1

        if self.config['dynamicTargetFlag'] and self.stepCount % self.config['targetMoveFreq'] == 0:
            move = random.randint(0, 3)
            i = self.targetState[0] + self.padding
            j = self.targetState[1] + self.padding
            if move == 0 and self.obsMap[i - 1, j] == 0:
                self.targetState[0] -= 1
            if move == 1 and self.obsMap[i + 1, j] == 0:
                self.targetState[0] += 1
            if move == 2 and self.obsMap[i, j - 1] == 0:
                self.targetState[1] -= 1
            if move == 3 and self.obsMap[i, j + 1] == 0:
                self.targetState[1] += 1


        if np.array_equal(self.currentState, self.targetState):
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        if self.stepCount > self.endStep:
            done = True
            reward = 0.0

        # update sensor information
        self.getSensorInfo()
        distance = self.targetState - self.currentState
 #       combineState = np.concatenate((self.sensorInfoArray, distance))
        info = {'currentState': np.array(self.currentState),
                'targetState': np.array(self.targetState)}
        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis = 0),
                         'target': np.array(distance) }
        return combinedState, reward, done, info



    def reset(self):
        self.stepCount = 0
        while True:
            index = random.randint(0, self.mapMat.size - 1)
            col = index % self.mapMat.shape[1]
            row = index // self.mapMat.shape[1]
            if self.mapMat[row, col] == 0:
                break
        self.currentState = np.array([row, col], dtype=np.int32)
        self.targetState = np.array(self.config['targetState'], dtype=np.int32)

        if self.config['dynamicTargetFlag']:
            while True:
                index = random.randint(0, self.mapMat.size - 1)
                col = index % self.mapMat.shape[1]
                row = index // self.mapMat.shape[1]
                if self.mapMat[row, col] == 0:
                    break
            self.targetState = np.array([row, col], dtype=np.int32)

        # update sensor information
        self.getSensorInfo()
        distance = self.targetState - self.currentState
        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array(distance)}

        return combinedState

class StochAgent(DetermAgent):
    def __init__(self, config, mapMat, obsMap, seed = 1):
        super(StochAgent, self).__init__(config, mapMat, obsMap)

        self.config = config
        self.nbActions = 2 # on and off

        self.Dr = 0.161
        self.Dt = 2.145e-13
        self.tau = 1 / self.Dr  # tau about 6.211180124223603
        self.a = 1e-6
        self.Tc = 0.1 * self.tau # Tc is control interval
        self.v = 2 * self.a / self.Tc
        self.angleStd = math.sqrt(2*self.Tc*self.Dr)
        self.xyStd = math.sqrt(2 * self.Tc * self.Dt) / self.a

        self.jumpMat = np.load(self.config['JumpMatrix'])['jm']
        #self.jumpMat = np.genfromtxt('trajSampleAll.txt')
        self.currentState = np.array([0.0, 0.0, 0.0])
        self.constructSensorArrayIndex()
        self.epiCount = -1

        # generate a target-particle contraint

        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.thresh_by_episode = lambda step: self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

        self.config['targetThreshFlag'] = False
        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        self.scaleFactor = 1.0
        if 'scaleFactor' in self.config:
            self.scaleFactor = self.config['scaleFactor']

        self.targetClipLength = 2*self.receptHalfWidth
        self.targetClipMap = lambda x: min(x, self.targetClipLength)

        self.stochMoveFlag = False
        if 'stochMoveFlag' in self.config:
            self.stochMoveFlag = self.config['stochMoveFlag']

        self.randomSeed = seed
        np.random.seed(self.randomSeed)
        random.seed(self.randomSeed)

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

    #def is_on_obstacle(self, i, j):
    #    return self.sensorInfoArray[self.sensorMap[(i, j)]] == 1


    def step(self, action):

        reward = 0.0
        rewardScale = 40.0
        if action == 1:
            # enforce deterministic
            if not self.stochMoveFlag:
                jmRaw = np.array([2.0, 0, 0], dtype=np.float32)
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

        else:
            jm = np.array([0.0, 0.0, jmRaw[2]], dtype=np.float32)
            # penality to hit wall
            reward -= 5 / rewardScale

        #print(action)
        #print(jm)
        # update current state using modified jump matrix
        self.currentState += jm
        # make sure orientation within 0 to 2pi
        self.currentState[2] = (self.currentState[2] + 2 * np.pi) % (2 * np.pi)

        #if action == 0:
            # angle phi will update randomly
        #    self.currentState[2] += random.gauss(0, self.angleStd)

        if self.config['dynamicTargetFlag'] and self.stepCount % self.config['targetMoveFreq'] == 0:
            move = random.randint(0, 3)
            i = self.targetState[0] + self.padding
            j = self.targetState[1] + self.padding
            if move == 0 and self.obsMap[i - 1, j] == 0:
                self.targetState[0] -= 1
            if move == 1 and self.obsMap[i + 1, j] == 0:
                self.targetState[0] += 1
            if move == 2 and self.obsMap[i, j - 1] == 0:
                self.targetState[1] -= 1
            if move == 3 and self.obsMap[i, j + 1] == 0:
                self.targetState[1] += 1

        distance = self.targetState - self.currentState[0:2]

        normDist = np.linalg.norm(distance, ord=2)
        done = False

        # rewards for achieving smaller distance
        if self.minDistSoFar > (normDist + 1):
            reward += (self.minDistSoFar - normDist) /rewardScale
            self.minDistSoFar = normDist

        if self.is_terminal(distance):
            reward += 40.0 / rewardScale
            done = True
        else:
            #reward -= 0.1 # penality for slowness
            done = False

        #if self.stepCount > self.endStep:
        #    done = True
        #    reward = 0.0

        # update sensor information
        self.getSensorInfo()
        # update step count
        self.stepCount += 1

        # distance will be changed from lab coordinate to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        dx = self.targetClipMap(dx) if dx > 0 else -self.targetClipMap(-dx)
        dy = self.targetClipMap(dy) if dy > 0 else -self.targetClipMap(-dy)

        info = {'currentState': np.array(self.currentState),
                'targetState': np.array(self.targetState)}
        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor])}
        return combinedState, reward, done, info

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < 2.0

    def reset_helper(self):

        if self.config['dynamicInitialStateFlag']:
            while True:
                index = random.randint(0, self.mapMat.size - 1)
                col = index % self.mapMat.shape[1]
                row = index % self.mapMat.shape[0]
                if self.mapMat[row, col] == 0:
                    break
            # set initial state
            self.currentState = np.array([row, col, random.random()*2*math.pi], dtype=np.float32)

        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)

        # set target information
        if self.config['dynamicTargetFlag']:
            while True:
                index = random.randint(0, self.mapMat.size - 1)
                col = index % self.mapMat.shape[1]
                row = index % self.mapMat.shape[0]
                distanctVec = np.array([row, col], dtype=np.float32) - self.currentState[0:2]
                distance = np.linalg.norm(distanctVec, ord=np.inf)
                if self.mapMat[row, col] == 0 and distance < targetThresh and not self.is_terminal(distanctVec):
                    break
            self.targetState = np.array([row, col], dtype=np.int32)

    def reset(self):
        self.stepCount = 0

        self.epiCount += 1
        # store random jump for this episode
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

        dx = self.targetClipMap(dx) if dx > 0 else -self.targetClipMap(-dx)
        dy = self.targetClipMap(dy) if dy > 0 else -self.targetClipMap(-dy)


        #angleDistance = math.atan2(distance[1], distance[0]) - self.currentState[2]
        combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array([dx / self.scaleFactor, dy / self.scaleFactor])}
        return combinedState

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        # the memo dict, where id-to-object correspondence is kept to reconstruct
        # complex object graphs perfectly
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class DynamicMaze:
    def __init__(self, config, seed = 1):
        self.config = config
        self.readMaze(config['mapName'])
        if self.config['dynamicObsFlag']:
            self.generateCircleObs()
        self.initObsMat()

        self.agent = DetermAgent(config, self.mapMat, self.obsMat)
        if self.config['stochAgent']:
            self.agent = StochAgent(config, self.mapMat, self.obsMat, seed)

        self.nbActions = self.agent.nbActions
        self.stateDim = self.agent.stateDim

    def readMaze(self, fileName):
        self.mapMat = np.genfromtxt(fileName + '.txt')
        self.mapShape = self.mapMat.shape
        
    def step(self, action):
        if self.config['dynamicObsFlag']:
            self.circObs.step(1)
        return self.agent.step(action)

    def renderMap(self):
        self.fig, self.ax = plt.subplots(figsize= (10, 8))
        im = self.ax.imshow(self.mapMat,interpolation='bilinear')
        
    def renderMapAndObs(self, filename):
        plt.close()
        self.renderMap()
        if self.config['dynamicObsFlag']:
            pos = self.circObs.getPos()
            self.ax.scatter(pos[:,0], pos[:,1],marker='o',c='r',s = 5)
        plt.savefig(filename)
    
    def renderAll(self): pass
    def generateCircleObs(self):
        self.numCircObs = self.config['numCircObs']
        N = 800
        margin = 8
        
        sequencer = ghalton.Halton(2)
        randSeq = np.array(sequencer.get(N))
        centerX = (self.mapShape[1] - 2*margin)*randSeq[:,0] + margin
        centerY = (self.mapShape[0] - 2*margin)*randSeq[:,1] + margin
        
        center = np.stack((centerX, centerY), axis = 1)
        initPhi = np.zeros((N,))
        speed = (np.random.rand(N) - 0.5)

        radius = (np.random.rand(N))*4+2
        obs = CircleObs(center, initPhi, radius, speed)
        goodObsIdx = obs.overlapTest(self.mapMat)
        obs.keepIdx(goodObsIdx[0])
        self.circObs = obs
    def reset(self):
        if self.config['dynamicObsFlag']:
            self.circObs.reset()
        return self.agent.reset()

    def initObsMat(self):
        if self.config['dynamicObsFlag']:
            raise NotImplementedError
        padW = self.config['obstacleMapPaddingWidth']
        obsMapSizeOne = self.mapMat.shape[0] + 2*padW
        obsMapSizeTwo = self.mapMat.shape[1] + 2*padW
        self.obsMat = np.ones((obsMapSizeOne, obsMapSizeTwo))
        self.obsMat[padW:-padW, padW:-padW] = self.mapMat
        np.savetxt(self.config['mapName']+'obsMap.txt', self.obsMat, fmt='%d', delimiter='\t')

    def updateObsMat(self):
        if self.config['dynamicObsFlag']:
            raise NotImplementedError

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
#the memo dict, where id-to-object correspondence is kept to reconstruct
#complex object graphs perfectly
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result