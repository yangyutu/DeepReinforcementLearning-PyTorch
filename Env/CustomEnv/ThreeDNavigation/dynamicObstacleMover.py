import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math

class Ellipsoid:
    def __init__(self, center, scale, orient):
        self.center = center
        self.scale = scale
        self.orient = orient

        self.volume = 4.0 / 3.0 * np.pi * 4.0 ** 2 * (self.scale) ** 3 * 1.5 * 1.03
        self.generateKeyPoints()

    def generateKeyPoints(self):
        a = 1.5 * self.scale
        b = 4.0 * self.scale
        c = 4.0 * self.scale

        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 40)
        Theta, Phi = np.meshgrid(theta, phi)

        self.keyPoints = np.array([(a * np.sin(Theta) * np.cos(Phi)).flatten(),
                                   (b * np.sin(Theta) * np.sin(Phi)).flatten(),
                                   (c * np.cos(Theta)).flatten()]).T
        phi = math.atan2(self.orient[1], self.orient[0])
        orientVec2 = np.array([-math.sin(phi), math.cos(phi), 0])
        orientVec3 = np.cross(self.orient, orientVec2)
        localFrame = np.array([self.orient, orientVec2, orientVec3])
        keyPoints_rot = np.transpose(np.dot(localFrame, self.keyPoints.T))

        self.keyPoints = keyPoints_rot + self.center

    def move(self, translationStepSize = 1, rotationStepSize = 0.5):
        self.posMove = np.random.randn(3) * translationStepSize
        self.orientMove = np.random.randn(3) * rotationStepSize

        self.center_old = self.center.copy()
        self.orient_old = self.orient.copy()
        self.keyPoints_old = self.keyPoints.copy()
        self.center += self.posMove
        self.orient += self.orientMove
        self.orient /= np.linalg.norm(self.orient)

        self.generateKeyPoints()

    def moveBack(self):
        self.center = self.center_old
        self.orient = self.orient_old
        self.keyPoints = self.keyPoints_old


class DynamicObstacleMover:
    def __init__(self, env):

        self.env = env
        self.radius = 4.0
        self.thresh = 0.5


    def initialize(self):
        self.wallRadius = self.env.wallRadius
        self.wallHeight = self.env.wallHeight
        self.storage = []
        self.centers = []

        # generate elliposde
        self.ellipsoids = []

        for obs in self.env.obstacles:
            self.ellipsoids.append(Ellipsoid(obs.center, obs.scale, obs.orientVec))
            self.centers.append(obs.center)
        self.numObjects = len(self.ellipsoids)


    def isInTube(self, idx):
        distance2Axis = np.linalg.norm(self.ellipsoids[idx].keyPoints[:, :2], axis=1)
        distanceCheck = distance2Axis < self.wallRadius
        capCheck = np.logical_and(self.ellipsoids[idx].keyPoints[:, 2] < self.wallHeight, self.ellipsoids[idx].keyPoints[:, 2] > 0)

        return np.all(np.logical_and(distanceCheck, capCheck))

    def isOverlap(self, first, second):
        dist = euclidean_distances(first.keyPoints, second.keyPoints)
        dist[dist == 0] = 100
        if np.any(dist < self.thresh):
            return True
        else:
            return False

    def checkOverlap(self, index):
        dist = euclidean_distances([self.centers[index]], self.centers)
        dist[dist == 0] = 100

        for i in range(dist.shape[1]):
            if dist[0, i] < self.radius * 2.0 * 2:
                if self.isOverlap(self.ellipsoids[index], self.ellipsoids[i]):
                    return True
        return False

    def move(self, index, translationStepSize = 1, rotationStepSize = 0.5):

        robotCenter = self.env.currentState[0:3]
        dist = np.linalg.norm(robotCenter - self.ellipsoids[index].center)
        if dist < 4:
            return

        self.ellipsoids[index].move(translationStepSize, rotationStepSize)
        if not self.isInTube(index):
            self.ellipsoids[index].moveBack()
            return
        self.centers[index] = self.ellipsoids[index].center
        if self.checkOverlap(index):
            self.ellipsoids[index].moveBack()
            self.centers[index] = self.ellipsoids[index].center

    def resetEnvObstacles(self):
        for i in range(self.numObjects):
            self.env.obstacles[i].center = self.ellipsoids[i].center
            self.env.obstacles[i].scale = self.ellipsoids[i].scale
            self.env.obstacles[i].orientVec = self.ellipsoids[i].orient
            self.env.obstacleCenters[i] = self.ellipsoids[i].center

    def simulate(self, steps=1, translationStepSize = 1, rotationStepSize = 0.5):
        print('obs simulate')
        for i in range(steps):
            for j in range(self.numObjects):
                self.move(j, translationStepSize, rotationStepSize)

        self.resetEnvObstacles()

    def saveObstacles(self):

        for i in range(self.numObjects):
            self.storage.append([i] + self.ellipsoids[i].center.tolist() + [self.ellipsoids[i].scale] + self.ellipsoids[i].orient.tolist())



    def outputObstacles(self, fname):

        output = np.array(self.storage)
        np.savetxt(fname, output)