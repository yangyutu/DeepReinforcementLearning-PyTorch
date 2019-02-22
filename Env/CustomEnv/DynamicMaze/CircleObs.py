
import numpy as np
import math
import matplotlib.pyplot as plt
import ghalton
import random


class CircleObs:
    def __init__(self, center, initPhi, radius, speed):
        self.center = center
        self.phi = initPhi
        self.initPhi = initPhi
        self.radius = radius
        self.speed = speed
        self.numP = self.center.shape[0]

    def reset(self):
        self.phi = self.initPhi

    def step(self, timeInterval):
        self.phi += self.speed * timeInterval

    def getPos(self):
        # convert to x and y
        self.pos = np.zeros((self.numP, 2))
        self.pos[:, 0] = self.center[:, 0] + np.cos(self.phi) * self.radius
        self.pos[:, 1] = self.center[:, 1] + np.sin(self.phi) * self.radius
        return self.pos

    def overlapTest(self, mapMat):
        NStep = 100
        speed = 2 * math.pi / NStep
        overLapCount = False
        phi = 0
        for i in range(NStep):
            phi += speed
            x = self.center[:, 0] + math.cos(phi) * self.radius
            y = self.center[:, 1] + math.sin(phi) * self.radius
            x_int = x.astype(int)
            y_int = y.astype(int)
            overLapCount += mapMat[y_int, x_int]
        return np.where(overLapCount == 0)

    def keepIdx(self, index):
        self.center = self.center[index, :]
        self.phi = self.phi[index]
        self.initPhi = self.initPhi[index]
        self.radius = self.radius[index]
        self.speed = self.speed[index]
        self.numP = self.center.shape[0]
