import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import json
height = 100
radius = 50
thresh = 0.5
numObs = 50

obsCount = 0
np.random.seed(2)
random.seed(2)

# each epllisoid has volume of 500
# cylinder has volume of 785400,
# need ~ 600 particle to fill an volume of 40%
# need ~ 450 particle to fill an volume of 30%
# need ~ 300 particle to fill an volume of 20%
# need ~ 150 particle to fill an volume of 10%

class Ellipsoid:
    def __init__(self, center, scale, orient):
        self.center = center
        self.scale = scale
        self.orient = orient

        self.volume = 4.0 / 3.0 * np.pi * 4.0 ** 2 * (self.scale) ** 3
        self.generateKeyPoints()

    def generateKeyPoints(self):
        a = 1.0 * self.scale
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

    def move(self):
        self.posMove = np.random.randn(3)
        self.orientMove = np.random.randn(3) * 0.5

        self.center_old = self.center.copy()
        self.orient_old = self.orient.copy()
        self.center += self.posMove
        self.orient += self.orientMove
        self.orient /= np.linalg.norm(self.orient)

        self.generateKeyPoints()

    def moveBack(self):
        self.center = self.center_old
        self.orient = self.orient_old

        self.generateKeyPoints()

    def isOverlap(self, second):
        dist = euclidean_distances(self.keyPoints, second.keyPoints)
        dist[dist == 0] = 100
        if np.any(dist < thresh):
            return True
        else:
            return False

    def isInTube(self):
        distance2Axis = np.linalg.norm(self.keyPoints[:, :2], axis=1)
        distanceCheck = distance2Axis < radius
        capCheck = np.logical_and(self.keyPoints[:, 2] < height, self.keyPoints[:, 2] > 0)

        return np.all(np.logical_and(distanceCheck, capCheck))


class MonteCarloSimulation:
    def __init__(self):
        self.radius = 4.0

    def generateInitialConfig(self):
        x = np.arange(-radius + 8.5, radius - 8.5, 18.5)
        y = np.arange(-radius + 8.5, radius - 8.5, 18.5)
        z = np.arange(3, height - 3, 6.5)
        [X, Y, Z] = np.meshgrid(x, y, z)
        candidateCenters = np.array([X.flatten(), Y.flatten(), Z.flatten()], dtype=np.float32).T

        scales = np.random.rand(len(candidateCenters)) * 0.5 + 1.5
        distance2Axis = np.linalg.norm(candidateCenters[:, :2], axis=1)
        distanceCheck = distance2Axis < (radius - self.radius * scales)
        print(np.sum(distanceCheck))

        self.centers = candidateCenters[distanceCheck, :]
        self.scales = scales[distanceCheck]
        self.ellipsoids = []
        for center, scale in zip(self.centers, self.scales):
            self.ellipsoids.append(Ellipsoid(center, scale, np.array([0, 0, 1.0])))

        self.numObjects = len(self.scales)
        print('numObjects', self.numObjects)
        for i in range(self.numObjects):
            if self.checkOverlap(i):
                print('overlap', i)



    def checkOverlap(self, index):
        dist = euclidean_distances([self.centers[index]], self.centers)
        dist[dist == 0] = 100

        for i in range(dist.shape[1]):
            if dist[0, i] < self.radius * self.scales[index] * 3:
                if self.ellipsoids[index].isOverlap(self.ellipsoids[i]):
                    return True
        return False
    def move(self, index):

        self.ellipsoids[index].move()
        if not self.ellipsoids[index].isInTube():
            self.ellipsoids[index].moveBack()
            return
        self.centers[index] = self.ellipsoids[index].center
        if self.checkOverlap(index):
            self.ellipsoids[index].moveBack()

    def simulate(self, steps = 10):
        for i in range(steps):
            print(i)
            for j in range(self.numObjects):
                self.move(j)

    def output(self):
        for i in range(self.numObjects):
            print(self.ellipsoids[i].center, self.ellipsoids[i].orient)


MC = MonteCarloSimulation()
MC.generateInitialConfig()

MC.simulate(500)
MC.output()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
ax = Axes3D(fig)

ax.scatter(MC.centers[:,0], MC.centers[:,1],MC.centers[:,2])

vol = 0
for i in range(MC.numObjects):
    vol += MC.ellipsoids[i].volume

print(vol)
print(np.pi * radius**2 * height)

output = {'numObstacles':MC.numObjects}
for i in range(MC.numObjects):
    e = MC.ellipsoids[i]
    output['obs' + str(i)] = {'center': e.center.tolist(), 'orient': e.orient.tolist(), 'scale': e.scale}

with open('config_RBC.json', 'w') as f:
    json.dump(output, f)