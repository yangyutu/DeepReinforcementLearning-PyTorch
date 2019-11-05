import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from Env.CustomEnv.ThreeDNavigation.NavigationExamples.Obstacles.CurveVessels.curvedVessel import CurvedVessel
import json
height = 500
radius = 50
thresh = 0.5
numObs = 50
R1 = 50
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

        self.volume = 4.0 / 3.0 * np.pi * 4.0 ** 2 * (self.scale) ** 3 * 1.5 * 1.03
        self.generateKeyPoints()
        self.curvedVessel = CurvedVessel(R1=R1)

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
        return not np.any(self.curvedVessel.isOutsideVec(self.keyPoints))


class MonteCarloSimulation:
    def __init__(self):
        self.radius = 4.0

    def generateInitialConfig(self):
        x = np.arange(-R1 + 8.5, R1 - 8.5, 18.5)
        y = np.arange(-R1 + 8.5, R1 - 8.5, 18.5)
        z = np.arange(3, height - 3, 8)
        [X, Y, Z] = np.meshgrid(x, y, z)
        candidateCenters = np.array([X.flatten(), Y.flatten(), Z.flatten()], dtype=np.float32).T

        scales = np.random.rand(len(candidateCenters)) * 0.5 + 1.5
        self.centers = []
        self.scales = []
        self.ellipsoids = []
        for center, scale in zip(candidateCenters, scales):
            ellipsoid = Ellipsoid(center, scale, np.array([0, 0, 1.0]))
            if ellipsoid.isInTube():
                self.ellipsoids.append(ellipsoid)

                self.centers.append(center)
                self.scales.append(scale)

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
vol = 0
for i in range(MC.numObjects):
    vol += MC.ellipsoids[i].volume

print(vol)
print(np.pi * radius**2 * height)
print(vol/(np.pi * radius**2 * height))

MC.simulate(100)
MC.output()



output = {'numObstacles':MC.numObjects}
output['heightRadius'] = [height, R1]
output['fractions'] = [vol, np.pi * radius**2 * height]

for i in range(MC.numObjects):
    e = MC.ellipsoids[i]
    output['obs' + str(i)] = {'center': e.center.tolist(), 'orient': e.orient.tolist(), 'scale': e.scale}

with open('config_RBC.json', 'w') as f:
    json.dump(output, f)