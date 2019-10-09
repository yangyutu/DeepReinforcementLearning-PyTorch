import numpy as np

import random

import math

import json
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(4)

height = 100
radius = 50
thresh = 2
numObs = 400

height = 20
radius = 20
thresh = 3
numObs = 2


obsCount = 0

keyPoints = np.array([[0, 4, 0],
                      [0, 2.82, 2.82],
                      [0, -2.82, 2.82],
                     [0, 2.82, -2.82],
                     [0, -2.82, -2.82],
                      [0, -4, 0],

                      [0, 0, 4],

                      [0, 0, -4],

                      [1, 0, 0],

                      [-1, 0, 0],
                      [-1, 2, 0],
                      [-1, -2, 0],
                      [1, 2, 0],
                      [1, -2, 0],
                      [-1, 2, 2],
                      [-1, -2, -2],
                      [1, 2, 2],
                      [1, -2, 2],
                      [-1, -2,-2],
                      [-1, -2, 2],
                      [1, 2, -2],
                      [1, -2, -2],
                      [0, 0, 0]])

volume = 4.0/3.0 * np.pi * 4 * 4



def isInTube(points, height, radius):

    distance2Axis = np.linalg.norm(points[:, :2], axis=1)

    distanceCheck = distance2Axis < (radius - 1)

    capCheck = np.logical_and(points[:, 2] < height, points[:, 2] > 0)



    return np.all(np.logical_and(distanceCheck, capCheck))





def notCollision(keyPointsAll, newKeyPoints, thresh):

    if len(keyPointsAll) == 0:

        return True

    dist = euclidean_distances(keyPointsAll, newKeyPoints)

    dist[dist == 0] = 100

    return np.all(dist > thresh)





centers = []

orients = []

scales = []

keyPointsAll = np.empty((0, 3))

volumes = []

while obsCount < numObs:

    print("obs", obsCount)

    itera = 0

    while itera < 500:

        itera += 1

        r = random.random() * radius

        phi = random.random() * 2 * np.pi

        z = random.random() * height

        x = r * math.cos(phi)

        y = r * math.sin(phi)

        scale = random.random() + 2.0



        orientVec = np.random.randn(3)

        orientVec = orientVec / np.linalg.norm(orientVec)

        phi = math.atan2(orientVec[1], orientVec[0])

        orientVec2 = np.array([-math.sin(phi), math.cos(phi), 0])

        orientVec3 = np.cross(orientVec, orientVec2)

        localFrame = np.array([orientVec, orientVec2, orientVec3])

        keyPoints_rot = np.transpose(np.dot(localFrame, keyPoints.T))

        keyPoints_rot *= scale

        keyPoints_rot += np.array([x, y, z])



        if notCollision(keyPointsAll, keyPoints_rot, thresh):

            if isInTube(keyPoints_rot, height, radius):

                centers.append([x, y, z])

                orients.append(orientVec)

                scales.append(scale)

                keyPointsAll = np.vstack((keyPointsAll, keyPoints_rot))

                volumes.append(volume * (scale)**3)

                print("iter", itera)

                break

    obsCount += 1

output = {}



output['numObstacles'] = len(scales)


print('total cell volume', sum(volumes))
print('total tube volume', np.pi * radius**2 * height)
print('total volume fraction', sum(volumes) / (np.pi * radius**2 * height))


for i in range(len(scales)):

    output['obs' + str(i)] = {'center': centers[i], 'orient': orients[i].tolist(), 'scale': scales[i]}
with open('config_RBC.json', 'w') as f:
    json.dump(output, f)