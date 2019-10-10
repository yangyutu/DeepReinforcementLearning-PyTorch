import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import math
import json
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv, RBCObstacle


class PathFinderThreeD:

    def __init__(self, config, config_RBC):
        self.config = config
        self.config_RBC = config_RBC

        self.height = self.config['height']
        self.radius = self.config['radius']

        self.generateRBC()

    def _generateRBC(self):
        obstacles, obstacleCenter = [], []

        for i in range(self.config_RBC['numObstacles']):
            name = 'obs' + str(i)
            obstacles.append(
                RBCObstacle(np.array(self.config_RBC[name]['center']), self.config_RBC[name]['scale'], np.array(self.config_RBC[name]['orient'])))

            obstacleCenter.append(obstacles[i].center)

        return obstacles, obstacleCenter

    def _generateGridPoints(self):
        numP = 2 * int(self.radius)
        x = np.linspace(-self.radius, self.radius, numP)
        y = np.linspace(-self.radius, self.radius, numP)

        numP = int(self.height)
        z = np.linspace(0, self.height, numP)
        [X, Y, Z] = np.meshgrid(x, y, z)
        self.gridPoints = np.concatenate(X.flatten(), Y.flatten(), Z.flatten())

    def _isInTube(self, points):
        distance2Axis = np.linalg.norm(points[:, :2], axis=1)
        distanceCheck = distance2Axis < self.radius
        capCheck = np.logical_and(points[:, 2] < self.height, points[:, 2] > 0)

        return np.logical_and(distanceCheck, capCheck)

    def _filterValidPoints(self):
        inTube = self._isInTube(self.gridPoints)
        self.gridPoints = self.gridPoints[inTube]

        for obs in self.obstacles:
            outsideVec = np.logical_not(obs.isInside(self.gridPoints))
            self.gridPoints = self.gridPoints[outsideVec]

    def _constructGraph(self):
        self.G = kneighbors_graph(self.gridPoints, 10, mode='distance', include_self=False)
        self.NX_G = nx.from_scipy_sparse_matrix(self.G)

    def findPath(self, startPoint, endPoint):
        dist = euclidean_distances([endPoint], self.gridPoints)
        self.targetConfigIdx = np.argmin(dist)
        dist = euclidean_distances([startPoint], self.gridPoints)
        self.startConfigIdx = np.argmin(dist)

        self.path = nx.dijkstra_path(self.NX_G, self.targetConfigIdx, self.startConfigIdx)
        pathLengthTotal = nx.dijkstra_path_length(self.NX_G, self.targetConfigIdx, self.startConfigIdx)
        pathLengthAll = nx.single_source_dijkstra_path_length(self.NX_G, self.targetConfigIdx)

        self.pathLength = [0.0]
        for i in range(1, len(self.path)):
            self.pathLength.append(pathLengthAll[self.path[i]])