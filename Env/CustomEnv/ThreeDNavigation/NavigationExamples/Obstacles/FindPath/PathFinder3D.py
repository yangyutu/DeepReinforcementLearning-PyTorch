import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import collections
import math
import json
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv, RBCObstacle


class PathFinderThreeD:

    def __init__(self, config_RBC, res = 2):
        self.config_RBC = config_RBC
        self.resolution = res
        self.height, self.radius = self.config_RBC['heightRadius']
        self.nextMoveCache = {}
        self._generateRBC()

        self._generateGridPoints()

        self._filterValidPoints()

        self._constructGraph()
    def _generateRBC(self):
        self.obstacles, self.obstacleCenter = [], []

        for i in range(self.config_RBC['numObstacles']):
            name = 'obs' + str(i)
            self.obstacles.append(
                RBCObstacle(np.array(self.config_RBC[name]['center']), self.config_RBC[name]['scale'], np.array(self.config_RBC[name]['orient'])))

            self.obstacleCenter.append(self.obstacles[i].center)

    def _generateGridPoints(self):
        numP = int(2 * self.radius / self.resolution)
        x = np.linspace(-self.radius, self.radius, numP)
        y = np.linspace(-self.radius, self.radius, numP)

        numP = int(self.height / self.resolution )
        z = np.linspace(0, self.height, numP)
        [X, Y, Z] = np.meshgrid(x, y, z)
        self.gridPoints = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    def _isInTube(self, points):
        distance2Axis = np.linalg.norm(points[:, :2], axis=1)
        distanceCheck = distance2Axis < (self.radius - 1)
        capCheck = np.logical_and(points[:, 2] < self.height, points[:, 2] > 0)

        return np.logical_and(distanceCheck, capCheck)

    def _filterValidPoints(self):
        inTube = self._isInTube(self.gridPoints)
        self.gridPoints = self.gridPoints[inTube]

        for obs in self.obstacles:
            outsideVec = np.logical_not(obs.isInside(self.gridPoints))
            self.gridPoints = self.gridPoints[outsideVec]

    def _constructGraph(self):
        self.G = kneighbors_graph(self.gridPoints, 26, mode='distance', include_self=False)
        self.NX_G = nx.from_scipy_sparse_matrix(self.G)

    def findPath(self, startPoint, endPoint):
        dist = euclidean_distances([endPoint], self.gridPoints)
        self.targetConfigIdx = np.argmin(dist)
        dist = euclidean_distances([startPoint], self.gridPoints)
        self.startConfigIdx = np.argmin(dist)

        self.path = nx.dijkstra_path(self.NX_G, self.targetConfigIdx, self.startConfigIdx)
        pathLengthTotal = nx.dijkstra_path_length(self.NX_G, self.targetConfigIdx, self.startConfigIdx)
        pathLengthAll = nx.single_source_dijkstra_path_length(self.NX_G, self.targetConfigIdx)

        pathCoordinates = []
        for i in range(1, len(self.path)):
            pathCoordinates.append(self.gridPoints[self.path[i]])

        return pathLengthTotal, pathCoordinates

    def findPath_SingleTarget(self, startPoints, endPoint):
        dist = euclidean_distances([endPoint], self.gridPoints)
        self.targetConfigIdx = np.argmin(dist)
        dist = euclidean_distances(startPoints, self.gridPoints)
        self.startConfigIdx = np.argmin(dist, axis=1)

        allLength = dict(nx.single_source_bellman_ford_path_length(self.NX_G, self.targetConfigIdx))

        queryLength = [allLength[i] for i in self.startConfigIdx]

        return queryLength

    def fillCache(self, endPoint):
        dist = euclidean_distances([endPoint], self.gridPoints)
        self.targetConfigIdx = np.argmin(dist)

        allPaths = nx.single_source_shortest_path(self.NX_G, self.targetConfigIdx)

        self.nextMove = {}
        for k, v in allPaths.items():
            if len(v) > 1:
                self.nextMove[k] = v[-2]
            else:
                self.nextMove[k] = v[-1]

    def getNextMove(self, startPoint):
        startPoint = startPoint.astype(np.int)
        startPointTuple = tuple(startPoint.tolist())
        if startPointTuple in self.nextMoveCache:
            startConfigIdx = self.nextMoveCache[startPointTuple]
            nextMoveIdx = self.nextMove[startConfigIdx]
        else:
            dist = euclidean_distances([startPoint], self.gridPoints)
            startConfigIdx = np.argmin(dist, axis=1)[0]
            self.nextMoveCache[startPointTuple] = startConfigIdx
            nextMoveIdx = self.nextMove[startConfigIdx]
        return self.gridPoints[nextMoveIdx]
