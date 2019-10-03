
import numpy as np


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
        Height = abs(np.dot(distanceVec, self.orientVec))
        distance2Axis = np.linalg.norm((distanceVec - Height * self.orientVec), ord = 2)

        if distance2Axis < self.radius \
                and (Height - self.centralHeight - distance2Axis * self.slope) < 0.0:
            return True

        return False

