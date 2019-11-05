import numpy as np
import math

class CurvedVessel:
    
    def __init__(self, k1 = 0.05, k2 = 0.02, R0 = 10, R1 = 10, midRadius = 25, length = 500, capFlag = True):
        self.k1 = k1
        self.k2 = k2
        self.R0 = R0
        self.R1 = R1
        self.midRadius = midRadius
        self.length = length
        self.capFlag = capFlag
        
    def isOutside(self, pos):
        z = pos[2]
        
        if self.capFlag:
            if z > (self.length - 1.0) or z < 1.0:
                return True
        
        y_c = self.R1 * math.cos(self.k1 * z)
        R_c = self.R0 * math.cos(self.k2 * z) + self.midRadius
        x_c = 0.0
        
        dist = math.sqrt((pos[0] - x_c)**2 + (pos[1] - y_c)**2)
        
        if dist > (R_c - 1.0):
            return True
        
        return False
    
    def isOutsideVec(self, pos):
        z = pos[:, 2]
        
        if self.capFlag:
            capResult = np.logical_or(z > (self.length - 1.0), z < 1.0)
        else:
            capResult = np.zeros(z.shape, dtype=bool)
            
        y_c = self.R1 * np.cos(self.k1 * z)
        R_c = self.R0 * np.cos(self.k2 * z) + self.midRadius
        x_c = 0.0
        
        
        dist = pos[:,0:2]
        dist[:,1] -= y_c
        distNorm = np.linalg.norm(dist, axis=1)
        
        return np.logical_or(capResult, (distNorm > (R_c - 1.0)))
        
