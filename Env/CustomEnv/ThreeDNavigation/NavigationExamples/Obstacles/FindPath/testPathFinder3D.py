from Agents.DDPG.DDPG import DDPGAgent
from Env.CustomEnv.StablizerOneD import StablizerOneDContinuous
from utils.netInit import xavier_init
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.OUNoise import OUNoise
from activeParticleEnv import ActiveParticleEnvMultiMap, ActiveParticleEnv
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv, RBCObstacle
from Env.CustomEnv.ThreeDNavigation.NavigationExamples.Obstacles.FindPath.PathFinder3D import PathFinderThreeD


import math
torch.manual_seed(1)

configName = 'config_RBC_R50_5PerTest.json'
with open(configName,'r') as f:
    config_RBC = json.load(f)

finder = PathFinderThreeD(config_RBC)

start = np.array([0, 0, 1])
end = np.array([0, 0, 500])

pathLengthTotal, pathCoordinates = finder.findPath(start, end)

print(pathLengthTotal)

print(pathCoordinates)

#pathLength2 = finder.findPath_SingleTarget([start], end)
#print(pathLength2)