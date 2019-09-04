

from Agents.DQN.DQN import DQNAgent
from Env.CustomEnv.MultiStageMaze.MultiStageMixedMaze import MultiStageMixedMaze
from utils.netInit import xavier_init
from Agents.DQN.DQNA3C import DQNA3CMaster, SharedAdam
import json
from torch import optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import os
import random


torch.manual_seed(1)
import torch.nn.functional as F
torch.set_num_threads(1)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
    
env = MultiStageMixedMaze(config)

N_S = env.stateDim[0]
N_A = env.nbActions

nSteps = 100

state = env.reset()
print(state)

for i in range(nSteps):
    print(i)

    if env.stageID == 1:
        action = random.randint(0, 1)
    elif env.stageID == 0:
        action = random.random() - 0.5
    print(action)
    state, reward, done, info = env.step(action)
    print(info)