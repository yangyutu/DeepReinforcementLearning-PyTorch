from Agents.StackedDQN.StackedDQN import StackedDQNAgent
from Agents.MultistageController.MultiStageController import MultiStageStackedController
from Env.CustomEnv.MultiStageMaze.MultiStageFreeMaze import CooperativeSimpleMazeTwoD
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze, TrajRecorder
from utils.netInit import xavier_init
from Agents.DQN.DQNMultiStageUnit import DQNMultiStageUnit
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

torch.manual_seed(1)
import torch.nn.functional as F
torch.set_num_threads(1)



from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(2)

# first construct the neutral network
config = dict()

config['trainStep'] = 3000
config['epsThreshold'] = 0.5
config['epsilon_start'] = 0.5
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 500
config['episodeLength'] = 200
config['numStages'] = 2
config['targetNetUpdateStep'] = 10
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.0001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMazeLog/traj'
config['logFrequency'] = 500
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['device'] = 'cpu'
config['mapWidth'] = 5
config['mapHeight'] = 5

env = CooperativeSimpleMazeTwoD(config)
N_S = env.stateDim
N_A = env.nbActions

# def stateProcessor(state, device = 'cpu'):
#     # given a list a dictions like { 'sensor': np.array, 'target': np.array}
#     # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
#     nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)
#
#     stateList = [item['state'] for item in state if item is not None]
#     nonFinalState = torch.tensor(stateList, dtype=torch.float32, device=device)
#     return nonFinalState, nonFinalMask


agents = []

for i in range(config['numStages']):
    netParameter = dict()
    netParameter['n_feature'] = N_S
    netParameter['n_hidden'] = [128]
    netParameter['n_output'] = N_A

    nPeriods = config['numStages']

    policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                        netParameter['n_hidden'],
                                        netParameter['n_output'])

    targetNet = MultiLayerNetRegression(netParameter['n_feature'],
                                        netParameter['n_hidden'],
                                        netParameter['n_output'])
    optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


    agent = DQNMultiStageUnit(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A, stateProcessor=None)
    agents.append(agent)


controller = MultiStageStackedController(config, agents, env)
controller.train()