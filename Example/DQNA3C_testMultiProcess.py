

from Agents.DQN.DQN import DQNAgent
from Agents.DQN.DQNA3C import DQNA3CMaster, SharedAdam
from Agents.Core.MLPNet import MultiLayerNetRegression
from Agents.Core.ReplayMemory import ReplayMemory, Transition

import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)

# first construct the neutral network
config = dict()

config['dataLogFolder'] = 'Log'
config['trainStep'] = 100
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 200
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFrequency'] = 100
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['numWorkers'] = 4

env = StablizerOneD()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [4]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

optimizer = SharedAdam(policyNet.parameters(), lr=1.0)


agent = DQNA3CMaster(config, policyNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A)

agent.test_multiProcess()