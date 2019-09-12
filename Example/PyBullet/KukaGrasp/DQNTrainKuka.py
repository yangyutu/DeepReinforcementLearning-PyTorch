

from Agents.DQN.DQN import DQNAgent
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

config['trainStep'] = 10000
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 50000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5


import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

env = KukaGymEnv(renders=True, isDiscrete=True)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A)

agent.train()

