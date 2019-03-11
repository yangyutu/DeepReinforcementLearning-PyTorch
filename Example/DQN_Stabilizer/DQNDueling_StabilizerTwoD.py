

from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
from Agents.Core.DuelingNet import DuelingMLP
from Agents.Core.ReplayMemory import ReplayMemory, Transition

import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerTwoD import StablizerTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)

# first construct the neutral network
config = dict()

config['trainStep'] = 500
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 1000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerTwoDLog/traj'
config['logFrequency'] = 100
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'policyNet'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5

env = StablizerTwoD()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A
netParameter['dueling_size'] = 64

policyNet = DuelingMLP(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'],
                                    netParameter['dueling_size'])

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A, config=config)

xSet = np.linspace(-1,1,100)
ySet = np.linspace(-1,1,100)

policy = np.zeros((100,100))
for i, x in enumerate(xSet):
    for j, y in enumerate(xSet):
        policy[i,j] = agent.getPolicy(np.array([x, y]))

np.savetxt('StabilizerTwoPolicyBeforeTrain.txt', policy, fmt='%d')

#agent.perform_random_exploration(10)
agent.train()
agent.testPolicyNet(100)

#
# def customPolicy(state):
#     x = state[0]
#     # move towards negative
#     if x > 0.1:
#         action = 2
#     # move towards positive
#     elif x < -0.1:
#         action = 1
#     # do not move
#     else:
#         action = 0
#     return action
# # storeMemory = ReplayMemory(100000)
# # agent.perform_on_policy(100, customPolicy, storeMemory)
# # storeMemory.write_to_text('performPolicyMemory.txt')
# # transitions = storeMemory.fetch_all_random()
policy = np.zeros((100,100))
for i, x in enumerate(xSet):
    for j, y in enumerate(xSet):
        policy[i,j] = agent.getPolicy(np.array([x, y]))

np.savetxt('StabilizerTwoPolicyAfterTrain.txt', policy, fmt='%d')

#plotPolicy(xSet, policy)


