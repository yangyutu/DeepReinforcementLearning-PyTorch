


from Agents.DQN.DQNAsynER import DQNAsynERMaster, SharedAdam
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

config['trainStep'] = 500
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


envs = []

for i in range(config['numWorkers']):
    env = StablizerOneD()
    envs.append(env)

N_S = envs[0].stateDim
N_A = envs[0].nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAsynERMaster(policyNet, targetNet, envs, optimizer, torch.nn.MSELoss(reduction='mean'), N_A, config=config)

xSet = np.linspace(-1,1,100)
policy = np.zeros_like(xSet)
for i, x in enumerate(xSet):
    policy[i] = agent.getPolicy(np.array([x]))

np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%d')

#agent.perform_random_exploration(10)
agent.train()
#storeMemory = ReplayMemory(100000)
agent.testPolicyNet(100)
#storeMemory.write_to_text('testPolicyMemory.txt')


def customPolicy(state):
    x = state[0]
    # move towards negative
    if x > 0.1:
        action = 2
    # move towards positive
    elif x < -0.1:
        action = 1
    # do not move
    else:
        action = 0
    return action
# storeMemory = ReplayMemory(100000)
# agent.perform_on_policy(100, customPolicy, storeMemory)
# storeMemory.write_to_text('performPolicyMemory.txt')
# transitions = storeMemory.fetch_all_random()
for i, x in enumerate(xSet):
    policy[i] = agent.getPolicy(np.array([x]))


np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%d')




