

from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)

def plotPolicy(policy, nbActions):
    idx, idy = np.where(policy >=0)
    action = policy[idx,idy]
    plt.scatter(idx, idy, c = action, marker='s', s = 10)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )


# first construct the neutral network
config = dict()

mapName = 'map.txt'

config['trainStep'] = 200
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 10
config['memoryCapacity'] = 1000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMazeLog/traj' + mapName
config['logFrequency'] = 50
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'targetNet'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5

env = SimpleMazeTwoD(mapName)
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

print(policyNet.state_dict())

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction = 'none') , N_A, config=config)

policy = deepcopy(env.map)
for i in range(policy.shape[0]):
    for j in range(policy.shape[1]):
        if env.map[i, j] == 0:
            policy[i, j] = -1
        else:
            policy[i, j] = agent.getPolicy(np.array([i, j]))


np.savetxt('DoubleQSimpleMazePolicyBeforeTrain' + mapName + '.txt', policy, fmt='%d', delimiter='\t')

plotPolicy(policy, N_A)

agent.train()

policy = deepcopy(env.map)
for i in range(policy.shape[0]):
    for j in range(policy.shape[1]):
        if env.map[i, j] == 0:
            policy[i, j] = -1
        else:
            policy[i, j] = agent.getPolicy(np.array([i, j]))


np.savetxt('DoubleQSimpleMazePolicyAfterTrain' + mapName +'.txt', policy, fmt='%d', delimiter='\t')

plotPolicy(policy, N_A)