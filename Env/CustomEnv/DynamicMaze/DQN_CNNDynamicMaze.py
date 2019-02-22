

from Agents.DQN.DQN import DQNAgent
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze

import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
torch.manual_seed(1)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, inputWdith, num_action):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential( # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(1,             # input channel
                      16,            # output channel
                      kernel_size=2, # filter size
                      stride=1,
                      padding=1),   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2
        # add a fully connected layer
        width = int(inputWdith / 4) + 1
        self.fc = nn.Linear(width * width * 32, num_action)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out



def plotPolicy(policy, nbActions):
    idx, idy = np.where(policy >=0)
    action = policy[idx,idy]
    plt.scatter(idx, idy, c = action, marker='s', s = 10)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )


# first construct the neutral network
config = dict()

mapName = 'SimpleMapSmall.txt'

config['trainStep'] = 200
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 2000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.01
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'SimpleMapSmall/traj' + mapName
config['logFrequency'] = 50

config['mazeFileName'] = 'simpleMapSmall.txt'
config['numCircObs'] = 2
config['dynamicObsFlag'] = False
config['agentReceptHalfWidth'] = 5
config['obstacleMapPaddingWidth'] = 5
config['targetState'] = (1, 1)
config['dynamicTargetFlag'] = False

# directory = config['mazeFileName'].split('.')[0]
# if not os.path.exists(directory):
#     os.makedirs(directory)



env = DynamicMaze(config)
env.reset()
N_S = env.stateDim[0]
N_A = env.nbActions


policyNet = ConvNet(N_S, N_A)

#print(policyNet.state_dict())

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(), N_A, config=config)

policy = deepcopy(env.mapMat)
for i in range(policy.shape[0]):
     for j in range(policy.shape[1]):
         if env.mapMat[i, j] == 1:
             policy[i, j] = -1
         else:
             sensorInfo = env.agent.getSensorInfoFromPos(np.array([i,j]))
             policy[i, j] = agent.getPolicy(sensorInfo)


np.savetxt('DynamicMazePolicyBeforeTrain' + mapName + '.txt', policy, fmt='%d', delimiter='\t')
#
# plotPolicy(policy, N_A)

agent.train()

for i in range(policy.shape[0]):
     for j in range(policy.shape[1]):
         if env.mapMat[i, j] == 1:
             policy[i, j] = -1
         else:
             sensorInfo = env.agent.getSensorInfoFromPos(np.array([i,j]))
             policy[i, j] = agent.getPolicy(sensorInfo)


np.savetxt('DynamicMazePolicyAfterTrain' + mapName +'.txt', policy, fmt='%d', delimiter='\t')

#plotPolicy(policy, N_A)