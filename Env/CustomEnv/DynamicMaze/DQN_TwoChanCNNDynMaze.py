

from Agents.DQN.DQN import DQNAgent
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze

import json
from torch import optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
torch.manual_seed(1)
import torch.nn.functional as F
torch.set_num_threads(1)
# Convolutional neural network (two convolutional layers)
class MulChanConvNet(nn.Module):
    def __init__(self, inputWdith, num_hidden, num_action):
        super(MulChanConvNet, self).__init__()
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
        self.fc1 = nn.Linear(width * width * 32 + 2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_action)

    def forward(self, state):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
       # xout.fill_(0)
        out = torch.cat((xout, y), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
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

config['trainStep'] = 1000
config['epsThreshold'] = 0.15
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMapTwoObs/traj' + mapName
config['logFrequency'] = 50

config['mazeFileName'] = 'simpleMapTwoObs.txt'
config['numCircObs'] = 2
config['dynamicObsFlag'] = False
config['agentReceptHalfWidth'] = 5
config['obstacleMapPaddingWidth'] = 5
config['targetState'] = (1, 1)
config['dynamicTargetFlag'] = True
config['verbose'] = False
config['targetMoveFreq'] = 3
config['stochAgent'] = False
config['loadExistingModel'] = True
config['saveModelFile'] = 'dynamicMazeModel.pt'
# directory = config['mazeFileName'].split('.')[0]
# if not os.path.exists(directory):
#     os.makedirs(directory)






def stateProcessor(state):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    senorList = [item['sensor'] for item in state]
    targetList = [item['target'] for item in state]
    output = {'sensor': torch.tensor(senorList, dtype=torch.float32),
              'target': torch.tensor(targetList, dtype=torch.float32),}
    return output

env = DynamicMaze(config)
env.reset()
N_S = env.stateDim[0]
N_A = env.nbActions


policyNet = MulChanConvNet(N_S, 100, N_A)

#print(policyNet.state_dict())

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(), N_A,
                 stateProcessor=stateProcessor, config=config)

policy = deepcopy(env.mapMat)
for i in range(policy.shape[0]):
      for j in range(policy.shape[1]):
          if env.mapMat[i, j] == 1:
              policy[i, j] = -1
          else:
              sensorInfo = env.agent.getSensorInfoFromPos(np.array([i,j]))
              distance = np.array([1, 1]) - np.array([i, j])
              state = {'sensor': sensorInfo, 'target': distance}
              policy[i, j] = agent.getPolicy(state)


np.savetxt('DynamicMazePolicyBeforeTrain' + mapName + '.txt', policy, fmt='%d', delimiter='\t')
#
# plotPolicy(policy, N_A)
if config['loadExistingModel']:
    checkpoint = torch.load(config['saveModelFile'])
    agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


agent.train()

targetTestSet = np.array([[1, 1], [1, 19], [8, 1], [8, 19], [8, 8], [15, 1], [15, 19]])
for target in targetTestSet:
    for i in range(policy.shape[0]):
          for j in range(policy.shape[1]):
              if env.mapMat[i, j] == 1:
                  policy[i, j] = -1
              else:
                  sensorInfo = env.agent.getSensorInfoFromPos(np.array([i,j]))
                  distance = target - np.array([i, j])
                  state = {'sensor': sensorInfo, 'target': distance}
                  policy[i, j] = agent.getPolicy(state)
    np.savetxt('DynamicMazePolicyAfterTrain' + str(target) + config['mazeFileName'] , policy, fmt='%d', delimiter='\t')


torch.save({
            'model_state_dict': agent.policyNet.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            }, config['saveModelFile'])


#plotPolicy(policy, N_A)