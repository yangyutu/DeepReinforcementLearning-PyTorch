

from Agents.DQN.DQN import DQNAgent
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze, TrajRecorder

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
        #xout.fill_(0)
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

mapName = 'singleObstacle'
config['mapName'] = mapName
config['trainStep'] = 6000
config['JumpMatrix'] = 'trajSampleHalf.npz'
config['epsThreshold'] = 0.1
config['epsilon_start'] = 0.3
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 2000
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = mapName + '/traj' + mapName
config['logFrequency'] = 50
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['nStepForward'] = 10
config['device'] = 'cuda'


config['mazeFileName'] = mapName + '.txt'
config['numCircObs'] = 2
config['dynamicObsFlag'] = False
config['agentReceptHalfWidth'] = 5
config['obstacleMapPaddingWidth'] = 10
config['targetState'] = (10.0, 10.0)
config['currentState'] = (1.0, 1.0, 0.0)
config['dynamicInitialStateFlag'] = True
config['dynamicTargetFlag'] = True
config['verbose'] = False
config['targetMoveFreq'] = 2
config['stochAgent'] = True
config['loadExistingModel'] = True
config['saveModelFile'] = mapName + 'dynamicMazeModel.pt'


# directory = config['mazeFileName'].split('.')[0]
# if not os.path.exists(directory):
#     os.makedirs(directory)


def stateProcessor(state):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    senorList = [item['sensor'] for item in state]
    targetList = [item['target'] for item in state]
    output = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=config['device']),
              'target': torch.tensor(targetList, dtype=torch.float32, device=config['device'])}
    return output

env = DynamicMaze(config)

N_S = env.stateDim[0]
N_A = env.nbActions


policyNet = MulChanConvNet(N_S, 100, N_A)

#print(policyNet.state_dict())

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A,
                 stateProcessor=stateProcessor, config=config)


trainFlag = True
testFlag = True

if trainFlag:

    if config['loadExistingModel']:
        checkpoint = torch.load(config['saveModelFile'])
        agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    plotPolicyFlag = True
    if plotPolicyFlag:

        for phiIdx in range(8):
            phi = phiIdx * np.pi/4.0
            policy = deepcopy(env.mapMat)
            for i in range(policy.shape[0]):
                  for j in range(policy.shape[1]):
                      if env.mapMat[i, j] == 1:
                          policy[i, j] = -1
                      else:
                          sensorInfo = env.agent.getSensorInfoFromPos(np.array([i, j, phi]))
                          distance = np.array(config['targetState']) - np.array([i, j])
                          dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
                          dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
                          state = {'sensor': sensorInfo, 'target': np.array([dx, dy])}
                          policy[i, j] = agent.getPolicy(state)
            np.savetxt('DynamicMazePolicyBeforeTrain' + config['mapName'] +'phiIdx'+ str(phiIdx) + '.txt', policy, fmt='%d', delimiter='\t')
    #
    # plotPolicy(policy, N_A)



    agent.train()


    if plotPolicyFlag:
        for phiIdx in range(8):
            phi = phiIdx * np.pi / 4.0
            policy = deepcopy(env.mapMat)
            for i in range(policy.shape[0]):
                for j in range(policy.shape[1]):
                    if env.mapMat[i, j] == 1:
                        policy[i, j] = -1
                    else:
                        sensorInfo = env.agent.getSensorInfoFromPos(np.array([i, j, phi]))
                        distance = np.array(config['targetState']) - np.array([i, j])
                        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
                        dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
                        state = {'sensor': sensorInfo, 'target': np.array([dx, dy])}
                        policy[i, j] = agent.getPolicy(state)
            np.savetxt('DynamicMazePolicyAfterTrain' + config['mapName'] + 'phiIdx' + str(phiIdx) + '.txt', policy, fmt='%d',
                       delimiter='\t')

    torch.save({
                'model_state_dict': agent.policyNet.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                }, config['saveModelFile'])


if testFlag:
    config['loadExistingModel'] = True

    if config['loadExistingModel']:
        checkpoint = torch.load(config['saveModelFile'])
        agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    recorder = TrajRecorder()
    agent.testPolicyNet(100, recorder)
    recorder.write_to_file(config['mapName'] + 'TestTraj.txt')

#plotPolicy(policy, N_A)
