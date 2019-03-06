

from Agents.DQN.DQN import DQNAgent
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze, TrajRecorder
from Agents.DQN.DQNA3CV2 import DQNA3CMasterV2, SharedAdam
import json
from torch import optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import os
import torch.multiprocessing as mp
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
# add this line to avoid cuda initilization error

torch.manual_seed(1)
import torch.nn.functional as F
#torch.set_num_threads(5)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)


def weights_init(m):
    if type(m) == nn.Conv2d:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# Convolutional neural network (two convolutional layers)
# this network use elu unit, following https://github.com/yangyutu/pytorch-a3c-1/blob/master/model.py
# https://blog.csdn.net/mao_xiao_feng/article/details/53242235
class MulChanConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(MulChanConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth)
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(1,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1

        self.fc0 = nn.Linear(2, 32)
        self.fc1 = nn.Linear(self.featureSize() + 32, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_action)
        self.apply(weights_init)

    def forward(self, state):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        # xout.fill_(0)
        yout = F.relu(self.fc0(y))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 1, *self.inputShape))).view(1, -1).size(1)


def plotPolicy(policy, nbActions):
    idx, idy = np.where(policy >=0)
    action = policy[idx,idy]
    plt.scatter(idx, idy, c = action, marker='s', s = 10)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )



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


policyNet = MulChanConvNet(N_S, 128, N_A)
targetNet = MulChanConvNet(N_S, 128, N_A)
optimizer = SharedAdam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNA3CMasterV2(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A,
                 stateProcessor=stateProcessor, config = config)


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
