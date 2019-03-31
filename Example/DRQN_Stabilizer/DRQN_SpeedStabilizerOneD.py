

from Agents.DRQN.DRQN import DRQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression, MultiLayerNetRegressionWithGRU, SingleGRULayerNetRegression
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Env.CustomEnv.SpeedStablizerOneD import SpeedStablizerOneD
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)


def plotPolicy(x, policy):
    plt.plot(x, policy)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )


# first construct the neutral network
config = dict()

config['trainStep'] = 2000
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 200000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.01
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'targetNet'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['sequenceLength'] = 4
config['netMemory'] = 'GRU'

env = SpeedStablizerOneD()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [10]
netParameter['n_output'] = N_A
netParameter['n_gru'] = 32


policyNet = MultiLayerNetRegressionWithGRU(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'],
                                    netParameter['n_gru'])
#policyNet = SingleGRULayerNetRegression(n_feature=N_S, n_output=N_A, gru_size=32)  # define the network


targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DRQNAgent(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A)

# xSet = np.linspace(-1,1,100)
# policy = np.zeros_like(xSet)
# for i, x in enumerate(xSet):
#     policy[i] = agent.getPolicy(np.array([x]))
#
# np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%d')

#agent.perform_random_exploration(10)
agent.train()
#storeMemory = ReplayMemory(100000)
#agent.testPolicyNet(100)
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
# for i, x in enumerate(xSet):
#     policy[i] = agent.getPolicy(np.array([x]))
#
#
# np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%d')
#
# plotPolicy(xSet, policy)


