from Env.CustomEnv.MultiAgentMaze.TwoAgentCooperativeTransport import TwoAgentCooperativeTransport
from Agents.MADQN.MADQN import MADQNAgent
import torch
import torch.nn as nn

torch.manual_seed(1)
import torch.nn.functional as F
torch.set_num_threads(1)



from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(3)

# first construct the neutral network
config = dict()

config['trainStep'] = 10000
config['epsThreshold'] = 0.5
config['epsilon_start'] = 0.5
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 1500
config['episodeLength'] = 200
config['numStages'] = 6
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.9
config['learningRate'] = 0.0001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFrequency'] = 500
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['device'] = 'cpu'
config['mapWidth'] = 6
config['mapHeight'] = 6
config['numAgents'] = 2

env = TwoAgentCooperativeTransport(config)

N_S = env.stateDim
N_A = env.nbActions

numAgents = env.numAgents
netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [128]
netParameter['n_output'] = N_A


policyNets = [MultiLayerNetRegression(N_S[n],
                                    netParameter['n_hidden'],
                                    N_A[n]) for n in range(numAgents)]

targetNets = [MultiLayerNetRegression(N_S[n],
                                    netParameter['n_hidden'],
                                    N_A[n]) for n in range(numAgents)]

optimizers = [optim.Adam(net.parameters(), lr=config['learningRate']) for net in policyNets]


agent = MADQNAgent(config, policyNets, targetNets, env, optimizers, torch.nn.MSELoss(reduction='none'), N_A)


policyFlag = True

if policyFlag:
    for n in range(numAgents):
        policy = np.zeros((env.mapHeight, env.mapWidth), dtype=np.int32)
        value = np.zeros((env.mapHeight, env.mapWidth), dtype=np.float)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                state = np.array([0 for _ in range(config['mapHeight'])] + [i / agent.env.lengthScale, \
                                  j / agent.env.lengthScale, \
                                  i / agent.env.lengthScale, \
                                  j / agent.env.lengthScale, \
                                  ])

                states = np.array([state, state])

                policy[i, j] = agent.select_action(agent.policyNets, states, -0.1)[n]
                Qvalue = agent.policyNets[n](torch.tensor(state, dtype=torch.float32))
                value[i, j] = Qvalue[policy[i, j]].cpu().item()
        np.savetxt('SimpleMazePolicyBeforeTrain_agent' + str(n) + '.txt', policy, fmt='%d', delimiter='\t')
        np.savetxt('SimpleMazeValueBeforeTrain_agent' + str(n) + '.txt', value, fmt='%f', delimiter='\t')



agent.train()





nTraj = 1
nSteps  = 80

if policyFlag:
    for n in range(numAgents):
        policy = np.zeros((env.mapHeight, env.mapWidth), dtype=np.int32)
        value = np.zeros((env.mapHeight, env.mapWidth), dtype=np.float)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                state = np.array([0 for _ in range(config['mapHeight'])] + [i / agent.env.lengthScale, \
                                  j / agent.env.lengthScale, \
                                  i / agent.env.lengthScale, \
                                  j / agent.env.lengthScale, \
                                  ])

                states = np.array([state, state])

                policy[i, j] = agent.select_action(agent.policyNets, states, -0.1)[n]
                Qvalue = agent.policyNets[n](torch.tensor(state, dtype=torch.float32))
                value[i, j] = Qvalue[policy[i, j]].cpu().item()
        np.savetxt('SimpleMazePolicyAfterTrain_agent' + str(n) + '.txt', policy, fmt='%d', delimiter='\t')
        np.savetxt('SimpleMazeValueAfterTrain_agent' + str(n) + '.txt', value, fmt='%f', delimiter='\t')