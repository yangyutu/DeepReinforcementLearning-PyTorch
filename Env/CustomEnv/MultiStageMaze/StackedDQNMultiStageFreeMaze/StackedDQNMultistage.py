from Agents.StackedDQN.StackedDQN import StackedDQNAgent
from Env.CustomEnv.MultiStageMaze.MultiStageFreeMaze import CooperativeSimpleMazeTwoD
from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze, TrajRecorder
from utils.netInit import xavier_init
from Agents.DQN.DQNA3C import DQNA3CMaster, SharedAdam
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



from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.SimpleMazeTwoD import SimpleMazeTwoD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(2)

# first construct the neutral network
config = dict()

config['trainStep'] = 5000
config['epsThreshold'] = 0.5
config['epsilon_start'] = 0.5
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 500
config['episodeLength'] = 200
config['numStages'] = 6
config['targetNetUpdateStep'] = 10
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.0001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMazeLog/traj'
config['logFrequency'] = 500
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['device'] = 'cpu'
config['mapWidth'] = 6
config['mapHeight'] = 6


def stateProcessor(state, device = 'cpu', done = None):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}

    if done is None:
        senorList = [item['state'] for item in state if item is not None]
        nonFinalState = torch.tensor(senorList, dtype=torch.float32, device=device)
        return nonFinalState, None
    else:
        stageID = state[0]['stageID']
        nonFinalMask = torch.tensor([not s['stage'][stageID] for s in done], device=device,
                                    dtype=torch.uint8)
        sensorList = [item['state'] for item, s in zip(state, nonFinalMask) if s]
        nonFinalState = torch.tensor(sensorList, device=device, dtype=torch.float32)
        finalMask = 1 - nonFinalMask
        sensorList = [item['state'] for item, s in zip(state,finalMask) if s]
        finalState = torch.tensor(sensorList, device=device, dtype=torch.float32)

        return nonFinalState, nonFinalMask, finalState, finalMask
env = CooperativeSimpleMazeTwoD(config)
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [128]
netParameter['n_output'] = N_A

nPeriods = config['numStages']

policyNets = [MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output']) for _ in range(nPeriods)]

targetNets = [MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output']) for _ in range(nPeriods)]
optimizers = [optim.Adam(net.parameters(), lr=config['learningRate']) for net in policyNets]


agent = StackedDQNAgent(config, policyNets, targetNets, env, optimizers, torch.nn.MSELoss(reduction='none'), N_A, stateProcessor=stateProcessor)


policyFlag = True

if policyFlag:
    for n in range(nPeriods):
        policy = np.zeros((env.mapHeight * env.numStages, env.mapWidth * env.numStages), dtype=np.int32)
        value = np.zeros((env.mapHeight * env.numStages, env.mapWidth * env.numStages), dtype=np.float)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                state = {'stageID': n,
                         'state': np.array([i / agent.env.lengthScale, \
                                            j / agent.env.lengthScale])
                         }
                policy[i, j] = agent.select_action(agent.policyNets[n], state, -0.1)
                stateTorch = stateProcessor([state], config['device'])[0]
                Qvalue = agent.policyNets[n](stateTorch)
                value[i, j] = Qvalue[0, policy[i, j]].cpu().item()
        np.savetxt('SimpleMazePolicyBeforeTrain_stage' + str(n) + '.txt', policy, fmt='%d', delimiter='\t')
        np.savetxt('SimpleMazeValueBeforeTrain_stage' + str(n) + '.txt', value, fmt='%f', delimiter='\t')











agent.train()





nTraj = 1
nSteps  = 80

# test for starting from second stage
for i in range(nTraj):
    state = agent.env.reset()
    agent.env.stageID = 1
    state['stageID'] = agent.env.stageID
    agent.env.done['stage'][0] = True
    agent.env.stepCount = 0
    agent.env.currentState = np.array([0.0, agent.env.mapWidth / 2 + 1])
    state['state']= np.array([0.0, agent.env.mapWidth / 2 + 1])

    for step in range(nSteps):
        timeStep = int(state['stageID'])
        action = agent.select_action(agent.policyNets[timeStep], state, -0.1)
        nextState, reward, done, info = agent.env.step(action)

        state = nextState

        if done['global']:
            print('finish step:', agent.env.stepCount)
            print(agent.env.currentState)
            break


# test for starting from second stage
for i in range(nTraj):
    state = agent.env.reset()
    agent.env.stageID = 0
    state['stageID'] = agent.env.stageID
    agent.env.done['stage'][0] = False
    agent.env.stepCount = 0
    agent.env.currentState = np.array([0.0, 0.0])
    state['state']= np.array([0.0, 0.0])

    for step in range(nSteps):
        timeStep = int(state['stageID'])
        action = agent.select_action(agent.policyNets[timeStep], state, -0.1)
        nextState, reward, done, info = agent.env.step(action)

        state = nextState

        if done['global']:
            print('finish step:', agent.env.stepCount)
            print(agent.env.currentState)
            break

policyFlag = True

if policyFlag:
    for n in range(nPeriods):
        policy = np.zeros((env.mapHeight * env.numStages, env.mapWidth * env.numStages), dtype=np.int32)
        value = np.zeros((env.mapHeight * env.numStages, env.mapWidth * env.numStages), dtype=np.float)
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                state = {'stageID': n,
                         'state': np.array([i / agent.env.lengthScale, \
                                            j / agent.env.lengthScale])
                         }
                policy[i, j] = agent.select_action(agent.policyNets[n], state, -0.1)
                stateTorch = stateProcessor([state], config['device'])[0]
                Qvalue = agent.policyNets[n](stateTorch)
                value[i, j] = Qvalue[0, policy[i, j]].cpu().item()
        np.savetxt('SimpleMazePolicyAfterTrain_stage' + str(n) + '.txt', policy, fmt='%d', delimiter='\t')
        np.savetxt('SimpleMazeValueAfterTrain_stage' + str(n) + '.txt', value, fmt='%f', delimiter='\t')