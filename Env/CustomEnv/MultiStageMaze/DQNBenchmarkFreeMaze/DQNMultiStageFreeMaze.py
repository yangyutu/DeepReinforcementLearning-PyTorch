from Agents.DQN.DQN import DQNAgent
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

config['trainStep'] = 1000
config['epsThreshold'] = 0.3
config['epsilon_start'] = 0.2
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 100
config['targetNetUpdateStep'] = 10
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 64
config['gamma'] = 0.99
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = 'SimpleMazeLog/traj'
config['logFrequency'] = 50
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'targetNet'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['device'] = 'cpu'
config['multiStage'] = False
def stateProcessor(state, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}

    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)

    senorList = [item['state'] for item in state if item is not None]
    nonFinalState = torch.tensor(senorList, dtype=torch.float32, device=device)
    return nonFinalState, nonFinalMask

env = CooperativeSimpleMazeTwoD(config=config)
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A


policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])
targetNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])
optimizers = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = DQNAgent(config, policyNet, targetNet, env, optimizers, torch.nn.MSELoss(reduction='none'), N_A, stateProcessor=stateProcessor)

agent.train()





nTraj = 100
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
        action = agent.select_action(agent.policyNet, state, -0.1)
        nextState, reward, done, info = agent.env.step(action)

        state = nextState

        if done:
            print('finish step:', agent.env.stepCount)
            print(agent.env.currentState)
            break


policyFlag = True

if policyFlag:
    policy = np.zeros((env.mapHeight, env.mapWidth), dtype=np.int32)
    value = np.zeros((env.mapHeight, env.mapWidth), dtype=np.float)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            state = {'state': np.array([i / agent.env.lengthScale, \
                                        j / agent.env.lengthScale])
                     }
            policy[i, j] = agent.select_action(agent.policyNet, state, -0.1)
            stateTorch = stateProcessor([state], config['device'])[0]
            Qvalue = agent.policyNet(stateTorch)
            value[i, j] = Qvalue[0, policy[i, j]].cpu().item()
    np.savetxt('SimpleMazePolicyAfterTrain_stage.txt', policy, fmt='%d', delimiter='\t')
    np.savetxt('SimpleMazeValueAfterTrain_stage.txt', value, fmt='%f', delimiter='\t')