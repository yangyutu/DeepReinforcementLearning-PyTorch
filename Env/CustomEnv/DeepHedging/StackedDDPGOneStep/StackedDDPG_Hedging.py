

from Agents.StackedDDPG.StackedDDPG  import StackedDDPGAgent
from Env.CustomEnv.StablizerTwoD import StablizerTwoDContinuousSP
from utils.netInit import xavier_init
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.OUNoise import OUNoise
import math
from Env.CustomEnv.DeepHedging.HedgingEnv import HedgingSimulator
torch.manual_seed(1)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(xavier_init)
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.apply(xavier_init)
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.5, max_sigma=0.5, min_sigma=0.05, decay_period=100000)
        self.noise.reset()
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.linear3(x)

        return action

    def select_action(self, state, noiseFlag = False):
        #noiseFlag = False
        if noiseFlag:
            action = self.forward(state)
            noise = self.noise.get_noise()
            action += torch.tensor(noise, dtype=torch.float32).unsqueeze(0)
            #action = torch.clamp(action, 0, 1)
            return action
        else:
            return self.forward(state)

def stateProcessor(state, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)

    stateList = [item['state'] for item in state if item is not None]
    nonFinalState = torch.tensor(stateList, dtype=torch.float32, device=device)
    return nonFinalState, nonFinalMask

configName = 'config.json'
with open(configName ,'r') as f:
    config = json.load(f)

env = HedgingSimulator(config)
N_S = env.stateDim
N_A = env.nbActions

configName = 'config.json'
with open(configName ,'r') as f:
    config = json.load(f)


nPeriods = config['episodeLength']

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 100
netParameter['n_output'] = N_A

actorNets = [Actor(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output']) for _ in range(nPeriods)]

actorTargetNets = [Actor(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output']) for _ in range(nPeriods)]

criticNets = [Critic(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden']) for _ in range(nPeriods)]

criticTargetNets = [Critic(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden']) for _ in range(nPeriods)]


actorOptimizers = [optim.Adam(actorNet.parameters(), lr=config['actorLearningRate']) for actorNet in actorNets]
criticOptimizers = [optim.Adam(criticNet.parameters(), lr=config['criticLearningRate']) for criticNet in criticNets]

actorNets = {'actor': actorNets, 'target': actorTargetNets}
criticNets = {'critic': criticNets, 'target': criticTargetNets}
optimizers = {'actor': actorOptimizers, 'critic':criticOptimizers}

timeIndexMap = {0:0, 1:0}

agent = StackedDDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A, stateProcessor=stateProcessor ,timeIndexMap = timeIndexMap)

S = np.linspace(0.5, 1.5, 101)
result = []
for s in S:
    state = np.array([0.0, 0.0, s])
    combinedState = {'state': state, 'timeStep': 0}

    action = agent.select_action(agent.actorNets[0], combinedState, False)
    result.append([s, action, action / s])

np.savetxt("resultBeforeTrain.txt", np.array(result))



agent.train()

S = np.linspace(0.5, 1.5, 101)
result = []
for s in S:
    state = np.array([0.0, 0.0, s])
    combinedState = {'state': state, 'timeStep': 0}

    action = agent.select_action(agent.actorNets[0], combinedState, False)
    result.append([s, action, action / s])

np.savetxt("resultAfterTrain.txt", np.array(result))