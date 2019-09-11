
from Agents.DQN.DQNMultiStageUnit import DQNMultiStageUnit
from Agents.DDPG.DDPG import DDPGAgent
from Env.CustomEnv.StablizerOneD import StablizerOneDContinuous
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
from activeParticleEnv import ActiveParticleEnvMultiMap, ActiveParticleEnv
from Env.CustomEnv.MultiStageMaze.TwoStageSPVisualMaze import TwoStageSPVisualMaze
from Agents.TDDDPG.TD3MultiStageUnit import TD3MultiStageUnit
from Agents.MultistageController.MultiStageController import MultiStageStackedController
from Agents.Core.MLPNet import MultiLayerNetRegression

import math
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
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.5, max_sigma=0.05, min_sigma=0.001, decay_period=10000)
        self.noise.reset()
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = torch.sigmoid(self.linear3(x))

        return action

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, 0, 1)
            return action
        return self.forward(state)

def experienceProcessor(state, action, nextState, reward, info):
    if nextState is not None:
        target = info['previousTarget']
        distance = target - info['currentState'][:2]
        phi = info['currentState'][2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = -distance[0] * math.sin(phi) + distance[1] * math.cos(phi)
        nextState['target'] = np.array([dx / info['scaleFactor'], dy / info['scaleFactor']])
    return state, action, nextState, reward

def stateProcessor(state, device = 'cpu', done = None):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}

    if done is None:
        senorList = [item['sensor'] for item in state if item is not None]
        targetList = [item['target'] for item in state if item is not None]
        nonFinalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
                         'target': torch.tensor(targetList, dtype=torch.float32, device=device)}
        return nonFinalState, None
    else:
        finalMask = torch.tensor(done, device=device, dtype=torch.uint8)
        nonFinalMask = 1 - finalMask

        senorList = [item['sensor'] for idx, item in enumerate(state) if nonFinalMask[idx]]
        targetList = [item['target'] for idx, item in enumerate(state) if nonFinalMask[idx]]
        nonFinalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
                         'target': torch.tensor(targetList, dtype=torch.float32, device=device)}

        senorList = [item['sensor'] for idx, item in enumerate(state) if finalMask[idx]]
        targetList = [item['target'] for idx, item in enumerate(state) if finalMask[idx]]
        finalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
                         'target': torch.tensor(targetList, dtype=torch.float32, device=device)}

        return nonFinalState, nonFinalMask, finalState, finalMask


configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)


env = TwoStageSPVisualMaze(config)

N_S = env.stateDim[0]
N_A = env.nbActions


agents = []

netParameter = dict()

N_S = 2
N_A = env.nbActions[0]

netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [128]




policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    N_A)

targetNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    N_A)
optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = DQNMultiStageUnit(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A,
                          stateProcessor=None)

agents.append(agent)
netParameter['n_hidden'] = 128
netParameter['n_output'] = env.nbActions[1]
N_A = env.nbActions[1]
actorNet = Actor(N_S,
                 netParameter['n_hidden'],
                 netParameter['n_output'])

actorTargetNet = deepcopy(actorNet)

criticNet = Critic(N_S + N_A, netParameter['n_hidden'])

criticNetTwo = deepcopy(criticNet)
criticTargetNet = deepcopy(criticNet)
criticTargetNetTwo = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])
criticOptimizerTwo = optim.Adam(criticNetTwo.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'criticOne': criticNet, 'criticTwo': criticNetTwo, 'targetOne': criticTargetNet,
              'targetTwo': criticTargetNetTwo}
optimizers = {'actor': actorOptimizer, 'criticOne': criticOptimizer, 'criticTwo': criticOptimizerTwo}

agent = TD3MultiStageUnit(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), netParameter['n_output'], stateProcessor=None)

agents.append(agent)


controller = MultiStageStackedController(config, agents, env)


controller.train()
