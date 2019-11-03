

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
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv

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
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.1, decay_period=100000)
        self.noise.reset()
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))

        return action

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = ActiveParticle3DEnv('config.json',1)
if config['localFrameFlag']:
    N_S = 6
else:
    N_S = 6

N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 256
netParameter['n_output'] = N_A

actorNet = Actor(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

actorTargetNet = deepcopy(actorNet)

criticNet = Critic(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden'])

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)


plotPolicyFlag = False
N = 100
if plotPolicyFlag:
    phi = 0.0

    xSet = np.linspace(-10,10,N)
    ySet = np.linspace(-10,10,N)
    policy = np.zeros((N, N))

    value = np.zeros((N, N))
    for i, x in enumerate(xSet):
        for j, y in enumerate(ySet):
            # x, y is the target position, (0, 0, 0) is the particle configuration
            distance = np.array([x, y])
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
            angle = math.atan2(dy, dx)
            if math.sqrt(dx ** 2 + dy ** 2) > env.targetClipLength:
                dx = env.targetClipLength * math.cos(angle)
                dy = env.targetClipLength * math.sin(angle)

            state = torch.tensor([dx, dy], dtype=torch.float32, device = config['device']).unsqueeze(0)
            action = agent.actorNet.select_action(state, noiseFlag = False)
            value[i, j] = agent.criticNet.forward(state, action).item()
            action = action.cpu().detach().numpy()
            policy[i, j] = action

    np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%+.3f')
    np.savetxt('StabilizerValueBeforeTrain.txt',value, fmt='%+.3f')

agent.train()

