from Agents.DDPG.DDPG import DDPGAgent
from Agents.TDDDPG.TDDDPG import TDDDPGAgent

from Env.CustomEnv.StablizerMultiD import StablizerMultiDContinuous
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
from Env.CustomEnv.TwoArmRobot.TwoArmRobotEnv import TwoArmEnvironmentContinuous
torch.manual_seed(1)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
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
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=10000)
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
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)

env = TwoArmEnvironmentContinuous(config)
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 100
netParameter['n_output'] = N_A
actorNet = Actor(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

actorTargetNet = deepcopy(actorNet)

criticNet = Critic(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden'])
criticNetTwo = deepcopy(criticNet)
criticTargetNet = deepcopy(criticNet)
criticTargetNetTwo = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])
criticOptimizerTwo = optim.Adam(criticNetTwo.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'criticOne': criticNet, 'criticTwo': criticNetTwo, 'targetOne': criticTargetNet, 'targetTwo': criticTargetNetTwo}
optimizers = {'actor': actorOptimizer, 'criticOne':criticOptimizer, 'criticTwo': criticOptimizerTwo}
agent = TDDDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)


agent.train()