

from Agents.DDPG.DDPG import DDPGAgent
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
        #noiseFlag = False
        if noiseFlag:
            action = self.forward(state)
            noise = self.noise.get_noise()
            action += torch.tensor(noise, dtype=torch.float32).unsqueeze(0)
            action = torch.clamp(action, 0, 1)
            return action
        else:
            return self.forward(state)



def plotPolicy(x, policy):
    plt.plot(x, policy)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )


# first construct the neutral network


env = HedgingSimulator()
N_S = env.stateDim
N_A = env.nbActions

configName = 'config.json'
with open(configName ,'r') as f:
    config = json.load(f)


nPeriods = config['nPeriods']

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
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)

plotPolicyFlag = True
N = 50
if plotPolicyFlag:
    for phiIdx in range(8):
        phi = phiIdx * np.pi / 4.0

        xSet = np.linspace(-4,4,N)
        ySet = np.linspace(-4,4,N)
        policy = np.zeros((N, N))

        value = np.zeros((N, N))
        for i, x in enumerate(xSet):
            for j, y in enumerate(ySet):
                distance = - np.array([x, y])
                dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
                dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)


                state = torch.tensor([dx, dy], dtype=torch.float32).unsqueeze(0)
                action = agent.actorNet.select_action(state, noiseFlag = False)
                value[i, j] = agent.criticNet.forward(state, action).item()
                action = action.detach().numpy()
                policy[i, j] = action

        np.savetxt('StabilizerPolicyBeforeTrain'+'phiIdx'+ str(phiIdx) + '.txt', policy, fmt='%+.3f')
        np.savetxt('StabilizerValueBeforeTrain'+'phiIdx'+ str(phiIdx) + '.txt', value, fmt='%+.3f')

agent.train()



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

if plotPolicyFlag:
    for phiIdx in range(8):
        phi = phiIdx * np.pi / 4.0

        xSet = np.linspace(-4,4,N)
        ySet = np.linspace(-4,4,N)
        policy = np.zeros((N, N))

        value = np.zeros((N, N))
        for i, x in enumerate(xSet):
            for j, y in enumerate(ySet):
                distance = - np.array([x, y])
                dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
                dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)

                state = torch.tensor([dx, dy], dtype=torch.float32).unsqueeze(0)
                action = agent.actorNet.select_action(state, noiseFlag=False)
                value[i, j] = agent.criticNet.forward(state, action).item()
                action = action.detach().numpy()
                policy[i, j] = action

        np.savetxt('StabilizerPolicyAfterTrain' + 'phiIdx' + str(phiIdx) + '.txt', policy, fmt='%+.3f')
        np.savetxt('StabilizerValueAfterTrain' + 'phiIdx' + str(phiIdx) + '.txt', value, fmt='%+.3f')


#np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%d')

#plotPolicy(xSet, policy)


