

from Agents.NAF.NAF import NAFAgent
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
torch.manual_seed(1)

class Policy(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)

        self.mu = nn.Linear(hidden_size, output_size)

        self.L = nn.Linear(hidden_size, output_size ** 2)


        self.tril_mask = torch.tril(torch.ones(output_size, output_size), diagonal=-1).unsqueeze(0)
        self.diag_mask = torch.diag(torch.diag(torch.ones(output_size, output_size))).unsqueeze(0)

        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=10000)
        self.noise.reset()
        self.apply(xavier_init)
    def forward(self, state, action):
        x = state
        u = action
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))

        V = self.V(x)
        mu = torch.sigmoid(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V

    def select_action(self, state, noiseFlag = True):
        action, _, _ = self.forward(state, None)

        if noiseFlag:
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32).unsqueeze(0)
            action = torch.clamp(action, 0, 1)

        return action

    def eval_Q_value(self, state, action):
        _, Q, _ = self.forward(state, action)
        return Q

    def eval_state_value(self, state):
        _, _, V = self.forward(state, None)
        return V

def plotPolicy(x, policy):
    plt.plot(x, policy)
    # for i in range(nbActions):
    #     idx, idy = np.where(policy == i)
    #     plt.plot(idx,idy, )


# first construct the neutral network
config = dict()

config['trainStep'] = 1500
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 20000
config['trainBatchSize'] = 64
config['gamma'] = 0.9
config['tau'] = 0.01
config['learningRate'] = 0.00025

config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['episodeLength'] = 200
config['hindSightER'] = True
config['hindSightERFreq'] = 10

env = StablizerTwoDContinuousSP()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 100
netParameter['n_output'] = N_A

policyNet = Policy(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

targetNet = Policy(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])


optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = NAFAgent(config, policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='mean'), N_A)


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
                action = agent.policyNet.select_action(state, noiseFlag = False)
                value[i, j] = agent.policyNet.eval_state_value(state).item()
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
                action = agent.policyNet.select_action(state, noiseFlag = False)
                value[i, j] = agent.policyNet.eval_state_value(state).item()
                action = action.detach().numpy()
                policy[i, j] = action

        np.savetxt('StabilizerPolicyAfterTrain' + 'phiIdx' + str(phiIdx) + '.txt', policy, fmt='%+.3f')
        np.savetxt('StabilizerValueAfterTrain' + 'phiIdx' + str(phiIdx) + '.txt', value, fmt='%+.3f')


#np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%d')

#plotPolicy(xSet, policy)


