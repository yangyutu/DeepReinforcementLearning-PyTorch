

from Agents.NAF.NAF import NAFAgent
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

torch.manual_seed(1)

class Policy(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, output_size)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, output_size ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

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
        mu = torch.tanh(self.mu(x))

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
            action = torch.clamp(action, -1, 1)

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
config['learningRate'] = 0.0005

config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['episodeLength'] = 200
env = StablizerOneDContinuous()
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

xSet = np.linspace(-4,4,100)
policy = np.zeros_like(xSet)
value = np.zeros_like(xSet)
for i, x in enumerate(xSet):
    state = torch.tensor([x], dtype=torch.float32).unsqueeze(0)
    action = agent.policyNet.select_action(state, noiseFlag=False)
    policy[i] = action
    value[i] = agent.policyNet.eval_state_value(state)

np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%f')
np.savetxt('StabilizerValueBeforeTrain.txt', value, fmt='%f')

agent.train()

xSet = np.linspace(-4,4,100)
policy = np.zeros_like(xSet)
value = np.zeros_like(xSet)
for i, x in enumerate(xSet):
    state = torch.tensor([x], dtype=torch.float32).unsqueeze(0)
    action = agent.policyNet.select_action(state, noiseFlag=False)
    policy[i] = action
    value[i] = agent.policyNet.eval_state_value(state)

np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%f')
np.savetxt('StabilizerValueAfterTrain.txt', value, fmt='%f')