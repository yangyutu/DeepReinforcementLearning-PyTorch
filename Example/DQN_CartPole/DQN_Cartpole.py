

from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
import gym
from torch import optim
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Env.CustomEnv.MountainCarEnv import MountainCarEnvCustom
from Env.CustomEnv.CartPoleEnvCustom import CartPoleEnvCustom
torch.manual_seed(10)
# first construct the neutral network

config = dict()

config['trainStep'] = 100
config['epsThreshold'] = 0.1
config['epsilon_start'] = 0.7
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 200
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 10000
config['trainBatchSize'] = 32
config['gamma'] = 0.8
config['learningRate'] = 0.001
#config['netGradClip'] = 10
config['logFlag'] = False
config['logFileName'] = ''
config['logFrequency'] = 50
config['verbose'] = False
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'targetNet'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5


# Get the environment and extract the number of actions.
env = CartPoleEnvCustom()

N_S = env.observation_space.shape[0]
N_A = env.action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [256]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


policyNet = Net(netParameter['n_feature'],
                                    256,
                                    netParameter['n_output'])

targetNet = deepcopy(policyNet)

#print(policyNet.state_dict())

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none'), N_A, config=config)


#checkpoint = torch.load('DQNCartPole.pt')
#agent.policyNet.load_state_dict(checkpoint['model_state_dict'])
#agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#agent.testPolicyNet(100)

agent.train()

# benchmark score
# https://github.com/openai/gym/wiki/Leaderboard#mountaincar-v0

