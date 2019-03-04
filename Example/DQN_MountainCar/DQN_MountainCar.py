

from Agents.DQN.DQN import DQNAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
import json
import gym
from torch import optim
from copy import deepcopy
import torch
from Env.CustomEnv.MountainCarEnv import MountainCarEnvCustom
torch.manual_seed(1)
# first construct the neutral network

config = dict()

config['trainStep'] = 2000
config['epsThreshold'] = 0.3
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 2000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = ''
config['logFrequency'] = 50

env = MountainCarEnvCustom()

N_S = env.observation_space.shape[0]
N_A = env.action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])
targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = DQNAgent(policyNet, targetNet, env, optimizer, torch.nn.MSELoss(reduction='none') ,N_A, config=config)

agent.train()

# benchmark score
# https://github.com/openai/gym/wiki/Leaderboard#mountaincar-v0

