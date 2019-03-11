

from Agents.DQN.DQNAsynER import DQNAsynERMaster, SharedAdam
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
config['epsilon_start'] = 0.3
config['epsilon_final'] = 0.05
config['epsilon_decay'] = 200
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 200000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.0001
config['netGradClip'] = 1
config['logFlag'] = False
config['logFileName'] = ''
config['logFrequency'] = 50
config['netUpdateOption'] = 'doubleQ'
config['nStepForward'] = 3
config['priorityMemoryOption'] = False
config['numWorkers'] = 7

envs = []
for i in range(config['numWorkers']):
    env = MountainCarEnvCustom()
    envs.append(env)

N_S = envs[0].observation_space.shape[0]
N_A = envs[0].action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])
targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])

agent = DQNAsynERMaster(policyNet, targetNet, envs, optimizer, torch.nn.MSELoss(reduction='mean') ,N_A, config=config)

agent.train()

# benchmark score
# https://github.com/openai/gym/wiki/Leaderboard#mountaincar-v0

