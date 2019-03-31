

from Agents.DQN.DQN import DQNAgent
from Agents.DQN.DQNSyn import DQNSynAgent
from Agents.Core.MLPNet import MultiLayerNetRegression
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)


config = dict()

config['trainStep'] = 100000
config['epsThreshold'] = 0.1
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 2000
config['trainBatchSize'] = 32
config['gamma'] = 0.9
config['learningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 100
config['priorityMemoryOption'] = False
config['netUpdateOption'] = 'doubleQ'
config['netUpdateFrequency'] = 1
config['priorityMemory_absErrUpper'] = 5
config['numWorkers'] = 5

env = StablizerOneD()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [100]
netParameter['n_output'] = N_A

policyNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


# we need a wrapper
def make_env(config, i):
    def _thunk():
        env = StablizerOneD(config, i)

        return env
    return _thunk

numWorkers = config['numWorkers']
envs = [make_env(config,  i) for i in range(numWorkers)]
if numWorkers > 1:
    envs = SubprocVecEnv(envs)
else:
    envs = SubprocVecEnv(envs)

agent = DQNSynAgent(config, policyNet, targetNet, envs, optimizer, torch.nn.MSELoss(reduction='none'), N_A)


agent.train()

print('done Training')
