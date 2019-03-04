

from Agents.ActorCriticVanilla.ActorCriticVanilla import ActorCriticVanilla
from Agents.Core.MLPNet import MultiLayerNetRegression, MultiLayerNetSoftmax
import json
import gym
from torch import optim
from copy import deepcopy
import torch

from Env.CustomEnv.CartPoleEnvCustom import CartPoleEnvCustom
torch.manual_seed(10)
# first construct the neutral network

config = dict()

config['trainStep'] = 2000
config['gamma'] = 0.99
config['learningRate'] = 0.005
config['netGradClip'] = 0.5
config['logFlag'] = False
config['logFileName'] = ''
config['logFrequency'] = 50
config['numStepsPerSweep'] = 30
config['randomSeed'] = 1
config['device'] = 'cuda'

# Get the environment and extract the number of actions.
# env = CartPoleEnvCustom()
trainEnv = gym.make("CartPole-v0")
testEnv = gym.make("CartPole-v0")
N_S = trainEnv.observation_space.shape[0]
N_A = trainEnv.action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [40, 40]
netParameter['n_output'] = N_A

actorNet = MultiLayerNetSoftmax(netParameter['n_feature'],
                                netParameter['n_hidden'],
                                N_A)

criticNet = MultiLayerNetRegression(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    1)

optimizer1 = optim.Adam(actorNet.parameters(), lr=config['learningRate'])
optimizer2 = optim.Adam(criticNet.parameters(), lr=config['learningRate'])

agent = ActorCriticVanilla(actorNet, criticNet, [trainEnv, testEnv], [optimizer1, optimizer2], torch.nn.MSELoss(), N_A, config)


agent.train()

agent.test(100)