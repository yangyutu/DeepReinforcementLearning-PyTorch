

from Agents.ActorCriticVanilla.ActorCriticVanilla import ActorCriticOneNet
from Agents.Core.MLPNet import MultiLayerNetActorCritic
import json
import gym
from torch import optim
from copy import deepcopy
import torch

from Env.CustomEnv.CartPoleEnvCustom import CartPoleEnvCustom
torch.manual_seed(10)
# first construct the neutral network

config = dict()

config['trainStep'] = 4000
config['gamma'] = 0.99
config['learningRate'] = 0.001
config['netGradClip'] = 0.5
config['logFlag'] = False
config['logFileName'] = ''
config['logFrequency'] = 50
config['numStepsPerSweep'] = 30
config['randomSeed'] = 1
config['device'] = 'cpu'

# Get the environment and extract the number of actions.
# env = CartPoleEnvCustom()
trainEnv = gym.make("CartPole-v0")
testEnv = gym.make("CartPole-v0")
N_S = trainEnv.observation_space.shape[0]
N_A = trainEnv.action_space.n

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = [128, 128]
netParameter['n_output'] = N_A

actorCriticNet = MultiLayerNetActorCritic(netParameter['n_feature'],
                                netParameter['n_hidden'],
                                N_A)


optimizer = optim.Adam(actorCriticNet.parameters(), lr=config['learningRate'])

agent = ActorCriticOneNet(actorCriticNet, [trainEnv, testEnv], optimizer, torch.nn.MSELoss(), N_A, config)


agent.train()

agent.test(100)