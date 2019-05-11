

from Agents.DDPG.DDPG import DDPGAgent
from Env.CustomEnv.StablizerTwoD import StablizerTwoDContinuous
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
        return self.forward(state)



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
config['actorLearningRate'] = 0.001
config['criticLearningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['episodeLength'] = 200
env = StablizerTwoDContinuous()
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

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)

xSet = np.linspace(-4,4,100)
ySet = np.linspace(-4,4,100)
policyX = np.zeros((100, 100))
policyY = np.zeros((100, 100))

value = np.zeros((100, 100))
for i, x in enumerate(xSet):
    for j, y in enumerate(ySet):
        state = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        action = agent.actorNet.select_action(state, noiseFlag = False)
        value[i, j] = agent.criticNet.forward(state, action).item()
        action = action.detach().numpy()
        policyX[i, j] = action[0][0]
        policyY[i, j] = action[0][1]



np.savetxt('StabilizerPolicyXBeforeTrain.txt', policyX, fmt='%f')
np.savetxt('StabilizerPolicyYBeforeTrain.txt', policyY, fmt='%f')

np.savetxt('StabilizerValueBeforeTrain.txt', value, fmt='%f')

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
xSet = np.linspace(-4,4,100)
ySet = np.linspace(-4,4,100)
policyX = np.zeros((100, 100))
policyY = np.zeros((100, 100))

value = np.zeros((100, 100))
for i, x in enumerate(xSet):
    for j, y in enumerate(ySet):
        state = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        action = agent.actorNet.select_action(state, noiseFlag = False)
        value[i, j] = agent.criticNet.forward(state, action).item()
        action = action.detach().numpy()
        policyX[i, j] = action[0][0]
        policyY[i, j] = action[0][1]



np.savetxt('StabilizerPolicyXAfterTrain.txt', policyX, fmt='%f')
np.savetxt('StabilizerPolicyYAfterTrain.txt', policyY, fmt='%f')
np.savetxt('StabilizerValueAfterTrain.txt', value, fmt='%f')


#np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%d')

#plotPolicy(xSet, policy)


