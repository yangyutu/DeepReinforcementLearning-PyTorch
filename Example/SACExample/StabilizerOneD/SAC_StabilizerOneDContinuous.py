from Agents.SAC.SAC import SACAgent
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
from torch.distributions import Normal

torch.manual_seed(1)


class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(xavier_init)


    def forward(self, state):
        """
        Params state and actions are torch tensors
        """

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QNet, self).__init__()
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



class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs,  hidden_dim, num_actions,log_sig_min = -20, log_sig_max = 1, epsilon = 1e-6):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max

        self.epsilon = epsilon

        self.apply(xavier_init)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std

    def select_action(self, state, noiseFlag = True, probFlag=False):
        mean, log_std = self.forward(state)
        if not noiseFlag:
            return torch.tanh(mean)

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        if not probFlag:
            return action

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound. Note that derivative of tanh(x) is 1 - tanh(x)^2
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob




# first construct the neutral network
config = dict()

config['trainStep'] = 200
config['targetNetUpdateStep'] = 100
config['memoryCapacity'] = 20000
config['trainBatchSize'] = 64
config['gamma'] = 0.9
config['tau'] = 0.01
config['actorLearningRate'] = 0.001
config['softQLearningRate'] = 0.001
config['valueLearningRate'] = 0.001
config['netGradClip'] = 1
config['logFlag'] = True
config['logFileName'] = 'StabilizerOneDLog/traj'
config['logFrequency'] = 1000
config['episodeLength'] = 200
config['SACAlpha'] = 0.01
env = StablizerOneDContinuous()
N_S = env.stateDim
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 100
netParameter['n_output'] = N_A

actorNet = GaussianPolicy(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

valueNet = ValueNet(netParameter['n_feature'],
                                    netParameter['n_hidden'])

valueTargetNet = deepcopy(valueNet)

softQNetOne = QNet(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden'])

softQNetTwo = QNet(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden'])


actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
valueOptimizer = optim.Adam(valueNet.parameters(), lr=config['valueLearningRate'])
softQOneOptimizer = optim.Adam(softQNetOne.parameters(), lr=config['softQLearningRate'])
softQTwoOptimizer = optim.Adam(softQNetTwo.parameters(), lr=config['softQLearningRate'])


actorNets = {'actor': actorNet}
criticNets = {'softQOne': softQNetOne, 'softQTwo': softQNetTwo, 'value': valueNet, 'valueTarget': valueTargetNet}
optimizers = {'actor': actorOptimizer, 'softQOne':softQOneOptimizer,\
              'softQTwo':softQTwoOptimizer, 'value':valueOptimizer}

agent = SACAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)

xSet = np.linspace(-4,4,100)
policy = np.zeros_like(xSet)
value = np.zeros_like(xSet)
for i, x in enumerate(xSet):
    state = torch.tensor([x], dtype=torch.float32).unsqueeze(0)
    action = agent.actorNet.select_action(state, noiseFlag = False)
    policy[i] = agent.actorNet.select_action(state, noiseFlag = False)
    value[i] = agent.valueNet.forward(state).item()

np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%f')
np.savetxt('StabilizerValueBeforeTrain.txt', value, fmt='%f')

agent.train()

policy = np.zeros_like(xSet)
value = np.zeros_like(xSet)

for i, x in enumerate(xSet):
    state = torch.tensor([x], dtype=torch.float32).unsqueeze(0)
    action = agent.actorNet.select_action(state, noiseFlag=False)
    policy[i] = agent.actorNet.select_action(state, noiseFlag=False)
    value[i] = agent.valueNet.forward(state).item()

np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%f')
np.savetxt('StabilizerValueAfterTrain.txt', value, fmt='%f')