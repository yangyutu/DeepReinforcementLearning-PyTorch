
from Agents.TDDDPG.TD3MultiStageUnit import TD3MultiStageUnit
from Agents.MultistageController.MultiStageController import MultiStageStackedController
from Env.CustomEnv.StablizerMultiD import StablizerMultiDContinuous
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
from Env.CustomEnv.TwoArmRobot.TwoArmRobotEnv import TwoArmEnvironmentContinuousTwoStage
torch.manual_seed(1)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
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
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)

env = TwoArmEnvironmentContinuousTwoStage(config)
N_S = env.stateDim
N_A = env.nbActions
numStages = 2
agents = []
for i in range(numStages):

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
    criticNetTwo = Critic(netParameter['n_feature'] + N_A,
                                        netParameter['n_hidden'])
    criticTargetNet = deepcopy(criticNet)
    criticTargetNetTwo = deepcopy(criticNetTwo)

    actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
    criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])
    criticOptimizerTwo = optim.Adam(criticNetTwo.parameters(), lr=config['criticLearningRate'])

    actorNets = {'actor': actorNet, 'target': actorTargetNet}
    criticNets = {'criticOne': criticNet, 'criticTwo': criticNetTwo, 'targetOne': criticTargetNet, 'targetTwo': criticTargetNetTwo}
    optimizers = {'actor': actorOptimizer, 'criticOne':criticOptimizer, 'criticTwo': criticOptimizerTwo}
    agent = TD3MultiStageUnit(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)
    agents.append(agent)

controller = MultiStageStackedController(config, agents, env)

checkpoint = torch.load('Log/Stage1Finalepoch36000_checkpoint.pt')
controller.agents[1].actorNet.load_state_dict(checkpoint['actorNet_state_dict'])
controller.agents[1].actorNet_target.load_state_dict(checkpoint['actorNet_state_dict'])
controller.agents[1].criticNetOne.load_state_dict(checkpoint['criticNetOne_state_dict'])
controller.agents[1].criticNet_targetOne.load_state_dict(checkpoint['criticNetOne_state_dict'])
controller.agents[1].criticNetTwo.load_state_dict(checkpoint['criticNetTwo_state_dict'])
controller.agents[1].criticNet_targetTwo.load_state_dict(checkpoint['criticNetTwo_state_dict'])


dx = np.linspace(-0.2, 0.2, 20)
dy = np.linspace(-0.2, 0.2, 20)

dX, dY = np.meshgrid(dx, dy)

value = np.zeros(dX.shape)

for i in range(value.shape[0]):
    for j in range(value.shape[1]):
        target = config['targetState']
        controller.env.targetState = np.array(target)
        x = target[0] + dX[i, j]
        y = target[1] + dY[i, j]
        theta1, theta2 = controller.env.inverseKM(x, y)
        controller.env.currentState = np.array([theta1, theta2])
        observation = controller.env.constructObservation()
        controller.env.stageID = 1
        observation[4:6] /= controller.env.stageDistanceScales[controller.env.stageID]
        stateTorch = torch.tensor([observation], dtype=torch.float32)
        value[i, j] = controller.agents[1].evaluate_state_value(stateTorch)

np.savetxt('testValue.txt', value, fmt='%+.3f')


for i in range(value.shape[0]):
    for j in range(value.shape[1]):
        target = config['targetState']
        controller.env.targetState = np.array(target)
        x = target[0] + dX[i, j]
        y = target[1] + dY[i, j]
        print('initial position', x, y)
        theta1, theta2 = controller.env.inverseKM(x, y)
        controller.env.currentState = np.array([theta1, theta2])
        controller.env.stageID = 1
        controller.env.done = {'stage': [True, False], 'global': False}
        controller.env.stepCount = 0
        observation = controller.env.constructObservation()
        observation[4:6] /= controller.env.stageDistanceScales[controller.env.stageID]

        for k in range(50):

            action = controller.agents[1].select_action(state=observation, noiseFlag=False)
            nextState,  reward, done, _ = controller.env.step(action)

            observation = nextState['state']

            if done['global']:
                break
        print('done in step:', k)
