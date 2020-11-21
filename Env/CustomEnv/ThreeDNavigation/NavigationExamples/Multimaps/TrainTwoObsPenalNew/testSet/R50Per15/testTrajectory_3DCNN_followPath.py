from Agents.DDPG.DDPG import DDPGAgent
from utils.netInit import xavier_init
import json
from torch import optim
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.OUNoise import OUNoise
#from activeParticleEnv import ActiveParticleEnvMultiMap, ActiveParticleEnv
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv, RBCObstacle
from Env.CustomEnv.ThreeDNavigation.NavigationExamples.optimalSearch.pathGuider import PathGuiderStraightLine



import math
torch.manual_seed(1)


# Convolutional neural network (two convolutional layers)
class CriticConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(CriticConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth, inputWidth)
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv3d(1,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1

        self.fc0 = nn.Linear(6 + num_action, 256)
        self.fc1 = nn.Linear(self.featureSize() + 256, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        self.apply(xavier_init)
    def forward(self, state, action):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        #xout.fill_(0)
        yout = F.relu(self.fc0(torch.cat((y, action), 1)))
        #actionOut = F.relu(self.fc0_action(action))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 1, *self.inputShape))).view(1, -1).size(1)

# Convolutional neural network (two convolutional layers)
class ActorConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(ActorConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth, inputWidth)
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv3d(1,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1

        # 6 dim for state
        self.fc0 = nn.Linear(6, 256)
        self.fc1 = nn.Linear(self.featureSize() + 256, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_action)
        self.apply(xavier_init)
        self.noise = OUNoise(num_action, seed=1, mu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.1, decay_period=1000000)
        self.noise.reset()

    def forward(self, state):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        #xout.fill_(0)
        yout = F.relu(self.fc0(y))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = torch.tanh(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 1, *self.inputShape))).view(1, -1).size(1)

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)



def stateProcessor(state, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)

    senorList = [item['sensor'] for item in state if item is not None]
    targetList = [item['target'] for item in state if item is not None]
    nonFinalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
              'target': torch.tensor(targetList, dtype=torch.float32, device=device)}
    return nonFinalState, nonFinalMask

def experienceProcessor(state, action, nextState, reward, info):
    if nextState is not None:
        target = info['previousTarget']
        distance = target - info['currentState'][:3]
        localDistance = np.dot(info['localFrame'], distance)
        nextState['target'][3:] = localDistance / info['scaleFactor']
    return state, action, nextState, reward



configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

def obstacleConstructorCallBack(configName = 'config_RBC.json'):

    with open(configName, 'r') as f:
        config = json.load(f)

    obstacles, obstacleCenter = [], []

    for i in range(config['numObstacles']):
        name = 'obs' + str(i)
        obstacles.append(RBCObstacle(np.array(config[name]['center']), config[name]['scale'], np.array(config[name]['orient'])))

        obstacleCenter.append(obstacles[i].center)

    return obstacles, obstacleCenter






env = ActiveParticle3DEnv('config.json',1, obstacleConstructorCallBack)

N_S = env.stateDim[0]
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 256
netParameter['n_output'] = N_A

actorNet = ActorConvNet(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

actorTargetNet = deepcopy(actorNet)

criticNet = CriticConvNet(netParameter['n_feature'] ,
                            netParameter['n_hidden'],
                        netParameter['n_output'])

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A, stateProcessor=stateProcessor, experienceProcessor=experienceProcessor)

checkpoint = torch.load('Epoch15000_checkpoint.pt')

agent.actorNet.load_state_dict(checkpoint['actorNet_state_dict'])

config['randomMoveFlag'] = True
config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [-20, -20, 1, 1, 0, 0]
config['currentState'] = [0, 0, 1, 1, 0, 0]
config['targetState'] = [15, 15, 25]
config['filetag'] = 'Traj/test'
config['trajOutputFlag'] = True
config['trajOutputInterval'] = 100
config['finishThresh'] = 2
config['gravity'] = 0
config['multiMapNames'] = ['config_RBC_R50_15PerTest.json']
config['multiMapProbs'] = [1.0]
#config['Dt'] = 1.63e-13 * 0

with open('config_test.json', 'w') as f:
    json.dump(config, f)

agent.env = ActiveParticle3DEnv('config_test.json',1, obstacleConstructorCallBack)


finalTarget = [0, 0, 499]

nTraj = 3
endStep = 700

recorder = []

guide = PathGuiderStraightLine()

for i in range(nTraj):
    print(i)
    target = guide.getTrajPos()
    agent.env.config['targetState'] = target
    state = agent.env.reset()

    done = False
    rewardSum = 0
    stepCount = 0
    info = [i, stepCount] + agent.env.currentState.tolist() + agent.env.targetState.tolist() + [0.0 for _ in range(N_A)]
    recorder.append(info)
    for stepCount in range(endStep):

        action = agent.select_action(agent.actorNet, state, noiseFlag=False)


        nextState, reward, done, infoDict = agent.env.step(action)
        info = [i, stepCount] + agent.env.currentState.tolist() + agent.env.targetState.tolist() + action.tolist()
        recorder.append(info)

        pos = agent.env.currentState[:3]
        guide.step(pos, 15)
        target = guide.getTrajPos()

        agent.env.targetState = target

        state = nextState
        rewardSum += reward

        done = (abs(finalTarget[2] - pos[2]) < 3.0)

        if done:
            print("done in step count: {}".format(stepCount))

            break
    print("reward sum = " + str(rewardSum))
    print(infoDict)
recorderNumpy = np.array(recorder)
