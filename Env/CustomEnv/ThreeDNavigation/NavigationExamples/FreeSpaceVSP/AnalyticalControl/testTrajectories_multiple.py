

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
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv

def analyticalController(config, state):

    #Here I assume the state is given by six dimensional vector
    # orientation vector and the scaled target distance to the robot center

    orient = state[0:3]
    targetDistance = state[3:] * config['distanceScale']

    projectedDist = np.dot(orient, targetDistance)

    threshold = config['maxSpeed'] * config['modelNStep'] * config['dt']

    speed = 0.0
    normalProjectedDist = projectedDist / np.linalg.norm(targetDistance)
    if projectedDist > 0.0 and normalProjectedDist > 0.7:
        speed = min(1.0, projectedDist / threshold)


    return np.array([speed])


configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)


config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [0, 0, 0, 0, 0, 1]
config['targetState'] = [0, 0, 10000]
config['filetag'] = 'test'
config['trajOutputInterval'] = 100
config['trajOutputFlag'] = False
config['randomMoveFlag'] = True
config['finishThresh'] = 3


with open('config_test.json', 'w') as f:
    json.dump(config, f)

env = ActiveParticle3DEnv('config_test.json',1)


nTraj = 200
endStep = 250
outputFreq = 5
recorder = []
dataSet = []
for i in range(nTraj):
    print(i)
    trajList = []

    randomOrient = np.random.randn(3)
    randomOrient /= np.linalg.norm(randomOrient)

    env.config['currentState'][3:] = randomOrient

    state = env.reset()

    x = env.currentState[0]
    y = env.currentState[1]
    z = env.currentState[2]

    x0 = x
    y0 = y
    z0 = z
    done = False
    rewardSum = 0
    stepCount = 0

    trajList.append([0, x - x0, y - y0, z - z0, (x - x0) ** 2 + (y - y0) ** 2 + (z - z0)**2])
    while not done:
        action = analyticalController(config, state)
        nextState, reward, done, info = env.step(action)
        stepCount += 1
        recorder.append([i, stepCount] + env.currentState.tolist())
        state = nextState
        rewardSum += reward
        x = env.currentState[0]
        y = env.currentState[1]
        z = env.currentState[2]

        if (stepCount + 1) % outputFreq == 0:
            trajList.append([stepCount + 1, x - x0, y - y0,  z - z0, (x - x0) ** 2 + (y - y0) ** 2 + (z - z0)**2])

        if stepCount > endStep:
            break
    dataSet.append(trajList)


trajData = np.array(dataSet)
meanTrajData = np.mean(trajData, axis = 0)
stdTrajData = np.std(trajData, axis = 0)

plt.figure(1)
plt.plot(meanTrajData[:,0]/10, meanTrajData[:,3])
np.savetxt('meanTrajDataCorrect.txt',meanTrajData)
np.savetxt('stdTrajDataCorrect.txt',stdTrajData)
np.savetxt('trajDataX.txt',trajData[:,:,1])
np.savetxt('trajDataY.txt',trajData[:,:,2])