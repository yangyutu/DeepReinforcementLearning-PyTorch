
import json

import numpy as np
import torch
import math
from Env.CustomEnv.ThreeDNavigation.activeParticle3DEnv import ActiveParticle3DEnv
torch.manual_seed(1)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)


def analyticalController(config, state):

    #Here I assume the state is given by six dimensional vector
    # orientation vector and the scaled target distance to the robot center

    orient = state[0:3]
    targetDistance = state[3:] * config['distanceScale']

    projectedDist = np.dot(orient, targetDistance)

    threshold = config['maxSpeed'] * config['modelNStep'] * config['dt']

    speed = 0.0
    normalProjectedDist = projectedDist / np.linalg.norm(targetDistance)
    if projectedDist > 0.0 and normalProjectedDist > 0.707:
        speed = min(1.0, projectedDist / threshold)

    return np.array([speed])





config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [15, 15, 15, 1, 0, 0]
config['targetState'] = [15, 15, 15]
config['filetag'] = 'Traj/test'
config['trajOutputFlag'] = True
config['trajOutputInterval'] = 10
config['finishThresh'] = 3
config['randomMoveFlag'] = True


with open('config_test.json', 'w') as f:
    json.dump(config, f)

env = ActiveParticle3DEnv('config_test.json',1)


delta = []



delta = np.array([[-15, -15, -15], [15, 15, -15], [-15, 15, 15], [15, 15, 15]])
delta = []

for x in [-15, 0, 15]:
    for y in [-15, 0, 15]:
        for z in [-15, 0, 15]:
            delta.append([x, y, z])

del delta[13]

delta = []

for x in [-15, 15]:
    for y in [-15,  15]:
        for z in [-15,  15]:
            delta.append([x, y, z])




delta = np.array(delta)
delta = delta * 2
targets = delta + config['currentState'][:3]
print(targets)

nTargets = len(targets)
nTraj = 1
endStep = 1000
for j in range(nTargets):
    recorder = []

    for i in range(nTraj):
        print(i)
        env.config['targetState'] = targets[j]
        state = env.reset()

        done = False
        rewardSum = 0
        stepCount = 0
        info = [i, stepCount] + env.currentState.tolist() + env.targetState.tolist() + [0.0 ]
        recorder.append(info)
        for stepCount in range(endStep):
            action = analyticalController(config, state)
            nextState, reward, done, info = env.step(action)
            info = [i, stepCount] + env.currentState.tolist() + env.targetState.tolist() + action.tolist()
            recorder.append(info)
            state = nextState
            rewardSum += reward
            if done:
                print("done in step count: {}".format(stepCount))
                #break
        print("reward sum = " + str(rewardSum))
        print(state)

    recorderNumpy = np.array(recorder)
    np.savetxt('Traj/testTraj_target_'+str(j)+'.txt', recorder, fmt='%.3f')