from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze
import random
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
from torch.multiprocessing import current_process
from copy import deepcopy
import numpy as np
mat = np.genfromtxt('simpleMap.txt')

#
# import os
#
#
config = {}
config['mapName'] = 'simpleMapSmall'
config['JumpMatrix'] = 'trajSampleHalf.npz'
config['numCircObs'] = 2
config['dynamicObsFlag'] = False
config['agentReceptHalfWidth'] = 5
config['obstacleMapPaddingWidth'] = 10
config['targetState'] = (4, 4)
config['dynamicTargetFlag'] = False
config['stochAgent'] = True
config['currentState'] = (1.0, 1.0, 0.0)
config['dynamicInitialStateFlag'] = True
#
# #directory = config['mazeFileName'].split('.')[0]
# #if not os.path.exists(directory):
# #    os.makedirs(directory)
#
# print(os.getcwd())
#
#



def runEnv(env):
    env.reset()
    print("Hello, World! from " + current_process().name + "\n")
    print('agentID')
    print(id(env.agent))
    print('MapID')
    print(id(env.mapMat))
    print('envID')
    print(id(env))
    for i in range(4):
         action = random.randint(0, env.nbActions - 1)
         state, reward, done, info = env.step(action)
         print("Hello, World! from " + current_process().name + "\n")
         print('step ' + str(i))
         print(state)
         print(reward)
         print(done)
         print(info)


         if done:
              print("great!!!!!!!!!!!!!!!!!!!")
              break
#     #maze.renderMapAndObs(directory +'/' + config['mazeFileName'].split(',')[0] + str(i) + '.png')
#
nProcess = 4
env = DynamicMaze(config)
#
processes = []
for i in range(nProcess):
    localEnv = deepcopy(env)
    print('envID')
    print(id(localEnv))
    processes.append(mp.Process(target=runEnv, args=(localEnv,)))

for p in processes:
    p.start()

for p in processes:
    p.join()
