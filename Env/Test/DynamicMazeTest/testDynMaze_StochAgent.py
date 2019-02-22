from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze
import random
import matplotlib.pyplot as plt
import os

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
env = DynamicMaze(config)
env.reset()
for i in range(10):
     action = random.randint(0, env.nbActions - 1)
     state, reward, done, info = env.step(action)
     print('step ' + str(i))
     print(state)
     print(reward)
     print(done)
     print(info)
     if done:
          print("great!!!!!!!!!!!!!!!!!!!")
          break
#     #maze.renderMapAndObs(directory +'/' + config['mazeFileName'].split(',')[0] + str(i) + '.png')