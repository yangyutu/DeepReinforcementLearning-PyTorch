# from Env.CustomEnv.DynamicMaze.DynamicMaze import DynamicMaze
# import random
# import matplotlib.pyplot as plt
# import os

import numpy as np
mat = np.genfromtxt('simpleMap.txt')

#
# import os
#
#
# config = {}
# config['mazeFileName'] = 'simpleMap.txt'
# config['numCircObs'] = 2
# config['dynamicObsFlag'] = False
# config['agentReceptHalfWidth'] = 5
# config['obstacleMapPaddingWidth'] = 5
#
# #directory = config['mazeFileName'].split('.')[0]
# #if not os.path.exists(directory):
# #    os.makedirs(directory)
#
# print(os.getcwd())
#
#
# env = DynamicMaze(config)
# for i in range(100):
#     action = random.randint(0, env.nbActions - 1)
#     state, reward, done, _ = env.step(action)
#     env.step()
#     #maze.renderMapAndObs(directory +'/' + config['mazeFileName'].split(',')[0] + str(i) + '.png')