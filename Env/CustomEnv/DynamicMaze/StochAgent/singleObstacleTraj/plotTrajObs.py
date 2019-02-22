#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:48:35 2019

@author: yangyutu123
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
data = np.genfromtxt('singleObstacleTestTraj.txt')
mapData = np.genfromtxt('singleObstacle.txt')
(obsX, obsY) = np.where(mapData == 1)
trajIdx = 12
mapSize = 25
margin = 4
traj = data[data[:,0] == trajIdx]

particle = traj[:,2:5]
target = traj[:,5:7]
action = traj[:,7]
time = traj[:,1]
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(particle[:,0], particle[:,1], c=time)
cmap = plt.cm.jet
ax.scatter(target[:,0], target[:,1], c=time,  cmap='cool', marker='o')
ax.set_xlim([-margin, mapSize + margin])
ax.set_ylim([-margin, mapSize + margin])
ax.scatter(obsX, obsY, c='black', marker='s', s = 10)

nframe = particle.shape[0]
arrowScale = 1
skip = 3
for i in range(nframe):
    if i % skip == 0:
        dx = arrowScale*math.cos(particle[i,2])
        dy = arrowScale*math.sin(particle[i,2])
        ax.arrow(particle[i,0], particle[i,1], dx, dy, head_width=arrowScale/3, color='blue')