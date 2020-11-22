#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:55:51 2019

@author: yangyutu123
"""

from activeParticle3DEnv import ActiveParticle3DEnv
import numpy as np
import math
import random

#import activeParticleSimulatorPython as model

env = ActiveParticle3DEnv('config.json',1)

step = 200

state = env.reset()
NA = env.nbActions
print(state)

for i in range(step):
    state = env.currentState
    action = np.random.randn(NA)
    nextState, reward, action, info = env.step(action)
    print(nextState)
    print(info)
    #if i%2 == 0 and i < 10:
    #    env.step(100, np.array([u, v, 1.0]))
    #else:
    #    env.step(100, np.array([u, v, 0]))
        

env = ActiveParticle3DEnv('config_ambientField.json',1)

step = 200

state = env.reset()
NA = env.nbActions
print(state)

for i in range(step):
    state = env.currentState
    action = [0.0, 0.0]
    nextState, reward, action, info = env.step(action)
    print(nextState)
    print(info)