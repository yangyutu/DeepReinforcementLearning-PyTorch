#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:30:33 2019

@author: yangyutu123
"""
import cppimport
import numpy as np
import math
import random
AP = cppimport.imp('ActiveParticle3DSimulatorPython')

model = AP.ActiveParticle3DSimulatorPython('config.json',1)


#import activeParticleSimulatorPython as model
step = 20
model.createInitialState(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

for i in range(step):
    
    w = np.random.rand(2)
    model.step(100, w)
    #if i%2 == 0 and i < 10:
    #    env.step(100, np.array([u, v, 1.0]))
    #else:
    #    env.step(100, np.array([u, v, 0]))
        

        
