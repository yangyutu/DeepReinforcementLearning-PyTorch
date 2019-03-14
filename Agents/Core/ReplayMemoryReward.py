from collections import namedtuple
import random
import math
import torch
import numpy as np
from copy import deepcopy
from Agents.Core.ReplayMemory import Transition
from Agents.Core.ReplayMemory import ReplayMemory
from collections import deque

# a replay memory with enhanced sampling on terminal states
class ReplayMemoryReward(ReplayMemory):

    def __init__(self, capacity, nStepBackup, gamma, terminalRatio = 0.3):
        super(ReplayMemoryReward, self).__init__(capacity)
        self.terminalMemory = []
        self.nStepBuffer = []
        self.gamma = gamma
        self.nStepBackup = nStepBackup
        self.terminalRatio = terminalRatio
        self.positionOne = 0
        self.positionTwo = 0

    def push(self, *args):
        """Saves a transition"""
        if len(args) == 1 and isinstance(*args, Transition):
            transition = args[0]
        else:
            transition = Transition(*args)

        # if it is terminal state
        if transition.next_state is None:

            if len(self.terminalMemory) < self.capacity:
                self.terminalMemory.append(None)

            self.terminalMemory[self.positionTwo] = transition
            self.positionTwo = (self.positionTwo + 1) % self.capacity
            count = 1
            R = transition.reward
            for trans in self.nStepBuffer[::-1]:
                # if nstepBackup is zero, then this is no backup
                if count > self.nStepBackup:
                    break
                R = trans.reward + self.gamma * R
                transNew = Transition(trans.state, trans.action, None, R)
                if len(self.terminalMemory) < self.capacity:
                    self.terminalMemory.append(None)
                self.terminalMemory[self.positionTwo] = transNew
                self.positionTwo = (self.positionTwo + 1) % self.capacity
                count += 1

            self.nStepBuffer.clear()
        else: # if it is terminal state
            if len(self.memory) < self.capacity:
                self.memory.append(None)

            self.nStepBuffer.append(transition)
            # write on the earlier experience
            self.memory[self.positionOne] = transition
            self.positionOne = (self.positionOne + 1) % self.capacity

    def sample(self, batchSize):
        batchSizeTwo = min(int(math.floor(batchSize*self.terminalRatio)), len(self.terminalMemory))
        batchSizeOne = batchSize - batchSizeTwo

        sampleOne = random.sample(self.memory, batchSizeOne)
        sampleTwo = random.sample(self.terminalMemory, batchSizeTwo)
        return sampleOne + sampleTwo




