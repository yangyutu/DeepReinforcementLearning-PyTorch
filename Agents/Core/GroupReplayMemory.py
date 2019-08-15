from Agents.DQN.DQN import DQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import os
import math
import pickle
from copy import deepcopy


class GroupReplayMemory(object):

    def __init__(self, capacity, numAgents):
        self.capacity = capacity
        self.numAgents = numAgents
        self.memory = [ []  for _ in range(self.numAgents)]
        self.position = 0

    def push(self, transitions):
        """Saves a transition"""


        if len(self.memory[0]) < self.capacity:
            for n in range(self.numAgents):
                self.memory[n].append(transitions[n])
        else:
            # write on the earlier experience
            for n in range(self.numAgents):
                self.memory[n][self.position] = transitions[n]
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        index = random.sample(range(0, len(self.memory[0])), batch_size)
        return [[self.memory[n][i] for i in index ] for n in range(self.numAgents)]

    def fetch_all(self):
        return self.memory

    def clear(self):
        for n in range(self.numAgents):
            self.memory[n].clear()
        self.position = 0

    def __len__(self):
        return len(self.memory[0])

    def __repr__(self):
        return str(self.memory)


