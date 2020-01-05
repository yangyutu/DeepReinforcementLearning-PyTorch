from collections import namedtuple
import random
import torch
import numpy as np
from copy import deepcopy

# Define a namedtuple with name Transition and attributes of state, action, next_state, reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """class to store experience and sample minibatch experience for training
        # Argument
        capacity: number of experiences to store. Depending on the complexicity of problem, typical capacity ranges from 1k to 1M.
        """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(args) == 1 and isinstance(*args, Transition):
            transition = args[0]
        else:
            transition = Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)

        # write on the earlier experience
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # sample of minibatch of experiences
        return random.sample(self.memory, batch_size)

    def fetch_all(self):
        return self.memory

    def fetch_all_random(self):
        out = deepcopy(self.memory)
        random.shuffle(out)
        return out

    def clear(self):
        self.memory.clear()
        self.position = 0

    def totensor(self):
        return torch.tensor(self.memory)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return str(self.memory)

    def write_to_text(self, fileName):
        with open(fileName, 'w') as writeFile:
            for e in self.memory:
                writeFile.write(e.__repr__() + '\n')


