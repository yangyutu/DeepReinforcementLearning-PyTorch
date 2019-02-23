import simplejson as json
import numpy as np
import math
import torch

class Agent(object):
    """ Abstract base class for all agent


    """
    def __init__(self, config=None):
        self.globalStepCount = 0

        self.config = config

        self.device = 'cpu'
        if 'device' in self.config and torch.cuda.is_available():
            self.device = self.config['device']

        self.epsThreshold = self.config['epsThreshold']

        self.epsilon_start = self.epsThreshold
        self.epsilon_final = self.epsThreshold
        self.epsilon_decay = 1000

        if 'epsilon_start' in self.config:
            self.epsilon_start = self.config['epsilon_start']
        if 'epsilon_final' in self.config:
            self.epsilon_final = self.config['epsilon_final']
        if 'epsilon_decay' in self.config:
            self.epsilon_decay = self.config['epsilon_decay']

        self.epsilon_by_step = lambda step: self.epsilon_final + (
                    self.epsilon_start - self.epsilon_final) * math.exp(-1. * step / self.epsilon_decay)
        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']

        self.losses = []
        self.rewards = []


    def train(self):
        pass

    def test(self):
        pass

    def save_config(self, fileName):
        with open(fileName, 'w') as f:
            json.dump(self.config, f)

    def saveLosses(self, fileName):
        np.savetxt(fileName, np.array(self.losses), fmt='%.5f', delimiter='\t')

    def saveRewards(self, fileName):
        np.savetxt(fileName, np.array(self.rewards), fmt='%.5f', delimiter='\t')




