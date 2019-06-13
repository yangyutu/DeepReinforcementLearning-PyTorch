

import os
import torch
import json
import random
import numpy as np
import math

class BaseDQNAgent(object):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, experienceProcessor=None):
        self.config = config
        self.read_config()
        self.policyNet = policyNet
        self.targetNet = targetNet
        self.env = env
        self.optimizer = optimizer
        self.numAction = nbAction
        self.stateProcessor = stateProcessor
        self.experienceProcessor = experienceProcessor
        self.netLossFunc = netLossFunc
        self.initialization()

    def initialization(self):
        # move model to correct device
        self.policyNet = self.policyNet.to(self.device)

        # in case targetNet is None
        if self.targetNet is not None:
            self.targetNet = self.targetNet.to(self.device)

        self.dirName = 'Log/'
        if 'dataLogFolder' in self.config:
            self.dirName = self.config['dataLogFolder']
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)

        self.identifier = ''
        self.epIdx = 0
        self.learnStepCounter = 0  #for target net update
        self.globalStepCount = 0
        self.losses = []
        self.rewards = []
        self.nStepBuffer = []

    def read_config(self):
        self.trainStep = self.config['trainStep']
        self.targetNetUpdateStep = 10000
        if 'targetNetUpdateStep' in self.config:
            self.targetNetUpdateStep = self.config['targetNetUpdateStep']

        self.trainBatchSize = self.config['trainBatchSize']
        self.gamma = self.config['gamma']

        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']
        self.netUpdateOption = 'targetNet'
        if 'netUpdateOption' in self.config:
            self.netUpdateOption = self.config['netUpdateOption']
        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']
        self.netUpdateFrequency = 1
        if 'netUpdateFrequency' in self.config:
            self.netUpdateFrequency = self.config['netUpdateFrequency']
        self.nStepForward = 1
        if 'nStepForward' in self.config:
            self.nStepForward = self.config['nStepForward']
        self.lossRecordStep = 500
        if 'lossRecordStep' in self.config:
            self.lossRecordStep = self.config['lossRecordStep']
        self.episodeLength = 500
        if 'episodeLength' in self.config:
            self.episodeLength = self.config['episodeLength']

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

        self.device = 'cpu'
        if 'device' in self.config and torch.cuda.is_available():
            self.device = self.config['device']


        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        random.seed(self.randomSeed)

        self.hindSightER = False
        if 'hindSightER' in self.config:
            self.hindSightER = self.config['hindSightER']
            self.hindSightERFreq = self.config['hindSightERFreq']

    def select_action(self, net, state, epsThreshold):

        # get a random number so that we can do epsilon exploration
        randNum = random.random()
        if randNum > epsThreshold:
            with torch.no_grad():
                # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
                # here state[np.newaxis,:] is to add a batch dimension
                if self.stateProcessor is not None:
                    state, _ = self.stateProcessor([state], self.device)
                    QValues = net(state)
                else:
                    stateTorch = torch.from_numpy(np.array(state[np.newaxis, :], dtype = np.float32))
                    QValues = net(stateTorch.to(self.device))
                action = torch.argmax(QValues).item()
        else:
            action = random.randint(0, self.numAction-1)
        return action

    def getPolicy(self, state):
        return self.select_action(self.policyNet, state, -0.01)

    def train(self):
        raise NotImplementedError

    def perform_random_exploration(self, episodes, memory=None):
        raise NotImplementedError
    def perform_on_policy(self, episodes, policy, memory=None):
        raise NotImplementedError

    def testPolicyNet(self, episodes, memory = None):
        raise NotImplementedError

    def save_all(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self, prefix):
        raise NotImplementedError

    def save_config(self, fileName):
        with open(fileName, 'w') as f:
            json.dump(self.config, f)

    def saveLosses(self, fileName):
        np.savetxt(fileName, np.array(self.losses), fmt='%.5f', delimiter='\t')

    def saveRewards(self, fileName):
        np.savetxt(fileName, np.array(self.rewards), fmt='%.5f', delimiter='\t')

    def loadLosses(self, fileName):
        self.losses = np.genfromtxt(fileName).tolist()
    def loadRewards(self, fileName):
        self.rewards = np.genfromtxt(fileName).tolist()