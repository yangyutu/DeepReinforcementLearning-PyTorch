import os
import torch
import json
import random
import numpy as np
import math

class BaseDQNAgent(object):
    """Abstract base class for DQN based agents.
        This base class contains common routines to perform basic initialization, action selection, and logging
        # Arguments
            config: a dictionary for training parameters
            policyNet: neural network for Q learning
            targetNet: a slowly changing policyNet to provide Q value targets
            env: environment for the agent to interact. env should implement same interface of a gym env
            optimizer: a network optimizer
            netLossFunc: loss function of the network, e.g., mse
            nbAction: number of actions
            stateProcessor: a function to process output from env, processed state will be used as input to the networks
            experienceProcessor: additional steps to process an experience
        """
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
        self.runningAvgEpisodeReward = 0.0

    def read_config(self):
        '''
        read parameters from self.config object
        initialize various flags
        trainStep: number of episodes to train
        targetNetUpdateStep: frequency in terms of training steps/episodes to reset target net
        trainBatchSize: mini batch size for gradient decent.
        gamma: discount factor
        netGradClip: gradient clipping parameter
        netUpdateOption: allowed strings are targetNet, policyNet, doubleQ
        verbose: bool, default false.
        nStepForward: multiple-step forward Q learning, default 1
        lossRecordStep: frequency to store loss.
        episodeLength: maximum steps in an episode
        netUpdateFrequency: frequency to perform gradient decent
        netUpdateStep: number of steps for gradient decent
        epsThreshold: const epsilon used throughout the training. Will be overridden by epsilon start, epsilon end, epsilon decay
        epsilon_start: start epsilon for scheduled epsilon exponential decay
        epsilon_final: end epsilon for scheduled epsilon exponential decay
        epsilon_decay: factor for exponential decay of epsilon
        device: cpu or cuda
        randomSeed
        hindSightER: bool variable for hindsight experience replay
        hindSightERFreq: frequency to perform hindsight experience replay
        return: None
        '''


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
        if self.netUpdateOption not in ['targetNet', 'policyNet', 'doubleQ']:
            raise Exception('netUpdateOption is invalid')

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


        self.epsThreshold = 0.1
        if 'epsThreshold' in self.config:
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

        self.netUpdateStep = 1
        if 'netUpdateStep' in self.config:
            self.netUpdateStep = self.config['netUpdateStep']

    def select_action(self, net=None, state=None, epsThreshold=None, noiseFlag = True):
        '''
        select action based on epsilon rule
        # Arguments
        net: which net used for action selection. default is policyNet
        state: observation or state as the input to the net
        epsThreshold: epsilon to used
        noiseFlag: if set False, will ignore epsilon and perform greedy selection.

        return: integer index with base 0
        '''
        if net is None:
            net = self.policyNet
        if epsThreshold is None:
            epsThreshold = self.epsThreshold

        randNum = random.random()
        # get a random number so that we can do epsilon exploration
        if noiseFlag and randNum < epsThreshold:
            action = random.randint(0, self.numAction - 1)
        else:
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

        return action

    def getPolicy(self, state):
        return self.select_action(net=self.policyNet, state=state, noiseFlag=False)

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