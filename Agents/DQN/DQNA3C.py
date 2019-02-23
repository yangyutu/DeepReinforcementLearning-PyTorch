from Agents.DQN.DQN import DQNAgent
from copy import deepcopy
from Agents.Core.Agent import Agent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.PrioritizedReplayMemory import PrioritizedReplayMemory
from utils.utils import torchvector
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import os
import math
import torch.multiprocessing as mp
from torch.multiprocessing import current_process

class DQNA3CWorker(DQNAgent):
    def __init__(self, localNet, env, globalNet, globalOptimizer, netLossFunc, nbAction, name, stateProcessor = None, **kwargs):
        super(DQNA3CWorker, self).__init__(localNet, None, env, globalOptimizer, netLossFunc, nbAction, stateProcessor, **kwargs)
        self.globalNet = globalNet
        self.identifier = name
        self.globalOptimizer = globalOptimizer
        self.localNet = self.policyNet
        self.device = 'cpu'

    def update_net(self, state, action, nextState, reward):
        # first store memory

        self.store_experience(state, action, nextState, reward)

        if len(self.memory) < self.trainBatchSize:
            return

        transitions_raw = self.memory.sample(self.trainBatchSize)

        transitions = Transition(*zip(*transitions_raw))

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state = self.stateProcessor(transitions.state)
            nextState = self.stateProcessor(transitions.next_state)
        else:
            state = torch.tensor(transitions.state, dtype=torch.float32)
            nextState = torch.tensor(transitions.next_state, dtype=torch.float32)

        action = torch.tensor(transitions.action, dtype=torch.long).unsqueeze(-1) # shape(batch, 1)
        reward = torch.tensor(transitions.reward, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)

        batchSize = reward.shape[0]

        QValues = self.policyNet(state).gather(1, action)
        # note that here use policyNet for target value
        QNext = self.policyNet(nextState).detach()
        targetValues = reward + self.gamma * QNext.max(dim=1)[0].unsqueeze(-1)

        loss = torch.mean(self.netLossFunc(QValues, targetValues))

        self.optimizer.zero_grad()

        loss.backward()

        # for lp, gp in zip(self.localNet.parameters(), self.globalNet.parameters()):
        #     gp._grad = lp._grad
        #
        # if self.netGradClip is not None:
        #     torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(), self.netGradClip)
        #
        # # global net update
        # self.globalOptimizer.step()
        #
        # # update local net
        # self.localNet.load_state_dict(self.globalNet.state_dict())

        if self.globalStepCount % self.lossRecordStep == 0:
            self.losses.append([self.globalStepCount, self.epIdx, loss])

    def test_multiProcess(self):
        print("Hello, World! from " + current_process().name + "\n")
        print(self.globalNet.state_dict())
        for gp in self.globalNet.parameters():
            gp.grad = torch.ones_like(gp)
            #gp.grad.fill_(1)

        self.globalOptimizer.step()

        print('globalNetID:')
        print(id(self.globalNet))
        print('globalOptimizer:')
        print(id(self.globalOptimizer))
        print('localNetID:')
        print(id(self.localNet))


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class DQNA3CMaster:
    def __init__(self, config, policyNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None):
        self.config = config
        self.globalNet = policyNet
        self.policyNet = policyNet
        self.globalNet.share_memory()

        self.env = env
        self.optimizer = optimizer
        self.nbAction = nbAction
        self.netLossFunc = netLossFunc
        self.stateProcessor = stateProcessor

        self.numWorkers = self.config['numWorkers']
        self.construct_works()

    def construct_works(self):
        self.workers = []
        for i in range(self.numWorkers):
            localEnv = deepcopy(self.env)
            localNet = deepcopy(self.globalNet)
            worker = DQNA3CWorker(localNet, localEnv, self.globalNet, self.optimizer, self.netLossFunc, self.nbAction, 'worker'+str(i), self.stateProcessor, config=self.config)
            self.workers.append(worker)

    def test_multiProcess(self):

        for gp in self.globalNet.parameters():
            gp.data.fill_(0.0)

        print('initial global net state dict')
        print(self.globalNet.state_dict())

        processes = [mp.Process(target=w.test_multiProcess) for w in self.workers]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('Final global net state dict')
        print(self.globalNet.state_dict())

    def train(self):

        processes = [mp.Process(target=w.train) for w in self.workers]
        for p in processes:
            p.start()

        for p in processes:
            p.join()


    def getPolicy(self, state):
        return self.workers[0].getPolicy(state)

    def testPolicyNet(self, episodes, memory = None):
        return self.workers[0].testPolicyNet(episodes, memory)