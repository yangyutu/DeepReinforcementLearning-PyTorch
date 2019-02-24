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




class DQNA3CWorkerV2(mp.Process):
    def __init__(self, config, localNet, env, globalNet, globalOptimizer, netLossFunc, nbAction, name,
                 globalEpisodeCount, globalEpisodeReward, globalRunningAvgReward, resultQueue, logFolder,
                stateProcessor = None):
        super(DQNA3CWorkerV2, self).__init__()
        self.config = config
        self.globalNet = globalNet
        self.identifier = name
        self.globalOptimizer = globalOptimizer
        self.localNet = localNet
        self.env = env
        self.netLossFunc = netLossFunc
        self.numAction = nbAction
        self.stateProcessor = stateProcessor
        self.device = 'cpu'
        self.epIdx = 0
        self.totalStep = 0
        self.updateGlobalFrequency = 10
        self.gamma = 0.99
        self.trainStep = self.config['trainStep']
        self.globalEpisodeCount = globalEpisodeCount
        self.globalEpisodeReward = globalEpisodeReward
        self.globalRunningAvgReward = globalRunningAvgReward
        self.resultQueue = resultQueue
        self.dirName = logFolder
        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']

    def select_action(self, net, state, epsThreshold):

        # get a random number so that we can do epsilon exploration
        randNum = random.random()
        if randNum > epsThreshold:
            with torch.no_grad():
                # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
                # here state[np.newaxis,:] is to add a batch dimension
                if self.stateProcessor is not None:
                    state = self.stateProcessor([state])
                    QValues = net(state)
                else:
                    QValues = net(torchvector(state[np.newaxis, :]).to(self.device))
                action = torch.argmax(QValues).item()
        else:
            action = random.randint(0, self.numAction-1)
        return action

    def run(self):

        bufferState, bufferAction, bufferReward, bufferNextState = [], [], [], []
        for self.epIdx in range(self.trainStep):

            print("episode index:" + str(self.epIdx) + " from" + current_process().name + "\n")
            state = self.env.reset()
            done = False
            rewardSum = 0
            stepCount = 0

            while not done:

                episode = 0.1
                action = self.select_action(self.localNet, state, episode)
                nextState, reward, done, info = self.env.step(action)

                bufferAction.append(action)
                bufferState.append(state)
                bufferReward.append(reward)
                bufferNextState.append(nextState)

                state = nextState
                rewardSum += reward



                if self.totalStep % self.updateGlobalFrequency == 0:  # update global and assign to local net
                    # sync
                    self.update_net_and_sync(bufferAction, bufferState, bufferReward, bufferNextState)
                    bufferAction.clear()
                    bufferState.clear()
                    bufferReward.clear()
                    bufferNextState.clear()

                if done:
#                    print("done in step count: {}".format(stepCount))
#                    print("reward sum = " + str(rewardSum))
                # done and print information
                #    pass
                    self.recordInfo(rewardSum, stepCount)

                stepCount += 1
                self.totalStep += 1
        #self.resultQueue.put(None)

    def recordInfo(self, reward, stepCount):
        with self.globalEpisodeReward.get_lock():
            self.globalEpisodeReward.value = reward
        with self.globalRunningAvgReward.get_lock():
            self.globalRunningAvgReward.value = (self.globalRunningAvgReward.value * self.globalEpisodeCount.value + reward) / (
                        self.globalEpisodeCount.value + 1)
        with self.globalEpisodeCount.get_lock():
            self.globalEpisodeCount.value += 1
            if self.globalEpisodeCount.value % self.config['logFrequency'] == 0:
                self.save_checkpoint()

        # resultQueue.put(globalEpisodeReward.value)
        self.resultQueue.put(
            [self.globalEpisodeCount.value, stepCount, self.globalEpisodeReward.value, self.globalRunningAvgReward.value])
        print(self.name)
        print("Episode: ", self.globalEpisodeCount.value)
        print("stepCount: ", stepCount)
        print("Episode Reward: ", self.globalEpisodeReward.value)
        print("Episode Running Average Reward: ", self.globalRunningAvgReward.value)

    def save_checkpoint(self):
        prefix = self.dirName + 'Epoch' + str(self.globalEpisodeCount.value + 1)
        #self.saveLosses(prefix + '_loss.txt')
        #self.saveRewards(prefix + '_reward.txt')
        torch.save({
            'epoch': self.globalEpisodeCount.value + 1,
            'model_state_dict': self.globalNet.state_dict(),
            'optimizer_state_dict': self.globalOptimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

    def update_net_and_sync(self, bufferAction, bufferState, bufferReward, bufferNextState):

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state = self.stateProcessor(bufferState)
            nextState = self.stateProcessor(bufferNextState)
        else:
            state = torch.tensor(bufferState, dtype=torch.float32)
            nextState = torch.tensor(bufferNextState, dtype=torch.float32)

        action = torch.tensor(bufferAction, dtype=torch.long).unsqueeze(-1) # shape(batch, 1)
        reward = torch.tensor(bufferReward, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)

        batchSize = reward.shape[0]

        QValues = self.localNet(state).gather(1, action)
        # note that here use policyNet for target value
        QNext = self.localNet(nextState).detach()
        targetValues = reward + self.gamma * QNext.max(dim=1)[0].unsqueeze(-1)

        loss = torch.mean(self.netLossFunc(QValues, targetValues))

        self.globalOptimizer.zero_grad()

        loss.backward()

        for lp, gp in zip(self.localNet.parameters(), self.globalNet.parameters()):
             gp._grad = lp._grad

        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.localNet.parameters(), self.netGradClip)

        # global net update
        self.globalOptimizer.step()
        #
        # # update local net
        self.localNet.load_state_dict(self.globalNet.state_dict())

        #if self.totalStep % self.lossRecordStep == 0:
        #    self.losses.append([self.globalStepCount, self.epIdx, loss])



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

class DQNA3CMasterV2(DQNAgent):
    def __init__(self, policyNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, **kwargs):
        super(DQNA3CMasterV2, self).__init__(policyNet, None, env, optimizer, netLossFunc, nbAction, stateProcessor, **kwargs)
        self.globalNet = policyNet
        self.globalNet.share_memory()
        self.numWorkers = self.config['numWorkers']

        self.globalEpisodeCount = mp.Value('i', 0)
        self.globalEpisodeReward = mp.Value('d', 0)
        self.globalRunningAvgReward = mp.Value('d', 0)
        self.resultQueue = mp.Queue()

        self.construct_workers()


    def construct_workers(self):
        self.workers = []
        for i in range(self.numWorkers):
            localEnv = deepcopy(self.env)
            localNet = deepcopy(self.globalNet)
            worker = DQNA3CWorkerV2(self.config, localNet, localEnv, self.globalNet, self.optimizer, self.netLossFunc,
                                    self.numAction, 'worker'+str(i), self.globalEpisodeCount, self.globalEpisodeReward,
                                    self.globalRunningAvgReward, self.resultQueue, self.dirName, self.stateProcessor)
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

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        self.resultQueue.put(None)


    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx + 1)
        torch.save({
            'epoch': self.globalEpisodeCount.value + 1,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

        self.rewards = []  # record episode reward to plot
        while True:
            r = self.resultQueue.get()
            if r is not None:
                self.rewards.append(r)
            else:
                break

        self.saveRewards(prefix + '_reward.txt')