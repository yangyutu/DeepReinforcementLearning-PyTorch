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
from setproctitle import setproctitle as ptitle
import torch.nn.functional as F


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

class A3CWorker(mp.Process):
    def __init__(self, rank, config, localNet, env, globalNet, globalOptimizer, netLossFunc, nbAction, name,
                 globalEpisodeCount, globalEpisodeReward, globalRunningAvgReward, resultQueue, logFolder,
                stateProcessor = None):
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

        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        # assign GPU id
        self.rank = rank
        self.gpuIDs = config['gpuIDs']
        self.gpuId = self.gpuIDs[self.rank % len(self.gpuIDs)]
        if self.gpuId >= 0:
            torch.cuda.manual_seed(self.config['randomSeed'] + rank)
            # if gpu is available, then put local net on the GPU
            with torch.cuda.device(self.gpuId):
                self.localNet = self.localNet.cuda()

    def processState(self, state):
        # change state to the right format and move it the right device
        if self.stateProcessor is not None:
            state = self.stateProcessor([state])
        else:
            state(torchvector(state[np.newaxis, :]))

        if self.gpuId >= 0:
            with torch.cuda.device(self.gpu_id):
                state = state.cuda()
        return state

    def run(self):

        ptitle('Training Agent: {}').format(self.rank)
        # env is running on CPU
        bufferReward, bufferLogProbs, bufferEntropies = [], [], []
        done = True
        for self.epIdx in range(self.trainStep):

            print("episode index:" + str(self.epIdx) + " from" + current_process().name + "\n")
            if done:
                state = self.env.reset()
                state = self.processState(state)
                done = False

            rewardSum = 0
            stepCount = 0

            while not done and stepCount < self.numSteps:
                # the calculation is on GPU is available
                value, logit = self.localNet(state)
                prob = F.softmat(logit, dim = 1)
                logProb = F.log_softmax(logit, dim = 1)
                entropy = -(logProb * prob).sum(1)

                action = prob.multinomial(1).data
                logProb = logProb.gather(1, action)

                nextState, reward, done, info = self.env.step(action.cpu().item())
                state = self.processState(nextState)

                bufferReward.append(reward)
                bufferLogProbs.append(logProb)
                rewardSum += reward

            # if done or stepCount == self.numSteps, we will update the net
            R = torch.zeros(1, 1, requires_grad=True)

            value, _ = self.localNet(state)
            R.data = value.data

            if self.gpuId >= 0:
                R = R.cuda()

            self.values.append(value)

            policyLoss = 0.0
            valueLoss = 0.0

            GAE = torch.zeros(1, 1, requires_grad=True)
            if self.gpuId >= 0:
                with torch.cuda.device(self.gpuId):
                    GAE = GAE.cuda()

            for i in reversed(range(len(self.rewards))):
                R = self.gamma * R + self.rewards[i]
                advantage = R - self.values[i]
                valueLoss += 0.5 * advantage.pow(2)

                # generalized advantage estimation
                # we use values.data to ensure delta_t is a torch tenor without grad
                delta_t = self.rewards[i] + self.gamma*self.values[i+1].data - self.values[i].data

                GAE = GAE * self.gamma * self.tau + delta_t

                policyLoss = policyLoss - self.logProb[i]*GAE - self.entropyPenality * self.entropies[i]

            self.localNet.zero_grad()
            (policyLoss + 0.5*valueLoss).backward()
            ensure_shared_grads(self.localNet, self.globalNet, gpu=self.gpuId > 0)
            self.globalOptimizer.step()

            self.clearup()


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



class A3CMaster(DQNAgent):
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