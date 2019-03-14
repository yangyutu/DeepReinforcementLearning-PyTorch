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




class DQNAsynERWorker(mp.Process):
    def __init__(self, config, localNet, env, globalNets, globalOptimizer, netLossFunc, nbAction, rank,
                 globalEpisodeCount, globalEpisodeReward, globalRunningAvgReward, resultQueue, logFolder,
                stateProcessor = None, lock = None):
        super(DQNAsynERWorker, self).__init__()
        self.config = config
        self.globalPolicyNet = globalNets[0]
        self.globalTargetNet = globalNets[1]
        self.rank = rank
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
        if 'updateGlobalFrequency' in self.config:
            self.updateGlobalFrequency = self.config['updateGlobalFrequency']

        self.gamma = 0.99
        if 'gamma' in self.config:
            self.gamma = self.config['gamma']

        self.trainStep = self.config['trainStep']
        self.globalEpisodeCount = globalEpisodeCount
        self.globalEpisodeReward = globalEpisodeReward
        self.globalRunningAvgReward = globalRunningAvgReward
        self.resultQueue = resultQueue
        self.dirName = logFolder

        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']

        self.randomSeed = 1 + self.rank
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed'] + self.rank
        torch.manual_seed(self.randomSeed)

        self.nStepForward = 1
        if 'nStepForward' in self.config:
            self.nStepForward = self.config['nStepForward']
        self.targetNetUpdateEpisode = 10
        if 'targetNetUpdateEpisode' in self.config:
            self.targetNetUpdateEpisode = self.config['targetNetUpdateEpisode']

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

        self.epsilon_by_episode = lambda step: self.epsilon_final + (
                self.epsilon_start - self.epsilon_final) * math.exp(-1. * step / self.epsilon_decay)

        self.netUpdateOption = 'targetNet'
        if 'netUpdateOption' in self.config:
            self.netUpdateOption = self.config['netUpdateOption']

        self.nStepBuffer = []

        self.memoryCapacity = self.config['memoryCapacity']
        self.trainBatchSize = self.config['trainBatchSize']
        self.memory = ReplayMemory(self.memoryCapacity)

        self.priorityMemoryOption = False

        self.episodeLength = 500
        if 'episodeLength' in self.config:
            self.episodeLength = self.config['episodeLength']

        self.synchLock = False
        if 'synchLock' in self.config:
            self.synchLock = self.config['synchLock']

        self.lock = lock

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
                    QValues = net(torchvector(state[np.newaxis, :]).to(self.device))
                action = torch.argmax(QValues).item()
        else:
            action = random.randint(0, self.numAction-1)
        return action


    def store_experience(self, state, action, nextState, reward):
        # caution: using multiple step forward return can increase variance
        if self.nStepForward > 1:

            # if this is final state, we want to do additional backup to increase useful learning experience
            if nextState is None:
                transitions = []
                transitions.append(Transition(state, action, nextState, reward))
                R = reward
                while len(self.nStepBuffer) > 0:
                    state, action, next_state, reward_old = self.nStepBuffer.pop(0)
                    R = reward_old + self.gamma * R
                    transNew = Transition(state, action, None, R)
                    transitions.append(transNew)
                for tran in transitions:
                    if self.priorityMemoryOption:
                        self.memory.store(tran)
                    else:
                        self.memory.push(tran)

            else:
                # otherwise we calculate normal n step return
                self.nStepBuffer.append((state, action, nextState, reward))

                if len(self.nStepBuffer) < self.nStepForward:
                    return

                R = sum([self.nStepBuffer[i][3]*(self.gamma**i) for i in range(self.nStepForward)])

                state, action, _, _ = self.nStepBuffer.pop(0)

                transition = Transition(state, action, nextState, R)

                if self.priorityMemoryOption:
                    self.memory.store(transition)
                else:
                    self.memory.push(transition)

        else:
            # if it is one step
            transition = Transition(state, action, nextState, reward)

            if self.priorityMemoryOption:
                self.memory.store(transition)
            else:
                self.memory.push(transition)


    def run(self):
        torch.set_num_threads(1)
        bufferState, bufferAction, bufferReward, bufferNextState = [], [], [], []
        for self.epIdx in range(self.trainStep):

            print("episode index:" + str(self.epIdx) + " from" + current_process().name + "\n")
            state = self.env.reset()
            done = False
            rewardSum = 0


            # clear the nstep buffer
            self.nStepBuffer.clear()

            for stepCount in range(self.episodeLength):

                epsilon = self.epsilon_by_episode(self.globalEpisodeCount.value)
                action = self.select_action(self.localNet, state, epsilon)
                nextState, reward, done, info = self.env.step(action)

                if stepCount == 0:
                    print("at step 0: from " + current_process().name + "\n")
                    print(info)

                if done:
                    nextState = None

                self.update_net_and_sync(state, action, nextState, reward)

                state = nextState
                rewardSum += reward * pow(self.gamma, stepCount)

                self.totalStep += 1
                if done:
#                    print("done in step count: {}".format(stepCount))
#                    print("reward sum = " + str(rewardSum))
                # done and print information
                #    pass
                    break

            self.recordInfo(rewardSum, stepCount)


        self.resultQueue.put(None)

    def recordInfo(self, reward, stepCount):
        with self.globalEpisodeReward.get_lock():
            self.globalEpisodeReward.value = reward
        with self.globalRunningAvgReward.get_lock():
            self.globalRunningAvgReward.value = (self.globalRunningAvgReward.value * self.globalEpisodeCount.value + reward) / (
                        self.globalEpisodeCount.value + 1)
        with self.globalEpisodeCount.get_lock():
            self.globalEpisodeCount.value += 1
            if self.config['logFlag'] and self.globalEpisodeCount.value % self.config['logFrequency'] == 0:
                self.save_checkpoint()
                # sync global target to global policy net
            if self.globalEpisodeCount.value % self.targetNetUpdateEpisode == 0:
                self.globalTargetNet.load_state_dict(self.globalPolicyNet.state_dict())

        # resultQueue.put(globalEpisodeReward.value)
        self.resultQueue.put(
            [self.globalEpisodeCount.value, stepCount, self.globalEpisodeReward.value, self.globalRunningAvgReward.value])
        print(self.name)
        print("Episode: ", self.globalEpisodeCount.value)
        print("stepCount: ", stepCount)
        print("Episode Reward: ", self.globalEpisodeReward.value)
        print("Episode Running Average Reward: ", self.globalRunningAvgReward.value)

    def save_checkpoint(self):
        prefix = self.dirName + 'Epoch' + str(self.globalEpisodeCount.value)
        torch.save({
            'epoch': self.globalEpisodeCount.value + 1,
            'model_state_dict': self.globalPolicyNet.state_dict(),
            'optimizer_state_dict': self.globalOptimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

    def update_net_and_sync(self, state, action, nextState, reward):

        self.store_experience(state, action, nextState, reward)

        if self.priorityMemoryOption:
            if len(self.memory) < self.config['memoryCapacity']:
                return
        else:
            if len(self.memory) < self.trainBatchSize:
                return

        if self.totalStep % self.updateGlobalFrequency == 0:
            transitions_raw = self.memory.sample(self.trainBatchSize)
            transitions = Transition(*zip(*transitions_raw))
            action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(
                -1)  # shape(batch, 1)
            reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(
                -1)  # shape(batch, 1)
            batchSize = reward.shape[0]


            # for some env, the output state requires further processing before feeding to neural network
            if self.stateProcessor is not None:
                state, _ = self.stateProcessor(transitions.state, self.device)
                nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
            else:
                state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
                nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)),
                                            device=self.device, dtype=torch.uint8)
                nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None],
                                                 device=self.device, dtype=torch.float32)
            if self.synchLock:

                self.lock.acquire()
                QValues = self.globalPolicyNet(state).gather(1, action)

                if self.netUpdateOption == 'targetNet':
                    # Here we detach because we do not want gradient flow from target values to net parameters
                    QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
                    QNext[nonFinalMask] = self.globalTargetNet(nonFinalNextState).max(1)[0].detach()
                    targetValues = reward + self.gamma * QNext.unsqueeze(-1)
                if self.netUpdateOption == 'policyNet':
                    raise NotImplementedError
                    targetValues = reward + self.gamma * torch.max(self.globalPolicyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
                if self.netUpdateOption == 'doubleQ':
                     # select optimal action from policy net
                     with torch.no_grad():
                        batchAction = self.globalPolicyNet(nonFinalNextState).max(dim=1)[1].unsqueeze(-1)
                        QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32).unsqueeze(-1)
                        QNext[nonFinalMask] = self.globalTargetNet(nonFinalNextState).gather(1, batchAction)
                        targetValues = reward + self.gamma * QNext

                loss = self.netLossFunc(QValues, targetValues)

                self.globalOptimizer.zero_grad()

                loss.backward()

                if self.netGradClip is not None:
                    torch.nn.utils.clip_grad_norm_(self.globalPolicyNet.parameters(), self.netGradClip)

                # global net update
                self.globalOptimizer.step()
                #
                # # update local net
                self.localNet.load_state_dict(self.globalPolicyNet.state_dict())

                self.lock.release()
            else:

                # update local net
                self.localNet.load_state_dict(self.globalPolicyNet.state_dict())

                QValues = self.localNet(state).gather(1, action)

                if self.netUpdateOption == 'targetNet':
                    # Here we detach because we do not want gradient flow from target values to net parameters
                    QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
                    QNext[nonFinalMask] = self.globalTargetNet(nonFinalNextState).max(1)[0].detach()
                    targetValues = reward + self.gamma * QNext.unsqueeze(-1)
                if self.netUpdateOption == 'policyNet':
                    raise NotImplementedError
                    targetValues = reward + self.gamma * torch.max(self.globalPolicyNet(nextState).detach(), dim=1)[
                        0].unsqueeze(-1)
                if self.netUpdateOption == 'doubleQ':
                    # select optimal action from policy net
                    with torch.no_grad():
                        batchAction = self.localNet(nonFinalNextState).max(dim=1)[1].unsqueeze(-1)
                        QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32).unsqueeze(-1)
                        QNext[nonFinalMask] = self.globalTargetNet(nonFinalNextState).gather(1, batchAction)
                        targetValues = reward + self.gamma * QNext

                loss = self.netLossFunc(QValues, targetValues)

                loss.backward()

                self.lock.acquire()

                self.globalOptimizer.zero_grad()

                for lp, gp in zip(self.localNet.parameters(), self.globalPolicyNet.parameters()):
                    gp._grad = lp._grad

                if self.netGradClip is not None:
                    torch.nn.utils.clip_grad_norm_(self.globalPolicyNet.parameters(), self.netGradClip)

                # global net update
                self.globalOptimizer.step()

                self.lock.release()
                #
                # # update local net
                self.localNet.load_state_dict(self.globalPolicyNet.state_dict())

    def test_multiProcess(self):
        print("Hello, World! from " + current_process().name + "\n")
        print(self.globalPolicyNet.state_dict())
        for gp in self.globalPolicyNet.parameters():
            gp.grad = torch.ones_like(gp)
            #gp.grad.fill_(1)

        self.globalOptimizer.step()

        print('globalNetID:')
        print(id(self.globalPolicyNet))
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

class DQNAsynERMaster(DQNAgent):
    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None):
        super(DQNAsynERMaster, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor)
        self.globalPolicyNet = policyNet
        self.globalTargetNet = targetNet
        self.globalPolicyNet.share_memory()
        self.globalTargetNet.share_memory()

        self.env = env[0]
        self.envs = env

        self.numWorkers = self.config['numWorkers']

        self.globalEpisodeCount = mp.Value('i', 0)
        self.globalEpisodeReward = mp.Value('d', 0)
        self.globalRunningAvgReward = mp.Value('d', 0)
        self.resultQueue = mp.Queue()

        self.synchLock = False
        if 'synchLock' in self.config:
            self.synchLock = self.config['synchLock']




        self.construct_workers()


    def construct_workers(self):
        self.workers = []
        lock = mp.Lock()
        for i in range(self.numWorkers):
            # local Net will not share memory
            localEnv = self.envs[i]
            localNet = deepcopy(self.globalPolicyNet)
            worker = DQNAsynERWorker(self.config, localNet, localEnv, [self.globalPolicyNet, self.globalTargetNet], self.optimizer, self.netLossFunc,
                                    self.numAction, i, self.globalEpisodeCount, self.globalEpisodeReward,
                                    self.globalRunningAvgReward, self.resultQueue, self.dirName, stateProcessor=self.stateProcessor, lock=lock)
            self.workers.append(worker)

    def test_multiProcess(self):

        for gp in self.globalPolicyNet.parameters():
            gp.data.fill_(0.0)

        print('initial global net state dict')
        print(self.globalPolicyNet.state_dict())

        processes = [mp.Process(target=w.test_multiProcess) for w in self.workers]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('Final global net state dict')
        print(self.globalPolicyNet.state_dict())

    def train(self):

        for w in self.workers:
            w.start()


        self.rewards = []  # record episode reward to plot
        while True:
            r = self.resultQueue.get()
            if r is not None:
                self.rewards.append(r)
            else:
                break

        for w in self.workers:
            w.join()

        print('all threads joined!')

        self.save_all()


    def save_all(self):
        prefix = self.dirName + 'Finalepoch' + str(self.globalEpisodeCount.value + 1)
        torch.save({
            'epoch': self.globalEpisodeCount.value + 1,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')



        self.saveRewards(prefix + '_reward.txt')