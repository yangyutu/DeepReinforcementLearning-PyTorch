from Agents.DQN.DQN import DQNAgent
from Agents.Core.ReplayMemory import Transition
from Agents.Core.GroupReplayMemory import GroupReplayMemory
from Agents.MADQN.Mixers import VDNMixer
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import os
import math
import pickle
from copy import deepcopy


class MADQNAgent:

    def __init__(self, config, policyNets, targetNets, env, optimizers, netLossFunc, nbActions, mixNet = None, mixNetOptimizer = None, stateProcessors = None):
        self.config = config
        self.policyNets = policyNets
        self.targetNets = targetNets
        self.env = env
        self.optimizers = optimizers
        self.netLossFunc = netLossFunc
        self.numActions = nbActions
        self.stateProcessors = stateProcessors

        if mixNet is None:
            self.mixNet = VDNMixer()
            self.mixNetOptimizer = None
        else:
            self.mixNet = mixNet
            self.mixNetOptimizer = mixNetOptimizer


        self.read_config()
        self.init_memory()
        self.initialization()
        self.init_nets()

    def init_nets(self):
        # move model to correct device
        for n in range(self.numAgents):
            self.policyNets[n] = self.policyNets[n].to(self.device)
            self.targetNets[n] = self.targetNets[n].to(self.device)

        self.mixNet = self.mixNet.to(self.device)

    def initialization(self):
        # move model to correct device
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


    def init_memory(self):

        self.memory=GroupReplayMemory(self.memoryCapacity, self.numAgents)


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
        self.memoryCapacity = self.config['memoryCapacity']
        random.seed(self.randomSeed)

        self.numAgents = self.config['numAgents']

    def select_action(self, nets, states, epsThreshold):
        # we need to select multiple actions
        actions = []

        epsThresholds = self.epsilon_by_step(self.globalStepCount)

        for n in range(self.numAgents):
            randNum = np.random.rand()

            if randNum > epsThresholds:
                with torch.no_grad():
                    # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
                    # here state[np.newaxis,:] is to add a batch dimension
                    if self.stateProcessors is not None:
                        state, _ = self.stateProcessors[n]([states[n]], self.device)
                        QValues = nets[n](state)
                    else:
                        stateTorch = torch.from_numpy(np.array(states[n][np.newaxis, :], dtype=np.float32))
                        QValues = nets[n](stateTorch.to(self.device))
                    action = torch.argmax(QValues).item()
            else:
                action = random.randint(0, self.numActions[n] - 1)
            actions.append(action)

        return actions


    def train(self):

        runningAvgEpisodeReward = 0.0

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            states = self.env.reset()
            rewardSum = 0


            for stepCount in range(self.episodeLength):

                actions = self.select_action(self.policyNets, states, self.epsThreshold)

                nextStates, rewards, done, info = self.env.step(actions)

                if stepCount == 0:
                    print("at step 0:")
                    print(info)

                if self.verbose:
                    print('step: ', trainStepCount)
                    print(info)

                if done:
                    nextStates = [None for _ in nextStates]

                # learn the transition
                self.update_net(states, actions, nextStates, rewards, info)

                states = nextStates

                rewardSum += np.sum(rewards*math.pow(self.gamma, stepCount))
                self.globalStepCount += 1

                if done:
                    break

            self.epIdx += 1
            runningAvgEpisodeReward = (runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
            print('episode', self.epIdx)
            print("done in step count:")
            print(stepCount)
            print('reward sum')
            print(rewardSum)
            print(info)
            # print('reward sum: ', rewardSum)
            print("running average episode reward sum: {}".format(runningAvgEpisodeReward))

            self.rewards.append([self.epIdx, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                self.save_checkpoint()

        self.save_all()

    def store_experience(self, states, actions, nextStates, reward, info):

        transitions = [Transition(states[n], actions[n], nextStates[n], reward) for n in range(self.numAgents)]
        self.memory.push(transitions)

    def update_net(self, states, actions, nextStates, reward, info):

        # first store memory

        self.store_experience(states, actions, nextStates, reward, info)

        if len(self.memory) < self.trainBatchSize:
            return

        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience
            info = {}
            transitions_raw_list = self.memory.sample(self.trainBatchSize)

            loss = self.update_net_on_transitions(transitions_raw_list, self.netLossFunc, 1, updateOption=self.netUpdateOption, netGradClip=self.netGradClip, info=info)

            if self.globalStepCount % self.lossRecordStep == 0:
                self.losses.append([self.globalStepCount, self.epIdx, loss])

            if self.learnStepCounter % self.targetNetUpdateStep == 0:
                for n in range(self.numAgents):
                    self.targetNets[n].load_state_dict(self.policyNets[n].state_dict())

            self.learnStepCounter += 1

    def prepare_minibatch(self, transitions_raw, n):
        # first store memory

        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1)  # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(
            -1)  # shape(batch, 1)

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessors is not None:
            state, _ = self.stateProcessors[n](transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessors[n](transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device,
                                        dtype=torch.uint8)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device,
                                             dtype=torch.float32)

        return state, nonFinalMask, nonFinalNextState, action, reward


    def update_net_on_transitions(self, transitions_raw_list, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):

        # order the data
        # convert transition list to torch tensors
        # use trick from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        QValuesList = []
        nextStateValuesList = []
        stateList = []
        for n in range(self.numAgents):
            transitions_raw = transitions_raw_list[n]
            state, nonFinalMask, nonFinalNextState, action, reward = self.prepare_minibatch(transitions_raw, n)
            stateList.append(state)

                # calculate Qvalues based on selected action batch
            QValues = self.policyNets[n](state).gather(1, action)

            if updateOption == 'targetNet':

                 # Here we detach because we do not want gradient flow from target values to net parameters
                 QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
                 QNext[nonFinalMask] = self.targetNets[n](nonFinalNextState).max(1)[0].detach()
                 nextStateValues = (self.gamma) * QNext.unsqueeze(-1)
            if updateOption == 'policyNet':
                raise NotImplementedError
                targetValues = reward + self.gamma * torch.max(self.policyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
            if updateOption == 'doubleQ':
                 # select optimal action from policy net
                 with torch.no_grad():
                    batchAction = self.policyNets[n](nonFinalNextState).max(dim=1)[1].unsqueeze(-1)
                    QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
                    QNext[nonFinalMask] = self.targetNets[n](nonFinalNextState).gather(1, batchAction).squeeze()
                    nextStateValues = (self.gamma) * QNext.unsqueeze(-1)

            QValuesList.append(QValues)
            nextStateValuesList.append(nextStateValues)

        QValuesVec = torch.cat(QValuesList, dim=1)

        nextStateValuesVec = torch.cat(nextStateValuesList, dim=1)

        # now mixing and add reward
        QValues_sum = self.mixNet(QValuesVec)
        #QValues_sum = QValuesVec[:, 0].unsqueeze(1)
        #QValues_sum = QValuesList[0]
        targetValues_sum = reward + self.mixNet(nextStateValuesVec).detach()
        #targetValues_sum = reward + nextStateValuesVec[:, 0].unsqueeze(1)
        #targetValues_sum = reward + nextStateValuesList[0]


        # Compute loss
        loss = loss_fun(QValues_sum, targetValues_sum).mean()

        # Optimize the model

        # zero gradient
        for n in range(self.numAgents):
            self.optimizers[n].zero_grad()
        if self.mixNetOptimizer is not None:
            self.mixNetOptimizer.zero_grad()

        loss.backward()

        for n in range(self.numAgents):
            if netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.policyNets[n].parameters(), netGradClip)
            self.optimizers[n].step()

        if self.mixNetOptimizer is not None:
            if netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.mixNet.parameters(), netGradClip)
            self.mixNetOptimizer.step()

        return loss.item()

    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')

    def save_checkpoint(self):
        prefix = self.dirName + self.identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        #self.loadLosses(prefix + '_loss.txt')
        #self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memory = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        for i in range(self.numAgents):
            self.policyNet.load_state_dict(checkpoint['model_state_dict'][i])
            self.targetNet.load_state_dict(checkpoint['model_state_dict'][i])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])

    def saveLosses(self, fileName):
        np.savetxt(fileName, np.array(self.losses), fmt='%.5f', delimiter='\t')

    def saveRewards(self, fileName):
        np.savetxt(fileName, np.array(self.rewards), fmt='%.5f', delimiter='\t')

    def loadLosses(self, fileName):
        self.losses = np.genfromtxt(fileName).tolist()
    def loadRewards(self, fileName):
        self.rewards = np.genfromtxt(fileName).tolist()