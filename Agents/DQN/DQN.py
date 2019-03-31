
from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.ReplayMemoryReward import ReplayMemoryReward
from Agents.Core.PrioritizedReplayMemory import PrioritizedReplayMemory
import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math
import pickle

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DQNAgent(BaseDQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None):
        super(DQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor)

        self.init_memory()


    def init_memory(self):

        if self.priorityMemoryOption:
            self.memory = PrioritizedReplayMemory(self.memoryCapacity, self.config)
        else:
            if self.memoryOption == 'natural':
                self.memory = ReplayMemory(self.memoryCapacity)
            elif self.memoryOption == 'reward':
                self.memory = ReplayMemoryReward(self.memoryCapacity, self.config['rewardMemoryBackupStep'],
                                                 self.gamma, self.config['rewardMemoryTerminalRatio'] )

    def read_config(self):
        super(DQNAgent, self).read_config()
        # read additional parameters
        self.memoryCapacity = self.config['memoryCapacity']

        self.memoryOption = 'natural'
        self.priorityMemoryOption = False
        if 'memoryOption' in self.config:
            self.memoryOption = self.config['memoryOption']
            if self.memoryOption == 'priority':
                self.priorityMemoryOption = True
                # reward memory requires nstep forward to be 1
            if self.memoryOption == 'reward':
                self.nStepForward = 1

    def work_At_Episode_Begin(self):
        # clear the nstep buffer
        self.nStepBuffer.clear()

    def train(self):

        runningAvgEpisodeReward = 0.0
        if len(self.rewards) > 0:
            runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            state = self.env.reset()
            done = False
            rewardSum = 0

            self.work_At_Episode_Begin()

            for stepCount in range(self.episodeLength):
                self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

                action = self.select_action(self.policyNet, state, self.epsThreshold)

                nextState, reward, done, info = self.env.step(action)

                if stepCount == 0:
                    print("at step 0:")
                    print(info)

                if done:
                    nextState = None

                # learn the transition
                self.update_net(state, action, nextState, reward, info)

                state = nextState
                rewardSum += reward * pow(self.gamma, stepCount)
                self.globalStepCount += 1

                if self.verbose:
                    print('action: ' + str(action))
                    print('state:')
                    print(nextState)
                    print('reward:')
                    print(reward)
                    print('info')
                    print(info)

                if done:
                    break

            runningAvgEpisodeReward = (runningAvgEpisodeReward*self.epIdx + rewardSum)/(self.epIdx + 1)
            print("done in step count: {}".format(stepCount))
            print("reward sum = " + str(rewardSum))
            print("running average episode reward sum: {}".format(runningAvgEpisodeReward))
            print(info)

            self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                self.save_checkpoint()



            self.epIdx += 1
        self.save_all()

    def store_experience(self, state, action, nextState, reward, info):

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




    def update_net(self, state, action, nextState, reward, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, info)

        if self.priorityMemoryOption:
            if len(self.memory) < self.config['memoryCapacity']:
                return
        else:
            if len(self.memory) < self.trainBatchSize:
                return


        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience
            info = {}
            if self.priorityMemoryOption:
                transitions_raw, b_idx, ISWeights = self.memory.sample(self.trainBatchSize)
                info['batchIdx'] = b_idx
                info['ISWeights'] = torch.from_numpy(ISWeights.astype(np.float32)).to(self.device)
            else:
                transitions_raw= self.memory.sample(self.trainBatchSize)

            loss = self.update_net_on_transitions(transitions_raw, self.netLossFunc, 1, updateOption=self.netUpdateOption, netGradClip=self.netGradClip, info=info)

            if self.globalStepCount % self.lossRecordStep == 0:
                self.losses.append([self.globalStepCount, self.epIdx, loss])

            if self.learnStepCounter % self.targetNetUpdateStep == 0:
                self.targetNet.load_state_dict(self.policyNet.state_dict())

            self.learnStepCounter += 1

    def update_net_on_transitions(self, transitions_raw, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):

        # order the data
        # convert transition list to torch tensors
        # use trick from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343

        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1) # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)
        batchSize = reward.shape[0]


        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device, dtype=torch.uint8)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device, dtype=torch.float32)

        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            QValues = self.policyNet(state).gather(1, action)

            if updateOption == 'targetNet':
                 # Here we detach because we do not want gradient flow from target values to net parameters
                 QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
                 QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()
                 targetValues = reward + (self.gamma**self.nStepForward) * QNext.unsqueeze(-1)
            if updateOption == 'policyNet':
                raise NotImplementedError
                targetValues = reward + self.gamma * torch.max(self.policyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
            if updateOption == 'doubleQ':
                 # select optimal action from policy net
                 with torch.no_grad():
                    batchAction = self.policyNet(nonFinalNextState).max(dim=1)[1].unsqueeze(-1)
                    QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32).unsqueeze(-1)
                    QNext[nonFinalMask] = self.targetNet(nonFinalNextState).gather(1, batchAction)
                    targetValues = reward + (self.gamma**self.nStepForward) * QNext

            # Compute loss
            loss_single = loss_fun(QValues, targetValues)
            if self.priorityMemoryOption:
                loss = torch.mean(info['ISWeights'] * loss_single)
                # update priority
                abs_error = np.abs((QValues - targetValues).data.numpy())
                self.memory.batch_update(info['batchIdx'], abs_error)
            else:
                loss = torch.mean(loss_single)

            # Optimize the model
            # print(loss)
            # zero gradient
            self.optimizer.zero_grad()

            loss.backward()
            if netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(), netGradClip)
            self.optimizer.step()

            return loss.item()

    def perform_random_exploration(self, episodes, memory=None):
        for epIdx in range(episodes):
            print("episode index:" + str(epIdx))
            state = self.env.reset()
            done = False
            rewardSum = 0
            stepCount = 0
            while not done:
                action = random.randint(0, self.numAction-1)
                nextState, reward, done, _ = self.env.step(action)
                stepCount += 1
                memory.push(state, action, nextState, reward)
                state = nextState
                rewardSum += reward
                if done:
                    print("done in step count: {}".format(stepCount))
                    break
            print("reward sum = " + str(rewardSum))

    def perform_on_policy(self, episodes, policy, memory=None):

        for epIdx in range(episodes):
            print("episode index:" + str(epIdx))
            state = self.env.reset()
            done = False
            rewardSum = 0
            stepCount = 0
            while not done:
                action = self.select_action(self.policyNet, state, -0.01)
                nextState, reward, done, info = self.env.step(action)

                if memory is not None:
                    memory.push(epIdx, stepCount, state, action, nextState, reward, info)
                state = nextState
                rewardSum += reward
                if done:
                    print("done in step count: {}".format(stepCount))
                    break

                if stepCount > self.episodeLength:
                    break

                stepCount += 1
            print("reward sum = " + str(rewardSum))

    def testPolicyNet(self, episodes, memory = None):
        return self.perform_on_policy(episodes, self.getPolicy, memory)

    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        self.loadLosses(prefix + '_loss.txt')
        self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memory = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        self.policyNet.load_state_dict(checkpoint['model_state_dict'])
        self.targetNet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])