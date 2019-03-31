from Agents.DQN.DQN import DQNAgent
from copy import deepcopy

from Agents.Core.RecurrentReplayMemory import RecurrentReplayMemory
from Agents.Core.ReplayMemory import Transition
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import os
import math
#import unittest

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DRQNAgent(DQNAgent):
    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None):
        super(DRQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor)

        self.resetNetState()

    def init_memory(self):
        self.memory = RecurrentReplayMemory(self.memoryCapacity, self.sequenceLength)

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

        self.sequenceLength = self.config['sequenceLength']

        self.netMemory = self.config['netMemory']
 #       unittest.assertIn(a, b，[msg = '测试失败时打印的信息'])

    def resetNetState(self):
        self.stateSequence = [self.policyNet.get_zero_input() for j in range(self.sequenceLength)]

    def select_action(self, net, state, epsThreshold):

        # get a random number so that we can do epsilon exploration
        randNum = random.random()
        if randNum > epsThreshold:
            with torch.no_grad():
                self.stateSequence.pop(0)
                self.stateSequence.append(state)

                # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
                # here state[np.newaxis,:] is to add a batch dimension
                if self.stateProcessor is not None:
                    state, _ = self.stateProcessor([state], self.device)

                    QValues, _ = net(state)
                else:
                    X = torch.tensor([self.stateSequence], device=self.device, dtype=torch.float)
                    # we do not provide initial hidden state value
                    # the output is 1d array
                    QValues = net(X)
                    #QValues = QValues[:, -1, :] # select the last element in the output sequence
                action = torch.argmax(QValues).item()
        else:
            action = random.randint(0, self.numAction-1)
        return action

    def getPolicy(self, stateSequence):
        # for DRQN, we need a series of states to get a policy
        if len(stateSequence) != self.sequenceLength:
            raise AssertionError('len(stateSequence not equal self.sequenceLength')

        self.stateSequence = stateSequence[:-1]
        return self.select_action(self.policyNet, stateSequence[:-1], -0.01)

    def work_At_Episode_Begin(self):
        # clear the nstep buffer
        self.nStepBuffer.clear()
        # reset net states
        self.resetNetState()

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




    def update_net(self, state, action, nextState, reward):

        # first store memory

        self.store_experience(state, action, nextState, reward)

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

    def prepare_miniBatch(self, transitions_raw):

        # order the data
        # convert transition list to torch tensors
        # use trick from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343

        transitions = self.memory.sample(self.trainBatchSize)

        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(self.trainBatchSize,self.sequenceLength)
        reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(self.trainBatchSize,self.sequenceLength)

        state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(self.trainBatchSize,self.sequenceLength, -1)

        # get set of next states for end of each sequence
        batch_next_state = tuple(
            [batch_next_state[i] for i in range(len(batch_next_state)) if (i + 1) % (self.sequenceLength) == 0])

        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.uint8)
        nonFinalNextState = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                             dtype=torch.float).unsqueeze(dim=1)
        nonFinalNextState = torch.cat([state[nonFinalMask, 1:, :], nonFinalNextState], dim=1)


        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)

        return state, action, reward, nonFinalNextState, nonFinalMask

    def update_net_on_transitions(self, transitions_raw, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):

        # prepare samples
        state, action, reward, nonFinalNextState, nonFinalMask = self.prepare_miniBatch(transitions_raw)


        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            # action has shape (batchSize, sequenceLength)
            lastAction = action[:,-1].view(self.trainBatchSize, 1)
            QValues = self.policyNet(state).gather(1, lastAction)

            if updateOption == 'targetNet':
                 # Here we detach because we do not want gradient flow from target values to net parameters
                 QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
                 QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()
                 # reward has shape (batchSize, sequenceLength)
                 targetValues = reward[:, -1] + (self.gamma**self.nStepForward) * QNext
            if updateOption == 'policyNet':
                raise NotImplementedError
                targetValues = reward + self.gamma * torch.max(self.policyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
            if updateOption == 'doubleQ':
                 # select optimal action from policy net
                 with torch.no_grad():
                    batchAction = self.policyNet(nonFinalNextState).max(dim=1)[1].unsqueeze(-1)
                    QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32).unsqueeze(-1)
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
