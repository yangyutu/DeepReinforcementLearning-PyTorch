from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.DQN.DQN import DQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.ExtendedReplayMemory import ExtendedReplayMemory, ExtendedTransition
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


class DQNMultiStageUnit(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, experienceProcessor=None):
        super(DQNMultiStageUnit, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor, experienceProcessor)


    def train(self):
        raise NotImplementedError

    def init_memory(self):
        self.memory = ExtendedReplayMemory(self.memoryCapacity)

    def store_experience(self, state, action, nextState, reward, done, info):
        # if it is one step
        transition = ExtendedTransition(state, action, nextState, reward, done)
        self.memory.push(transition)

    def prepare_minibatch(self, transitions_raw):
        # first store memory

        transitions = ExtendedTransition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1)  # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(
            -1)  # shape(batch, 1)

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask, finalNextState, finalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nextState = torch.tensor(transitions.next_state, device=self.device, dtype=torch.float32)
            # final mask is one that have stage done
            finalMask = torch.tensor(transitions.done, device=self.device, dtype=torch.uint8)
            nonFinalMask = 1 - finalMask
            finalNextState = [nextState[i] for i in range(self.trainBatchSize) if finalMask[i]]
            nonFinalNextState = [nextState[i] for i in range(self.trainBatchSize) if nonFinalMask[i]]

        if len(nonFinalNextState):
            nonFinalNextState = torch.stack(nonFinalNextState)

        if len(finalNextState):
            finalNextState = torch.stack(finalNextState)

        return state, nonFinalMask, nonFinalNextState, finalMask, finalNextState, action, reward



    def update_net_on_memory_given_target(self, targetAgent = None):

        if len(self.memory) < self.trainBatchSize:
            return

        transitions_raw = self.memory.sample(self.trainBatchSize)

        state, nonFinalMask, nonFinalNextState, finalMask, finalNextState, action, reward = self.prepare_minibatch(transitions_raw)

        # calculate Qvalues based on selected action batch
        QValues = self.policyNet(state).gather(1, action)

        # Here we detach because we do not want gradient flow from target values to net parameters
        QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)

        if len(nonFinalNextState):
        # if we do not have stage done
        # we use our own target net to bootstrap
            QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()

        if len(finalNextState):
        # if we have stage done,
        # 1) if it is not the last stage, we use external target agent to bootstrap
        # 2) if it is the last stage, we do not bootstrap
            if targetAgent is not None:
                QNext[finalMask] = targetAgent.evaluate_state_value(finalNextState)

        targetValues = reward + (self.gamma) * QNext.unsqueeze(-1)

        # Compute loss
        loss_single = self.netLossFunc(QValues, targetValues)
        loss = torch.mean(loss_single)

        # Optimize the model
        # zero gradient
        self.optimizer.zero_grad()

        loss.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(), self.netGradClip)
        self.optimizer.step()

        if self.globalStepCount % self.lossRecordStep == 0:
            self.losses.append([self.globalStepCount, self.epIdx, loss])

        if self.learnStepCounter % self.targetNetUpdateStep == 0:
            self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.learnStepCounter += 1

    def evaluate_state_value(self, state):
        return self.targetNet(state).max(1)[0].detach()