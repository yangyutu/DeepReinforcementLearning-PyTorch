import torch
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.DDPG.DDPG import DDPGAgent

from Agents.Core.ExtendedReplayMemory import ExtendedReplayMemory, ExtendedTransition

import pickle


class DDPGMultiStageUnit(DDPGAgent):
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None):

        super(DDPGMultiStageUnit, self).__init__(config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction,
                                          stateProcessor, experienceProcessor)

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
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.float32)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32)

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

    def update_net_on_memory_given_target(self, targetAgent=None):


        if len(self.memory) < self.trainBatchSize:
            return

        transitions_raw = self.memory.sample(self.trainBatchSize)


        state, nonFinalMask, nonFinalNextState, finalMask, finalNextState, action, reward = self.prepare_minibatch(transitions_raw)

        # now do net update

        # Critic loss
        QValues = self.criticNet.forward(state, action).squeeze()
        QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)


        numNonFinalNextState = sum(nonFinalMask)
        numFinalNextState = sum(finalMask)

        if numNonFinalNextState:

            next_actions = self.actorNet_target.forward(nonFinalNextState)

            # if we do not have stage done
            # we use our own target net to bootstrap
            QNext[nonFinalMask] = self.criticNet_target.forward(nonFinalNextState, next_actions.detach()).squeeze()

        if numFinalNextState:
            if targetAgent is not None:
                QNext[finalMask] = targetAgent.evaluate_state_value(finalNextState)

        targetValues = reward + self.gamma * QNext
        critic_loss = self.netLossFunc(QValues, targetValues)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.criticNet.parameters(), self.netGradClip)

        self.critic_optimizer.step()



        # update networks
        if self.learnStepCounter % self.policyUpdateFreq == 0:

            # Actor loss
            # we try to maximize criticNet output(which is state value)
            policy_loss = -self.criticNet.forward(state, self.actorNet.forward(state)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            if self.netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), self.netGradClip)

            self.actor_optimizer.step()

            if self.globalStepCount % self.lossRecordStep == 0:
                self.losses.append([self.globalStepCount, self.epIdx, critic_loss.item(), policy_loss.item()])


        # update networks
        if self.learnStepCounter % self.policyUpdateFreq == 0:
            # update target networks
            for target_param, param in zip(self.actorNet_target.parameters(), self.actorNet.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.criticNet_target.parameters(), self.criticNet.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.learnStepCounter += 1

    def evaluate_state_value(self, state):

        optimal_actions = self.actorNet_target.forward(state)
        QNext = self.criticNet_target.forward(state, optimal_actions.detach()).squeeze()
        return QNext
