from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.DQN.DQN import DQNAgent

from Agents.Core.ExtendedReplayMemory import ExtendedReplayMemory, ExtendedTransition
import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math
import pickle




class StackedDQNAgent(DQNAgent):

    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None, timeIndexMap=None):

        self.policyNets = policyNet
        self.targetNets = targetNet
        self.optimizers = optimizer

        super(StackedDQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction,
                                              stateProcessor, experienceProcessor)
        self.policyNet = None
        self.targetNet = None
        self.optimizer = None
        self.timeIndexMap = timeIndexMap
        self.init_memory()

    def initialization(self):
        # move model to correct device
        for i in range(len(self.policyNets)):
            self.policyNets[i] = self.policyNets[i].to(self.device)

        # in case targetNet is None
        for i in range(len(self.targetNets)):
            if self.targetNets[i] is not None:
                self.targetNets[i] = self.targetNets[i].to(self.device)

        self.dirName = 'Log/'
        if 'dataLogFolder' in self.config:
            self.dirName = self.config['dataLogFolder']
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)

        self.identifier = ''
        self.epIdx = 0
        self.learnStepCounter = 0  # for target net update
        self.globalStepCount = 0
        self.losses = []
        self.rewards = []
        self.nStepBuffer = []

    def init_memory(self):

        self.memories = [ExtendedReplayMemory(self.memoryCapacity) for _ in range(len(self.policyNets))]

    def store_experience(self, state, action, nextState, reward, done, info):

        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, done, info)
            # caution: using multiple step forward return can increase variance
            # if it is one step

        timeStep = state['stageID']
        done['id'] = self.globalStepCount
        transition = ExtendedTransition(state, action, nextState, reward, done)
        if done['stage'][0] and state['state'][1] < 1 and state['stageID'] == 0:
            print('issue!!!!!!!')
        self.memories[timeStep].push(transition)

    def train(self):

        runningAvgEpisodeReward = 0.0
        if len(self.rewards) > 0:
            runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            state = self.env.reset()

            rewardSum = 0

            for stepCount in range(self.episodeLength):

                timeStep = state['stageID']

                epsThreshold = self.epsilon_by_step(self.globalStepCount * (timeStep + 1) / len(self.policyNets))

                action = self.select_action(self.policyNets[timeStep], state, epsThreshold)

                nextState, reward, doneDict, info = self.env.step(action)

                done = doneDict['global']

                #if doneDict['stage'][0] and state['state'][1] < 1 and state['stageID'] == 0:
                #    print('issue!!!!!!!')

                if stepCount == 0:
                    print("at step 0:")
                    print(info)


                # learn the transition
                self.update_net(state, action, nextState, reward, doneDict, info)

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

            runningAvgEpisodeReward = (runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
            print("done in step count: {}".format(stepCount))
            print("reward sum = " + str(rewardSum))
            print("running average episode reward sum: {}".format(runningAvgEpisodeReward))
            print(info)

            self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
            if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                self.save_checkpoint()

            self.epIdx += 1
        self.save_all()

    def update_net(self, state, action, nextState, reward, done, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, done, info)




        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience

            for i in range(len(self.memories) - 1, -1, -1):
                if len(self.memories[i]) < self.trainBatchSize:
                    continue

                transitions_raw = self.memories[i].sample(self.trainBatchSize)
                self.policyNet = self.policyNets[i]
                self.targetNet = self.targetNets[i]
                self.optimizer = self.optimizers[i]

                # when finishing current stage, we need the value from next stage to guide the learning process
                # when we are currently at final stage and finishes, we do not from value target from Q function
                if self.netUpdateOption == 'targetNet' or self.netUpdateOption == 'doubleQ':
                    if i < (len(self.memories) - 1):
                        self.nextStageTargetNet = self.targetNets[i + 1]
                    else:
                        self.nextStageTargetNet = None
                if self.netUpdateOption == 'policyNet':
                    raise NotImplementedError

                loss = self.update_net_on_transitions(transitions_raw, self.netLossFunc, 1,
                                                      updateOption=self.netUpdateOption, netGradClip=self.netGradClip,
                                                      info=info)

                if self.globalStepCount % self.lossRecordStep == 0:
                    self.losses.append([self.globalStepCount, self.epIdx, loss])

                if self.learnStepCounter % self.targetNetUpdateStep == 0:
                    self.targetNets[i].load_state_dict(self.policyNets[i].state_dict())

            self.learnStepCounter += 1

    def prepare_miniBatch(self, transitions_raw):
        transitions = ExtendedTransition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1) # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, device = self.device)
            nonFinalNextState, nonFinalMask, finalNextState, finalMask = self.stateProcessor(transitions.next_state, device = self.device, done = transitions.done)
        else:
            raise NotImplementedError
            # state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            # non final if there is at least one stage not finish
            # nonFinalMask = torch.tensor(tuple(map(lambda s: not np.all(s['stage']), transitions.done)), device=self.device, dtype=torch.uint8)
            # nonFinalNextState = torch.tensor(transitions.next_state[nonFinalMask], device=self.device, dtype=torch.float32)
            # finalMask = [map(lambda s: s['global']), transitions.done]
            # finalNextState = torch.tensor(transitions.next_state[finalMask], device=self.device, dtype=torch.float32)

        return state, action, reward, nonFinalNextState, nonFinalMask, finalNextState, finalMask


    def update_net_on_transitions(self, transitions_raw, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):

        # prepare samples
        # final next state is the state with stage done
        # nonFinalNextState is the state without both stage done and global done (global done requires all stage done)
        #
        state, action, reward, nonFinalNextState, nonFinalMask, finalNextState, finalMask = self.prepare_miniBatch(transitions_raw)


        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            QValues = self.policyNet(state).gather(1, action)

            if updateOption == 'targetNet':
                 # Here we detach because we do not want gradient flow from target values to net parameters

                 QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
                 if self.nextStageTargetNet is not None and len(finalNextState):
                     # Q values for current stage done but not next stage done (or not global done)
                     # if nextStageTargetNet is None, means it is in the final stage, and we will not bootstrap for finite-horizon MDP
                    QNext[finalMask] = self.nextStageTargetNet(finalNextState).max(1)[0].detach()

                # Q values for states not stage done
                 if len(nonFinalNextState):
                     QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()
                 # reward has shape (batchSize, sequenceLength)
                 targetValues = reward + (self.gamma) * QNext.unsqueeze(-1)
            if updateOption == 'policyNet':
                raise NotImplementedError
                targetValues = reward + self.gamma * torch.max(self.policyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
            if updateOption == 'doubleQ':

                 # select optimal action from policy net
                 with torch.no_grad():
                    batchAction = self.policyNet(nonFinalNextState).max(dim=1)[1]
                    QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)

                    if self.nextStageTargetNet is not None and len(finalNextState):
                        # Q values for current stage done but not next stage done (or not global done)
                        # if nextStageTargetNet is None, means it is in the final stage, and we will not bootstrap for finite-horizon MDP
                        QNext[finalMask] = self.nextStageTargetNet(finalNextState).max(1)[0].detach()

                    # Q values for states not stage done
                    if len(nonFinalNextState):
                        QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()

                    targetValues = reward + (self.gamma) * QNext.unsqueeze(-1)

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


    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')

    def save_checkpoint(self):
        prefix = self.dirName + self.identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': [net.state_dict() for net in self.policyNets],
            'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers]
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        # self.loadLosses(prefix + '_loss.txt')
        # self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memories = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        for i in range(len(self.policyNets)):
            self.policyNet.load_state_dict(checkpoint['model_state_dict'][i])
            self.targetNet.load_state_dict(checkpoint['model_state_dict'][i])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])