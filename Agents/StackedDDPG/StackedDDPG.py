import torch
import os
import numpy as np
from Agents.Core.ExtendedReplayMemory import ExtendedReplayMemory, ExtendedTransition
from Agents.DDPG.DDPG import DDPGAgent
import pickle


class StackedDDPGAgent(DDPGAgent):
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None, timeIndexMap=None):

        super(StackedDDPGAgent, self).__init__(config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction,
                                               stateProcessor, experienceProcessor)
        self.timeIndexMap = timeIndexMap

    def initalizeNets(self, actorNets, criticNets, optimizers):
        self.actorNets = actorNets['actor']
        self.actorNet_targets = actorNets['target'] if 'target' in actorNets else None
        self.criticNets = criticNets['critic']
        self.criticNet_targets = criticNets['target'] if 'target' in criticNets else None
        self.actor_optimizers = optimizers['actor']
        self.critic_optimizers = optimizers['critic']

        self.actorNet = None
        self.criticNet = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actorNet_target = None
        self.criticNet_target = None

        self.net_to_device()

    def init_memory(self):
        self.memories = [ReplayMemory(self.memoryCapacity) for _ in range(self.episodeLength)]

    def net_to_device(self):
        # move model to correct device

        for i in range(len(self.actorNets)):
            self.actorNets[i] = self.actorNets[i].to(self.device)
            self.criticNets[i] = self.criticNets[i].to(self.device)

        # in case targetNet is None
        if self.actorNet_targets is not None:
            for i in range(len(self.actorNet_targets)):
                self.actorNet_targets[i] = self.actorNet_targets[i].to(self.device)
        # in case targetNet is None
        if self.criticNet_targets is not None:
            for i in range(len(self.criticNet_targets)):
                self.criticNet_targets[i] = self.criticNet_targets[i].to(self.device)

    def process_experienceAugmentation(self, state, action, nextState, reward, info):
        pass

    def process_hindSightExperience(self, state, action, nextState, reward, info):

        timeStep = state['timeStep']
        if nextState is not None and self.globalStepCount % self.hindSightERFreq == 0:
            stateNew, actionNew, nextStateNew, rewardNew = self.env.getHindSightExperience(state, action, nextState, info)
            if stateNew is not None:
                transition = Transition(stateNew, actionNew, nextStateNew, rewardNew)
                self.memories[timeStep].push(transition)
                if self.experienceAugmentation:
                    self.process_experienceAugmentation(state, action, nextState, reward, info)


    def store_experience(self, state, action, nextState, reward, info):
        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, info)

        timeStep = state['timeStep']
        transition = Transition(state, action, nextState, reward)
        self.memories[timeStep].push(transition)

        if self.experienceAugmentation:
            self.process_experienceAugmentation(state, action, nextState, reward, info)

        if self.hindSightER:
            self.process_hindSightExperience(state, action, nextState, reward, info)

    def update_net(self, state, action, nextState, reward, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, info)

        for n in range(len(self.actorNets) - 1, -1, -1):
            #        if self.globalStepCount % self.netUpdateFrequency == 0:
            # update target networks
            if len(self.memories[n]) < self.trainBatchSize:
                return

            self.actorNet = self.actorNets[n]
            self.criticNet = self.criticNets[n]

            # actorNet_target is used to select best action in the next state
            self.actorNet_target = self.actorNet_targets[n + 1] if n < len(self.actorNets) - 1 \
                else self.actorNet_targets[n]

            self.criticNet_target = self.criticNet_targets[n + 1] if n < len(self.actorNets) - 1 \
                else self.criticNet_targets[n]

            self.critic_optimizer = self.critic_optimizers[n]
            self.actor_optimizer = self.actor_optimizers[n]

            transitions_raw = self.memories[n].sample(self.trainBatchSize)

            self.update_net_on_transitions(transitions_raw)

            # do soft update
            if self.learnStepCounter % self.policyUpdateFreq == 0:
                for target_param, param in zip(self.actorNet_targets[n].parameters(), self.actorNets[n].parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.criticNet_targets[n].parameters(), self.criticNets[n].parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            self.learnStepCounter += 1

    def work_before_step(self, state):
        # use the correct actor net
        timeStep = state['timeStep']
        timeIdx = self.timeIndexMap[timeStep]
        self.actorNet = self.actorNets[timeIdx]

    def saveLosses(self, fileName):
        np.savetxt(fileName, np.array(self.losses), fmt='%.5f', delimiter='\t')

    def saveRewards(self, fileName):
        np.savetxt(fileName, np.array(self.rewards), fmt='%.5f', delimiter='\t')

    def loadLosses(self, fileName):
        self.losses = np.genfromtxt(fileName).tolist()

    def loadRewards(self, fileName):
        self.rewards = np.genfromtxt(fileName).tolist()

    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'actorNet_state_dict': [net.state_dict() for net in self.actorNets],
            'criticNet_state_dict': [net.state_dict() for net in self.criticNets],
            'actor_optimizer_state_dict': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer_state_dict': [opt.state_dict() for opt in self.critic_optimizers]
        }, prefix + '_checkpoint.pt')

    def save_checkpoint(self):
        prefix = self.dirName + self.identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memories, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'actorNet_state_dict': [net.state_dict() for net in self.actorNets],
            'criticNet_state_dict': [net.state_dict() for net in self.criticNets],
            'actor_optimizer_state_dict': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer_state_dict': [opt.state_dict() for opt in self.critic_optimizers]
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        self.loadLosses(prefix + '_loss.txt')
        self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memories = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        for i in range(len(self.actorNets)):
            self.actorNets[i].load_state_dict(checkpoint['actorNet_state_dict'])
            self.actorNet_targets[i].load_state_dict(checkpoint['actorNet_state_dict'])
            self.criticNets[i].load_state_dict(checkpoint['criticNet_state_dict'])
            self.criticNet_targets[i].load_state_dict(checkpoint['criticNet_state_dict'])

        for i in range(len(self.actor_optimizers)):
            self.actor_optimizer[i].load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer[i].load_state_dict(checkpoint['critic_optimizer_state_dict'])