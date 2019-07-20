import torch
import os
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.SAC.SAC import SACAgent
import pickle


class StackedSACAgent(SACAgent):
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None, timeIndexMap=None):

        super(StackedSACAgent, self).__init__(config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction,
                                               stateProcessor, experienceProcessor)
        self.timeIndexMap = timeIndexMap

    def initalizeNets(self, actorNets, criticNets, optimizers):
        # totally five nets
        self.actorNets = actorNets['actor']

        self.softQNetsOne = criticNets['softQOne']
        self.softQNetsTwo = criticNets['softQTwo']

        self.valueNets = criticNets['value']
        self.valueTargetNets = criticNets['valueTarget']

        self.actor_optimizers = optimizers['actor']
        self.softQOne_optimizers = optimizers['softQOne']
        self.softQTwo_optimizers = optimizers['softQTwo']
        self.value_optimizers = optimizers['value']

        # totally five nets
        self.actorNet = None
        self.softQNetOne = None
        self.softQNetTwo = None
        self.valueNet = None
        self.valueTargetNet = None

        self.actor_optimizer = None
        self.softQOne_optimizer = None
        self.softQTwo_optimizer = None
        self.value_optimizer = None

        self.net_to_device()

    def init_memory(self):
        self.memories = [ReplayMemory(self.memoryCapacity) for _ in range(self.episodeLength)]

    def net_to_device(self):
        # move model to correct device

        for i in range(len(self.actorNets)):
            self.actorNets[i] = self.actorNets[i].to(self.device)
            self.softQNetsOne[i] = self.softQNetsOne[i].to(self.device)
            self.softQNetsTwo[i] = self.softQNetsTwo[i].to(self.device)
            self.valueNets[i] = self.valueNets[i].to(self.device)
            self.valueTargetNets[i] = self.valueTargetNets[i].to(self.device)

    def process_experienceAugmentation(self, state, action, nextState, reward, info):
        pass

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
            self.softQNetOne = self.softQNetsOne[n]
            self.softQNetTwo = self.softQNetsTwo[n]
            self.valueNet = self.valueNets[n]


            self.valueTargetNet = self.valueTargetNets[n + 1] if n < len(self.actorNets) - 1 \
                else self.valueTargetNets[n]



            self.softQOne_optimizer = self.softQOne_optimizers[n]
            self.softQTwo_optimizer = self.softQTwo_optimizers[n]
            self.actor_optimizer = self.actor_optimizers[n]
            self.value_optimizer = self.value_optimizers[n]

            transitions_raw = self.memories[n].sample(self.trainBatchSize)

            self.update_net_on_transitions(transitions_raw)

            # do soft update
            if self.learnStepCounter % self.policyUpdateFreq == 0:
                for target_param, param in zip(self.valueTargetNets[n].parameters(), self.valueNets[n].parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def train(self):

        runningAvgEpisodeReward = 0.0
        if len(self.rewards) > 0:
            runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):

            print("episode index:" + str(self.epIdx))
            state = self.env.reset()
            done = False
            rewardSum = 0

            for stepCount in range(self.episodeLength):
                timeStep = state['timeStep']
                timeIdx = self.timeIndexMap[timeStep]

                action = self.select_action(self.actorNets[timeIdx], state, noiseFlag=True)

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
            'valueNet_state_dict': [net.state_dict() for net in self.valueNets],
            'softQNetOne_state_dict': [net.state_dict() for net in self.softQNetsOne],
            'softQNetTwo_state_dict': [net.state_dict() for net in self.softQNetsTwo],
            'actor_optimizer_state_dict': [opt.state_dict() for opt in self.actor_optimizers],
            'value_optimizer_state_dict': [opt.state_dict() for opt in self.value_optimizers],
            'softQOne_optimizer_state_dict': [opt.state_dict() for opt in self.softQOne_optimizers],
            'softQTwo_optimizer_state_dict': [opt.state_dict() for opt in self.softQTwo_optimizers]
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
            'valueNet_state_dict': [net.state_dict() for net in self.valueNets],
            'softQNetOne_state_dict': [net.state_dict() for net in self.softQNetsOne],
            'softQNetTwo_state_dict': [net.state_dict() for net in self.softQNetsTwo],
            'actor_optimizer_state_dict': [opt.state_dict() for opt in self.actor_optimizers],
            'value_optimizer_state_dict': [opt.state_dict() for opt in self.value_optimizers],
            'softQOne_optimizer_state_dict': [opt.state_dict() for opt in self.softQOne_optimizers],
            'softQTwo_optimizer_state_dict': [opt.state_dict() for opt in self.softQTwo_optimizers]
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
            self.actorNets[i].load_state_dict(checkpoint['actorNet_state_dict'][i])
            self.valueNets[i].load_state_dict(checkpoint['criticNet_state_dict'][i])
            self.valueTargetNets[i].load_state_dict(checkpoint['criticNet_state_dict'][i])
            self.softQNetsOne[i].load_state_dict(checkpoint['softQNetOne_state_dict'][i])
            self.softQNetsTwo[i].load_state_dict(checkpoint['softQNetTwo_state_dict'][i])

        for i in range(len(self.actor_optimizers)):
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizer_state_dict'][i])
            self.value_optimizers[i].load_state_dict(checkpoint['value_optimizer_state_dict'][i])
            self.softQOne_optimizers[i].load_state_dict(checkpoint['softQOne_optimizer_state_dict'][i])
            self.softQTwo_optimizers[i].load_state_dict(checkpoint['softQTwo_optimizer_state_dict'][i])
