import torch
import os
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition
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

    def update_net_on_transitions(self, transitions_raw):
        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.float32)  # shape(batch, numActions)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32)  # shape(batch)
        batchSize = reward.shape[0]

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device,
                                        dtype=torch.uint8)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device,
                                             dtype=torch.float32)

        # Critic loss
        QValues = self.criticNet.forward(state, action).squeeze()
        # next action is calculated using target actor network

        QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
        if len(nonFinalNextState):
            next_actions = self.actorNet_target.forward(nonFinalNextState)
            QNext[nonFinalNextState] = self.criticNet_target.forward(nonFinalNextState, next_actions.detach()).squeeze()

        targetValues = reward + self.gamma * QNext
        critic_loss = self.netLossFunc(QValues, targetValues)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.criticNet.parameters(), self.netGradClip)

        self.critic_optimizer.step()

        # Actor loss
        # we try to maximize criticNet output(which is state value)

        # update networks
        if self.learnStepCounter % self.policyUpdateFreq == 0:
            policy_loss = -self.criticNet.forward(state, self.actorNet.forward(state)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            if self.netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), self.netGradClip)

            self.actor_optimizer.step()

            if self.globalStepCount % self.lossRecordStep == 0:
                self.losses.append([self.globalStepCount, self.epIdx, critic_loss.item(), policy_loss.item()])

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