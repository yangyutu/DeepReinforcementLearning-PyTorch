import torch
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.DDPG.DDPG import DDPGAgent

import pickle


class TDDDPGAgent(DDPGAgent):
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor=None, experienceProcessor = None):

        self.config = config
        self.read_config()
        self.actorNet = actorNets['actor']
        self.actorNet_target = actorNets['target'] if 'target' in actorNets else None
        self.criticNetOne = criticNets['criticOne']
        self.criticNet_targetOne = criticNets['targetOne'] if 'targetOne' in criticNets else None
        self.criticNetTwo = criticNets['criticTwo']
        self.criticNet_targetTwo = criticNets['targetTwo'] if 'targetTwo' in criticNets else None

        self.env = env
        self.actor_optimizer = optimizers['actor']
        self.criticOne_optimizer = optimizers['criticOne']
        self.criticTwo_optimizer = optimizers['criticTwo']
        self.numAction = nbAction
        self.stateProcessor = stateProcessor
        self.netLossFunc = netLossFunc
        self.experienceProcessor = experienceProcessor
        self.initialization()

    def read_config(self):
        super(TDDDPGAgent, self).read_config()

        self.policyUpdateFreq = 2
        if 'policyUpdateFreq' in self.config:
            self.policyUpdateFreq = self.config['policyUpdateFreq']
        self.policySmoothNoise = 0.01
        if 'policySmoothNoise' in self.config:
            self.policyUpdateFreq = self.config['policySmoothNoise']

    def net_to_device(self):
        # move model to correct device
        self.actorNet = self.actorNet.to(self.device)
        self.criticNetOne = self.criticNetOne.to(self.device)
        self.criticNetTwo = self.criticNetTwo.to(self.device)

        # in case targetNet is None
        if self.actorNet_target is not None:
            self.actorNet_target = self.actorNet_target.to(self.device)
        # in case targetNet is None
        if self.criticNet_targetOne is not None:
            self.criticNet_targetOne = self.criticNet_targetOne.to(self.device)
        if self.criticNet_targetTwo is not None:
            self.criticNet_targetTwo = self.criticNet_targetTwo.to(self.device)

    def prepare_minibatch(self, state, action, nextState, reward, info):
        # first store memory

        self.store_experience(state, action, nextState, reward, info)
        if len(self.memory) < self.trainBatchSize:
            return
        transitions_raw = self.memory.sample(self.trainBatchSize)
        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.float32)  # shape(batch, numActions)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32)  # shape(batch)

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

        return state, nonFinalMask, nonFinalNextState, action, reward

    def update_net(self, state, action, nextState, reward, info):

        # state, nonFinalMask, nonFinalNextState, action, reward = self.prepare_minibatch(state, action, nextState, reward, info)
        self.store_experience(state, action, nextState, reward, info)
        if len(self.memory) < self.trainBatchSize:
            return
        transitions_raw = self.memory.sample(self.trainBatchSize)
        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.float32)  # shape(batch, numActions)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32)  # shape(batch)

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

        batchSize = reward.shape[0]

        # Critic loss
        QValuesOne = self.criticNetOne.forward(state, action).squeeze()
        QValuesTwo = self.criticNetTwo.forward(state, action).squeeze()

        actionNoise = torch.randn((nonFinalNextState.shape[0], self.numAction), dtype=torch.float32, device=self.device)
        next_actions = self.actorNet_target.forward(nonFinalNextState) + actionNoise * self.policySmoothNoise

        # next_actions = self.actorNet_target.forward(nonFinalNextState)

        QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
        QNextCriticOne = self.criticNet_targetOne.forward(nonFinalNextState, next_actions.detach()).squeeze()
        QNextCriticTwo = self.criticNet_targetTwo.forward(nonFinalNextState, next_actions.detach()).squeeze()

        QNext[nonFinalMask] = torch.min(QNextCriticOne, QNextCriticTwo)

        targetValues = reward + self.gamma * QNext

        criticOne_loss = self.netLossFunc(QValuesOne, targetValues)
        criticTwo_loss = self.netLossFunc(QValuesTwo, targetValues)

        self.criticOne_optimizer.zero_grad()
        self.criticTwo_optimizer.zero_grad()

        # https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
        criticOne_loss.backward(retain_graph=True)
        criticTwo_loss.backward()

        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.criticNetOne.parameters(), self.netGradClip)
            torch.nn.utils.clip_grad_norm_(self.criticNetTwo.parameters(), self.netGradClip)

        self.criticOne_optimizer.step()
        self.criticTwo_optimizer.step()

        if self.learnStepCounter % self.policyUpdateFreq:
            # Actor loss
            # we try to maximize criticNet output(which is state value)
            policy_loss = -self.criticNetOne.forward(state, self.actorNet.forward(state)).mean()

            # update networks
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            if self.netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), self.netGradClip)

            self.actor_optimizer.step()

            if self.globalStepCount % self.lossRecordStep == 0:
                self.losses.append([self.globalStepCount, self.epIdx, criticOne_loss.item(), criticTwo_loss.item(),
                                    policy_loss.item()])

                # update target networks
                for target_param, param in zip(self.actorNet_target.parameters(), self.actorNet.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.criticNet_targetOne.parameters(), self.criticNetOne.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.criticNet_targetTwo.parameters(), self.criticNetTwo.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.learnStepCounter += 1

    def save_all(self):
        prefix = self.dirName + self.identifier + 'Finalepoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'actorNet_state_dict': self.actorNet.state_dict(),
            'criticNetOne_state_dict': self.criticNetOne.state_dict(),
            'criticNetTwo_state_dict': self.criticNetTwo.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'criticOne_optimizer_state_dict': self.criticOne_optimizer.state_dict(),
            'criticTwo_optimizer_state_dict': self.criticOne_optimizer.state_dict()
        }, prefix + '_checkpoint.pt')

    def save_checkpoint(self):
        prefix = self.dirName + self.identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'actorNet_state_dict': self.actorNet.state_dict(),
            'criticNetOne_state_dict': self.criticNetOne.state_dict(),
            'criticNetTwo_state_dict': self.criticNetTwo.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'criticOne_optimizer_state_dict': self.criticOne_optimizer.state_dict(),
            'criticTwo_optimizer_state_dict': self.criticTwo_optimizer.state_dict()
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        self.loadLosses(prefix + '_loss.txt')
        self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memory = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        self.actorNet.load_state_dict(checkpoint['actorNet_state_dict'])
        self.actorNet_target.load_state_dict(checkpoint['actorNet_state_dict'])
        self.criticNetOne.load_state_dict(checkpoint['criticNetOne_state_dict'])
        self.criticNet_targetOne.load_state_dict(checkpoint['criticNetOne_state_dict'])
        self.criticNetTwo.load_state_dict(checkpoint['criticNetTwo_state_dict'])
        self.criticNet_targetTwo.load_state_dict(checkpoint['criticNetTwo_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.criticOne_optimizer.load_state_dict(checkpoint['criticOne_optimizer_state_dict'])
        self.criticTwo_optimizer.load_state_dict(checkpoint['criticTwo_optimizer_state_dict'])