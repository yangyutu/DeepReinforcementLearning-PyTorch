import torch
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.DDPG.DDPG import DDPGAgent

import pickle


class SACAgent(DDPGAgent):
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor=None,
                 experienceProcessor=None):

        super(SACAgent, self).__init__(config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction,
                                          stateProcessor, experienceProcessor)

    def initalizeNets(self, actorNets, criticNets, optimizers):

        # totally five nets
        self.actorNet = actorNets['actor']

        self.softQNetOne = criticNets['softQOne']
        self.softQNetTwo = criticNets['softQTwo']

        self.valueNet = criticNets['value']
        self.valueTargetNet = criticNets['valueTarget']

        self.actor_optimizer = optimizers['actor']
        self.softQOne_optimizer = optimizers['softQOne']
        self.softQTwo_optimizer = optimizers['softQTwo']
        self.value_optimizer = optimizers['value']

        self.net_to_device()

    def init_memory(self):
        self.memory = ReplayMemory(self.memoryCapacity)

    def read_config(self):
        super(SACAgent, self).read_config()

        self.SACAlpha = 1
        if 'SACAlpha' in self.config:
            self.SACAlpha = self.config['SACAlpha']

    def net_to_device(self):
        # move model to correct device, totally 5 nets
        self.actorNet = self.actorNet.to(self.device)
        self.softQNetOne = self.softQNetOne.to(self.device)
        self.softQNetTwo = self.softQNetTwo.to(self.device)
        self.valueNet = self.valueNet.to(self.device)
        self.valueTargetNet = self.valueTargetNet.to(self.device)


    def update_net(self, state, action, nextState, reward, info):

        # first store memory

        self.store_experience(state, action, nextState, reward, info)

        # prepare mini-batch
        if len(self.memory) < self.trainBatchSize:
            return

        state, nonFinalMask, nonFinalNextState, action, reward = \
            self.prepare_minibatch(state, action, nextState, reward, info)

        # now do net update
        # Q and value nets evaluation
        predicted_q_value1 = self.softQNetOne(state, action).squeeze()
        predicted_q_value2 = self.softQNetTwo(state, action).squeeze()

        predicted_value = self.valueNet(state).squeeze()

        # action for next state
        next_action, log_prob = self.actorNet.select_action(state, probFlag=True)

        # Training Q Function, using value function as target
        target_value = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
        target_value[nonFinalMask] = self.valueTargetNet(nonFinalNextState).squeeze()
        target_q_value = reward + self.gamma * target_value

        q_value_loss1 = self.netLossFunc(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.netLossFunc(predicted_q_value2, target_q_value.detach())

        self.softQOne_optimizer.zero_grad()
        q_value_loss1.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.softQNetOne.parameters(), self.netGradClip)
        self.softQOne_optimizer.step()

        self.softQTwo_optimizer.zero_grad()
        q_value_loss2.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.softQNetTwo.parameters(), self.netGradClip)
        self.softQTwo_optimizer.step()

        # Training Value Function
        predicted_new_q_value = torch.min(self.softQNetOne(state, next_action), self.softQNetTwo(state, next_action))
        ## the log_prob is the entropy term
        target_value_func = (predicted_new_q_value - self.SACAlpha * log_prob).squeeze()
        value_loss = self.netLossFunc(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.valueNet.parameters(), self.netGradClip)
        self.value_optimizer.step()


        # Training Policy Function
        policy_loss = - (predicted_new_q_value - self.SACAlpha * log_prob).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.netGradClip is not None:
            torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), self.netGradClip)
        self.actor_optimizer.step()

        for target_param, param in zip(self.valueTargetNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


        if self.globalStepCount % self.lossRecordStep == 0:
            self.losses.append([self.globalStepCount, self.epIdx, value_loss.item(), policy_loss.item()])

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
            'softQNetOne_state_dict': self.softQNetOne.state_dict(),
            'softQNetTwo_state_dict': self.softQNetTwo.state_dict(),
            'valueNet_state_dict': self.valueNet.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'softQOne_optimizer_state_dict': self.softQOne_optimizer.state_dict(),
            'softQTwo_optimizer_state_dict': self.softQTwo_optimizer.state_dict()

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
            'softQNetOne_state_dict': self.softQNetOne.state_dict(),
            'softQNetTwo_state_dict': self.softQNetTwo.state_dict(),
            'valueNet_state_dict': self.valueNet.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'softQOne_optimizer_state_dict': self.softQOne_optimizer.state_dict(),
            'softQTwo_optimizer_state_dict': self.softQTwo_optimizer.state_dict()

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
        self.softQNetOne.load_state_dict(checkpoint['softQNetOne_state_dict'])
        self.softQNetTwo.load_state_dict(checkpoint['softQNetTwo_state_dict'])
        self.valueNet.load_state_dict(checkpoint['valueNet_state_dict'])
        self.valueTargetNet.load_state_dict(checkpoint['valueNet_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.softQOne_optimizer.load_state_dict(checkpoint['softQOne_optimizer_state_dict'])
        self.softQTwo_optimizer.load_state_dict(checkpoint['softQTwo_optimizer_state_dict'])
        self.valueNet_optimizer.load_state_dict(checkpoint['valueNet_optimizer_state_dict'])