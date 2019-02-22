
from Agents.Core.Agent import Agent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.PrioritizedReplayMemory import PrioritizedReplayMemory
from utils.utils import torchvector
import random
import torch
import torch.optim
import numpy as np
from enum import Enum
import simplejson as json
import os
import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DQNAgent(Agent):

    def __init__(self, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, **kwargs):
        super(DQNAgent, self).__init__(**kwargs)

        self.read_config()
        self.policyNet = policyNet
        self.targetNet = targetNet
        self.env = env
        self.optimizer = optimizer
        self.numAction = nbAction
        if self.priorityMemoryOption:
            self.memory = PrioritizedReplayMemory(self.memoryCapacity, self.config)
        else:
            self.memory = ReplayMemory(self.memoryCapacity)
        self.learnStepCounter = 0  #for target net update
        self.stateProcessor = stateProcessor
        self.netLossFunc = netLossFunc

        # move model to correct device
        self.policyNet = self.policyNet.to(self.device)
        self.targetNet = self.targetNet.to(self.device)

        self.dirName = self.config['mapName'] + 'Log/'
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)


        self.nStepBuffer = []

    def read_config(self):
        self.trainStep = self.config['trainStep']
        self.targetNetUpdateStep = self.config['targetNetUpdateStep']
        self.memoryCapacity = self.config['memoryCapacity']
        self.trainBatchSize = self.config['trainBatchSize']
        self.gamma = self.config['gamma']

        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']
        self.netUpdateOption = 'targetNet'
        if 'netUpdateOption' in self.config:
            self.netUpdateOption = self.config['netUpdateOption']
        self.priorityMemoryOption = False
        if 'priorityMemoryOption' in self.config:
            self.priorityMemoryOption = self.config['priorityMemoryOption']
        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']
        self.netUpdateFrequency = 1
        if 'netUpdateFrequency' in self.config:
            self.netUpdateFrequency = self.config['netUpdateFrequency']
        self.nStepForward = 1
        if 'nStepForward' in self.config:
            self.nStepForward = self.config['nStepForward']


    def select_action(self, state, epsThreshold):

        # get a random number so that we can do epsilon exploration
        randNum = random.random()
        if randNum > epsThreshold:
            with torch.no_grad():
                # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
                # here state[np.newaxis,:] is to add a batch dimension
                if self.stateProcessor is not None:
                    state = self.stateProcessor([state])
                    QValues = self.policyNet(state)
                else:
                    QValues = self.policyNet(torchvector(state[np.newaxis, :]).to(self.device))
                action = torch.argmax(QValues).item()

        else:
            action = random.randint(0, self.numAction-1)
        return action

    def getPolicy(self, state):
        return self.select_action(state, -0.01)

    def train(self):

        runningAvgEpisodeReward = 0.0

        self.epIdx = 0
        for self.epIdx in range(self.trainStep):
            print("episode index:" + str(self.epIdx))
            state = self.env.reset()
            done = False
            rewardSum = 0
            stepCount = 0

            # clear the nstep buffer
            self.nStepBuffer.clear()

            while not done:
                self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

                action = self.select_action(state, self.epsThreshold)

                nextState, reward, done, info = self.env.step(action)

                if stepCount == 0:
                    print("at step 0:")
                    print(info)

                # learn the transition
                self.update_net(state, action, nextState, reward)

                state = nextState
                rewardSum += reward


                if self.verbose:
                    print('action: ' + str(action))
                    print('state:')
                    print(nextState)
                    print('reward:')
                    print(reward)
                    print('info')
                    print(info)


                if done:
                    runningAvgEpisodeReward = (runningAvgEpisodeReward*self.epIdx + rewardSum)/(self.epIdx + 1)
                    print("done in step count: {}".format(stepCount))
                    print("reward sum = " + str(rewardSum))
                    print("running average episode reward sum: {}".format(runningAvgEpisodeReward))
                    print(info)



                    self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, runningAvgEpisodeReward])
                    if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
                        self.save_checkpoint()
                    break

                stepCount += 1
                self.globalStepCount += 1

        self.save_all()

    def store_experience(self, state, action, nextState, reward):
        self.nStepBuffer.append((state, action, nextState, reward))

        if len(self.nStepBuffer) < self.nStepForward:
            return

        R = sum([self.nStepBuffer[i][3]*(self.gamma**i) for i in range(self.nStepForward)])

        state, action, _, _ = self.nStepBuffer.pop(0)

        if self.priorityMemoryOption:
            self.memory.store(Transition(state, action, nextState, R))
        else:
            self.memory.push(state, action, nextState, R)

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

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state = self.stateProcessor(transitions.state)
            nextState = self.stateProcessor(transitions.next_state)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nextState = torch.tensor(transitions.next_state, device=self.device, dtype=torch.float32)
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1) # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)

        batchSize = reward.shape[0]

        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            QValues = self.policyNet(state).gather(1, action)

            if updateOption == 'targetNet':
                 # Here we detach because we do not want gradient flow from target values to net parameters
                 QNext = self.targetNet(nextState).detach()
                 targetValues = reward + self.gamma * QNext.max(dim=1)[0].unsqueeze(-1)
            if updateOption == 'policyNet':
                targetValues = reward + self.gamma * torch.max(self.policyNet(nextState).detach(), dim=1)[0].unsqueeze(-1)
            if updateOption == 'doubleQ':
                 # select optimal action from policy net
                 with torch.no_grad():
                    batchAction = self.policyNet(nextState).max(dim=1)[1].unsqueeze(-1)
                    QNext = self.targetNet(nextState).detach()
                    targetValues = reward + self.gamma * QNext.gather(1, batchAction)

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
                action = self.select_action(state, -0.01)
                nextState, reward, done, info = self.env.step(action)
                stepCount += 1
                if memory is not None:
                    memory.push(epIdx, stepCount, state, action, nextState, reward, info)
                state = nextState
                rewardSum += reward
                if done:
                    print("done in step count: {}".format(stepCount))
                    break
            print("reward sum = " + str(rewardSum))

    def testPolicyNet(self, episodes, memory = None):
        return self.perform_on_policy(episodes, self.getPolicy, memory)

    def save_all(self):
        prefix = self.dirName + self.config['mapName'] + 'Finalepoch' + str(self.epIdx + 1)
        torch.save({
            'epoch': self.epIdx + 1,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')

    def save_checkpoint(self):
        prefix = self.dirName + self.config['mapName'] + 'Epoch' + str(self.epIdx + 1)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        torch.save({
            'epoch': self.epIdx + 1,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')