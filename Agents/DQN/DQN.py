
from Agents.DQN.BaseDQN import BaseDQNAgent
from Agents.Core.ReplayMemory import ReplayMemory, Transition
from Agents.Core.ReplayMemoryReward import ReplayMemoryReward
from Agents.Core.PrioritizedReplayMemory import PrioritizedReplayMemory
import random
import torch
import torch.optim
import numpy as np
import simplejson as json
import pickle


class DQNAgent(BaseDQNAgent):
    """class for DQN  agents.
        This class contains implementation of DQN learning with various enhancement: double Q,
        prioritized experience replay, hindsight experience replay, etc.
        # Arguments
            config: a dictionary for training parameters
            policyNet: neural network for Q learning
            targetNet: a slowly changing policyNet to provide Q value targets
            env: environment for the agent to interact. env should implement same interface of a gym env
            optimizer: a network optimizer
            netLossFunc: loss function of the network, e.g., mse
            nbAction: number of actions
            stateProcessor: a function to process output from env, processed state will be used as input to the networks
            experienceProcessor: additional steps to process an experience
        """
    def __init__(self, config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor = None, experienceProcessor=None):
        super(DQNAgent, self).__init__(config, policyNet, targetNet, env, optimizer, netLossFunc, nbAction, stateProcessor, experienceProcessor)
        # initialize memory units
        self.init_memory()


    def init_memory(self):

        if self.priorityMemoryOption:
            self.memory = PrioritizedReplayMemory(self.memoryCapacity, self.config)
        else:
            # most commonly experience replay memory
            if self.memoryOption == 'natural':
                self.memory = ReplayMemory(self.memoryCapacity)
            elif self.memoryOption == 'reward':
                self.memory = ReplayMemoryReward(self.memoryCapacity, self.config['rewardMemoryBackupStep'],
                                                 self.gamma, self.config['rewardMemoryTerminalRatio'] )

    def read_config(self):
        '''
        reading additional configurations
        memoryCapacity, memoryOption
        priorityMemoryOption

        '''
        #

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

    def work_At_Episode_Begin(self):
        '''
        stuff to do before each episode

        :return:
        '''
        # clear the nstep buffer
        self.nStepBuffer.clear()

    def work_before_step(self, state=None):
        '''
        stuff to do before each step
        :param state:
        :return:
        '''
        self.epsThreshold = self.epsilon_by_step(self.globalStepCount)

    def train_one_episode(self):

        print("episode index:" + str(self.epIdx))
        state = self.env.reset()
        done = False
        rewardSum = 0

        # any work to be done at episode begin
        self.work_At_Episode_Begin()

        for stepCount in range(self.episodeLength):

            # any work to be done before select actions
            self.work_before_step(state)

            action = self.select_action(self.policyNet, state, self.epsThreshold)

            nextState, reward, done, info = self.env.step(action)

            if stepCount == 0:
                print("at step 0:")
                print(info)

            if done:
                nextState = None

            # store, augment, learn the transition
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

        self.runningAvgEpisodeReward = (self.runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
        print("done in step count: {}".format(stepCount))
        print("reward sum = " + str(rewardSum))
        print("running average episode reward sum: {}".format(self.runningAvgEpisodeReward))
        print(info)

        self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, self.runningAvgEpisodeReward])
        if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
            self.save_checkpoint()

        self.epIdx += 1

        return stepCount, rewardSum

    def train(self):

        if len(self.rewards) > 0:
            self.runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):
            self.train_one_episode()
        self.save_all()

    def store_experience(self, state, action, nextState, reward, info):
        '''
        store experience tuple (state, action, nextState, reward)
        '''
        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, info)
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




    def update_net(self, state, action, nextState, reward, info):
        '''
        This routine will store, transform, augment experiences and sample experiences for gradient descent.
        '''


        self.store_experience(state, action, nextState, reward, info)

        if self.hindSightER and nextState is not None and self.globalStepCount % self.hindSightERFreq == 0:
            stateNew, actionNew, nextStateNew, rewardNew = self.env.getHindSightExperience(state, action, nextState, info)
            if stateNew is not None:
                self.store_experience(stateNew, actionNew, nextStateNew, rewardNew, info)


        if self.priorityMemoryOption:
            if len(self.memory) < self.config['memoryCapacity']:
                return
        else:
            if len(self.memory) < self.trainBatchSize:
                return


        # update net with specified frequency
        if self.globalStepCount % self.netUpdateFrequency == 0:
            # sample experience
            for nStep in range(self.netUpdateStep):
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

    def prepare_minibatch(self, transitions_raw):
        '''
        do some proprocessing work for transitions_raw
        order the data
        convert transition list to torch tensors
        use trick from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        '''

        transitions = Transition(*zip(*transitions_raw))
        action = torch.tensor(transitions.action, device=self.device, dtype=torch.long).unsqueeze(-1)  # shape(batch, 1)
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(-1)  # shape(batch, 1)

        # for some env, the output state requires further processing before feeding to neural network
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device,
                                        dtype=torch.bool)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device,
                                             dtype=torch.float32)

        return state, nonFinalMask, nonFinalNextState, action, reward

    def update_net_on_transitions(self, transitions_raw, loss_fun, gradientStep = 1, updateOption='policyNet', netGradClip=None, info=None):
        '''
        This function performs gradient gradient on the network
        '''

        state, nonFinalMask, nonFinalNextState, action, reward = self.prepare_minibatch(transitions_raw)

        for step in range(gradientStep):
            # calculate Qvalues based on selected action batch
            QValues = self.policyNet(state).gather(1, action)

            if updateOption == 'targetNet':
                 # Here we detach because we do not want gradient flow from target values to net parameters
                 QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)
                 QNext[nonFinalMask] = self.targetNet(nonFinalNextState).max(1)[0].detach()
                 targetValues = reward + (self.gamma**self.nStepForward) * QNext.unsqueeze(-1)
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
            # zero gradient
            self.optimizer.zero_grad()

            loss.backward()
            if netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.policyNet.parameters(), netGradClip)
            self.optimizer.step()

            return loss.item()


    def save_all(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        prefix = self.dirName + identifier + 'Finalepoch' + str(self.epIdx)
        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')

    def save_checkpoint(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        prefix = self.dirName + identifier + 'Epoch' + str(self.epIdx)
        self.saveLosses(prefix + '_loss.txt')
        self.saveRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'model_state_dict': self.policyNet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):
        #self.loadLosses(prefix + '_loss.txt')
        #self.loadRewards(prefix + '_reward.txt')
        with open(prefix + '_memory.pickle', 'rb') as file:
            self.memory = pickle.load(file)

        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.epIdx = checkpoint['epoch']
        self.globalStepCount = checkpoint['globalStep']
        self.policyNet.load_state_dict(checkpoint['model_state_dict'])
        self.targetNet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])