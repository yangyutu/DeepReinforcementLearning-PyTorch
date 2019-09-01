import torch
import os
import numpy as np
from Agents.Core.ReplayMemory import ReplayMemory, Transition

import pickle
class DDPGAgent:
    def __init__(self, config, actorNets, criticNets, env, optimizers, netLossFunc, nbAction, stateProcessor = None, experienceProcessor = None):

        self.config = config
        self.read_config()
        self.env = env
        self.numAction = nbAction
        self.stateProcessor = stateProcessor
        self.netLossFunc = netLossFunc
        self.experienceProcessor = experienceProcessor


        self.initialization()
        self.init_memory()
        self.initalizeNets(actorNets, criticNets, optimizers)


    def initalizeNets(self, actorNets, criticNets, optimizers):
        self.actorNet = actorNets['actor']
        self.actorNet_target = actorNets['target'] if 'target' in actorNets else None
        self.criticNet = criticNets['critic']
        self.criticNet_target = criticNets['target'] if 'target' in criticNets else None
        self.actor_optimizer = optimizers['actor']
        self.critic_optimizer = optimizers['critic']

        self.net_to_device()

    def init_memory(self):
        self.memory = ReplayMemory(self.memoryCapacity)

    def read_config(self):
        self.trainStep = self.config['trainStep']
        self.targetNetUpdateStep = 10000
        if 'targetNetUpdateStep' in self.config:
            self.targetNetUpdateStep = self.config['targetNetUpdateStep']

        self.trainBatchSize = self.config['trainBatchSize']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']
        self.netUpdateOption = 'targetNet'
        if 'netUpdateOption' in self.config:
            self.netUpdateOption = self.config['netUpdateOption']
        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']
        self.netUpdateFrequency = 1
        if 'netUpdateFrequency' in self.config:
            self.netUpdateFrequency = self.config['netUpdateFrequency']
        self.nStepForward = 1
        if 'nStepForward' in self.config:
            self.nStepForward = self.config['nStepForward']
        self.lossRecordStep = 10
        if 'lossRecordStep' in self.config:
            self.lossRecordStep = self.config['lossRecordStep']
        self.episodeLength = 500
        if 'episodeLength' in self.config:
            self.episodeLength = self.config['episodeLength']

        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']

        self.device = 'cpu'
        if 'device' in self.config and torch.cuda.is_available():
            self.device = self.config['device']


        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        self.memoryCapacity = self.config['memoryCapacity']

        self.hindSightER = False
        if 'hindSightER' in self.config:
            self.hindSightER = self.config['hindSightER']
            self.hindSightERFreq = self.config['hindSightERFreq']

        self.experienceAugmentation = False
        if 'experienceAugmentation' in self.config:
            self.experienceAugmentation = self.config['experienceAugmentation']
            self.experienceAugmentationFreq = self.config['experienceAugmentationFreq']


        self.policyUpdateFreq = 1
        if 'policyUpdateFreq' in self.config:
            self.policyUpdateFreq = self.config['policyUpdateFreq']

    def net_to_device(self):
        # move model to correct device
        self.actorNet = self.actorNet.to(self.device)
        self.criticNet = self.criticNet.to(self.device)

        # in case targetNet is None
        if self.actorNet_target is not None:
            self.actorNet_target = self.actorNet_target.to(self.device)
        # in case targetNet is None
        if self.criticNet_target is not None:
            self.criticNet_target = self.criticNet_target.to(self.device)

    def initialization(self):

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

        self.runningAvgEpisodeReward = 0.0

    def select_action(self, net, state, noiseFlag = False):

        with torch.no_grad():
            # self.policyNet(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
            # here state[np.newaxis,:] is to add a batch dimension
            if self.stateProcessor is not None:
                state, _ = self.stateProcessor([state], self.device)
                action = net.select_action(state, noiseFlag)
            else:
                stateTorch = torch.from_numpy(np.array(state[np.newaxis, :], dtype = np.float32))
                action = net.select_action(stateTorch.to(self.device), noiseFlag)

        return action.cpu().data.numpy()[0]
    def process_hindSightExperience(self, state, action, nextState, reward, info):
        if nextState is not None and self.globalStepCount % self.hindSightERFreq == 0:
            stateNew, actionNew, nextStateNew, rewardNew = self.env.getHindSightExperience(state, action, nextState, info)
            if stateNew is not None:
                transition = Transition(stateNew, actionNew, nextStateNew, rewardNew)
                self.memory.push(transition)
                if self.experienceAugmentation:
                    self.process_experienceAugmentation(state, action, nextState, reward, info)

    def process_experienceAugmentation(self, state, action, nextState, reward, info):
        if self.globalStepCount % self.experienceAugmentationFreq == 0:
            state_Augs, action_Augs, nextState_Augs, reward_Augs = self.env.getExperienceAugmentation(state, action, nextState,
                                                                                            reward, info)
            for i in range(len(state_Augs)):
                transition = Transition(state_Augs[i], action_Augs[i], nextState_Augs[i], reward_Augs[i])
                self.memory.push(transition)

    def store_experience(self, state, action, nextState, reward, info):
        if self.experienceProcessor is not None:
            state, action, nextState, reward = self.experienceProcessor(state, action, nextState, reward, info)

        transition = Transition(state, action, nextState, reward)
        self.memory.push(transition)

        if self.experienceAugmentation:
            self.process_experienceAugmentation(state, action, nextState, reward, info)

        if self.hindSightER:
            self.process_hindSightExperience(state, action, nextState, reward, info)


    def prepare_minibatch(self, transitions_raw):
        # first store memory

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

        # first store memory

        self.store_experience(state, action, nextState, reward, info)

        # prepare mini-batch
        if len(self.memory) < self.trainBatchSize:
            return

        transitions_raw = self.memory.sample(self.trainBatchSize)

        self.update_net_on_transitions(transitions_raw)

        self.copy_nets()

        self.learnStepCounter += 1

    def copy_nets(self):
        # update networks
        if self.learnStepCounter % self.policyUpdateFreq == 0:
            # update target networks
            for target_param, param in zip(self.actorNet_target.parameters(), self.actorNet.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.criticNet_target.parameters(), self.criticNet.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update_net_on_transitions(self, transitions_raw):

        state, nonFinalMask, nonFinalNextState, action, reward = self.prepare_minibatch(transitions_raw)

        # Critic loss
        QValues = self.criticNet.forward(state, action).squeeze()
        QNext = torch.zeros(self.trainBatchSize, device=self.device, dtype=torch.float32)

        if len(nonFinalNextState):
            # next action is calculated using target actor network
            next_actions = self.actorNet_target.forward(nonFinalNextState)
            QNext[nonFinalMask] = self.criticNet_target.forward(nonFinalNextState, next_actions.detach()).squeeze()

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


    def work_before_step(self, state):
        pass


    def train_one_episode(self):

        print("episode index:" + str(self.epIdx))
        state = self.env.reset()
        done = False
        rewardSum = 0

        for stepCount in range(self.episodeLength):

            # any work to be done before select actions
            self.work_before_step(state)

            action = self.select_action(self.actorNet, state, noiseFlag=True)

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

        self.runningAvgEpisodeReward = (self.runningAvgEpisodeReward * self.epIdx + rewardSum) / (self.epIdx + 1)
        print("done in step count: {}".format(stepCount))
        print("reward sum = " + str(rewardSum))
        print("running average episode reward sum: {}".format(self.runningAvgEpisodeReward))
        print(info)

        self.rewards.append([self.epIdx, stepCount, self.globalStepCount, rewardSum, self.runningAvgEpisodeReward])
        if self.config['logFlag'] and self.epIdx % self.config['logFrequency'] == 0:
            self.save_checkpoint()

        self.epIdx += 1

    def train(self):

        # continue on historical training
        if len(self.rewards) > 0:
            self.runningAvgEpisodeReward = self.rewards[-1][-1]

        for trainStepCount in range(self.trainStep):
            self.train_one_episode()
            

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
            pickle.dump(self.memory, file)

        torch.save({
            'epoch': self.epIdx,
            'globalStep': self.globalStepCount,
            'actorNet_state_dict': self.actorNet.state_dict(),
            'criticNet_state_dict': self.criticNet.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
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
            'criticNet_state_dict': self.criticNet.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
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
        self.criticNet.load_state_dict(checkpoint['criticNet_state_dict'])
        self.criticNet_target.load_state_dict(checkpoint['criticNet_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])