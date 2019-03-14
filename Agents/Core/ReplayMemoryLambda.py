
from collections import deque
from collections import namedtuple
import random
import numpy as np
import math
import torch
# Define a namedtuple with name Transition and attributes of state, action,
# next_state, reward
AugumentedTransition = namedtuple('AugumentedTransition', ('state', 'action', 'next_state', 'reward', 'process_reward'))

class ReplayMemoryLambda(object):
    def __init__(self, config, capacity, net = None, stateProcessor=None):
    ## capacity of episodes
        self.capacity = capacity
        self.partitionNum = config['eligibilityTracePartitionNum']
        self.partitionLength = self.capacity / self.partitionNum
        self.paritionldx = 0
        self.paritionRange = (0, 0)
        self.memory = deque()
        self.episodeLengths = deque()
        self.position = 0
        self.net = net
        self.lamb = config['eligibilityTraceLambda']
        self.gamma = config['gamma']
        self.partitionUpdateFlag = False
        if 'partitionUpdateFlag' in config:
            self.partitionUpdateFlag = config['partitionUpdateFlag']
        self.stateProcessor = stateProcessor

        self.device = 'cpu'
        if 'device' in config:
            self.device = config['device']

    def push(self, episode):
    #Saves an episode of transitions
        if len(self.memory) < self.capacity:
            self.append(episode)
        else:
            self.popLeft()
            self.append(episode)

    def append(self, episode):
        self.episodeLengths.append(len(episode))
        self.totalLength += len(episode)
        for e in episode:
            e_aug = AugumentedTransition(e[0], e[1], e[2], e[3], 0)
            self.memory.append(e_aug)

    def popLeft(self):
        for i in range(self.episodeLengths[0]):
            self.memory.popleft()
        self.totalLength -= self.episodeLengths[0]
        self.episodeLengths.popLeft()

    def sample(self, batch_size):
        return random.sample(self.memory[self.partitionRange[0]:self.partitionRange[1]], batch_size)

    def refresh(self):
        if self.partitionUpdateFlag:
            # every time refresh, we only refresh current partition
            # first update partition if necessary
            if len(self.episodeLengths) == self.capacity:
                actualPartitionNum = self.partitionNum
            else:
                actualPartitionNum = int(math.floor(self.totalLength / self.partitionLength))

            accumLength = np.cumsum(self.episodeLengths)
            selectPartition = random.randomInt(0, actualPartitionNum)
            episodeStart = selectPartition*self.partitionLength
            episodeEnd = (selectPartition + 1)*self.paritionLength
            self.partitionRange = (accumLength[episodeStart] - self.episodeLength[episodeStart],
            accumLength[episodeEnd - 1])
        else:
            episodeStart = 0
            episodeEnd = len(self.memory)
            self.partitionRange = (0, self.totalLength)

        transitions = AugumentedTransition(*zip(*self.memory[self.partitionRange[0]: self.partitionRange[1]]))
        reward = torch.tensor(transitions.reward, device=self.device, dtype=torch.float32).unsqueeze(-1) # shape(batch, 1)
        batchSize = reward.shape[0]



        # get select partition of memory
        if self.stateProcessor is not None:
            state, _ = self.stateProcessor(transitions.state, self.device)
            nonFinalNextState, nonFinalMask = self.stateProcessor(transitions.next_state, self.device)
        else:
#            state = torch.tensor(transitions.state, device=self.device, dtype=torch.float32)
            nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, transitions.next_state)), device=self.device,
                                        dtype=torch.uint8)
            nonFinalNextState = torch.tensor([s for s in transitions.next_state if s is not None], device=self.device,
                                             dtype=torch.float32)
        QNext = torch.zeros(batchSize, device=self.device, dtype=torch.float32)
        QNext[nonFinalMask] = self.net(nonFinalNextState).max(1)[0].detach()

        offSet = accumLength[episodeStart] - self.episodeLengths[episodeStart]
        for episode in reversed(range(episodeStart, episodeEnd)):
            startIdx = accumLength[episode] - self.episodeLengths[episode]
            endIdx = accumLength[episode]
            # for each episode, we update the lambda return
            if self.memory[endIdx - 1].next_state is None:
                self.memory[idx].process_reward = self.memory[idx].reward
            else:
                self.memory[idx].process_reward = self.memory[idx].reward + self.gamma*QNext[endIdx - 1 - offSet]

            for idx in reversed(range(startIdx, endIdx - 1)):
                self.memory[idx].process_reward = self.memory[idx].reward
                + self.gamma*(self.lamb*self.memory[idx + 1].process_return
                + (1 - self.lamb) * QNext[idx - offSet])


    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return str(self.memory)