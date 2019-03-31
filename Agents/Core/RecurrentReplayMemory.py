import random
import numpy as np
from Agents.Core.ReplayMemory import Transition

# the implementation is from https://github.com/yangyutu/DeepRL-Tutorials/blob/master/11.DRQN.ipynb



class RecurrentReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x - self.seq_length for x in finish]
        samples = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            # final is a list
            final = self.memory[max(start + 1, 0):end + 1]

            # correct for sampling across episodes
            # remove experiences belongs to previous episode
            for i in range(len(final) - 2, -1, -1):
                if final[i][3] is None:
                    final = final[i + 1:]
                    break

            # pad beginning to for sequence that end earlier
            while (len(final) < self.seq_length):
                dummyTransition = Transition(np.zeros_like(self.memory[0][0]), 0, np.zeros_like(self.memory[0][2]), 0)
                final = [dummyTransition] + final

            samples += final

        # returns flattened version
        return samples

    def __len__(self):
        return len(self.memory)