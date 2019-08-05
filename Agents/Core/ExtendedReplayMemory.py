from collections import namedtuple
# Define a namedtuple with name Transition and attributes of state, action, next_state, reward
ExtendedTransition = namedtuple('ExtendedTransition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExtendedReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(transition)

        # write on the earlier experience
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def fetch_all(self):
        return self.memory

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return str(self.memory)