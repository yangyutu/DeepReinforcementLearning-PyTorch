import numpy as np


# Ornstein-Ulhenbeck Process
# Adapted from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, seed = 1, mu=0.0, theta=0.5, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        np.random.seed(seed)
        #self.low = action_space.low
        #self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim, dtype=np.float32) * self.mu
        self.t = 0.0

    def get_noise(self):
        self.t += 1.0
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    #def get_action(self, action, t=0):
    #    ou_state = self.evolve_state()
    #    self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
    #    return np.clip(action + ou_state, self.low, self.high)
