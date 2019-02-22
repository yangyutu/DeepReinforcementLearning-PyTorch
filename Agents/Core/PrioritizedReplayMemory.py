"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
from Agents.Core.ReplayMemory import Transition
np.random.seed(1)


# source code from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
# other implementation see https://github.com/yangyutu/DeepRL-Tutorials/blob/master/06.DQN_PriorityReplay.ipynb

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """


    def __init__(self, capacity):
        self.data_pointer = 0
        self.data_count = 0
        self.capacity = capacity  # for all priority values
        # represent the tree by an array
        # for a complete binary tree with n layers, there are total 2^(n+1) - 1 nodes with 2^n leaves
        # for array representation of binary trees, see https://www.geeksforgeeks.org/binary-tree-array-implementation/
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

        # p is priority and data is the transition data
    def add(self, p, data):
        # note that leaves starting from index of self.capacity - 1
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
        self.data_count += 1.0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # here v is the priority value
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PrioritizedReplayMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, config,  epsilon = 0.01, alpha = 0.6, beta = 0.4, beta_increment_per_sampling = 0.001, abs_err_upper = 1.):
        self.tree = SumTree(capacity)

        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.0  # clipped abs error

        if 'priorityMemory_epsilon' in config:
            self.epsilon = config['priorityMemory_epsilon']
        if 'priorityMemory_alpha' in config:
            self.alpha = config['priorityMemory_alpha']
        if 'priorityMemory_beta' in config:
            self.beta = config['priorityMemory_beta']
        if 'priorityMemory_betaIncrement' in config:
            self.beta_increment_per_sampling = config['priorityMemory_betaIncrement']
        if 'priorityMemory_absErrUpper' in config:
            self.abs_err_upper = config['priorityMemory_absErrUpper']


    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # get the maximum priority in the memory
        if max_p == 0:
            max_p = self.abs_err_upper
        # set the max p for new p such that new experiences have higher probability to be selected
        self.tree.add(max_p, transition)


    # sample n transitions
    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = []
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment

        # increase beta after every sampling
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
           # b_memory stores the list of transitions
        return  b_memory, b_idx, ISWeights


    # after training on these samples, we need to update the priority of these samples
    # this is because some samples will become of less priority
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return int(self.tree.data_count)