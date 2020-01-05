import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.netInit import xavier_init

class DuelingMLP(nn.Module):
    """An example dueling network for Q learning
        see 'Dueling Network Architectures for Deep Reinforcement Learning'
        # Argument
        n_feature: dimensionality of input feature
        n_hidden: a list of hidden units
        n_output: dimensionality of output feature
        dueling_size: the hidden unit for advantage function and value function
        """


    def __init__(self, n_feature, n_hidden, n_output, dueling_size):
        super(DuelingMLP, self).__init__()
        if len(n_hidden) < 1:
            raise ValueError("number of hidden layer should be greater than 0")
        # here we need to use nn.ModuleList in order to build a list of layers.
        # we cannot use ordinary list
        self.layers = torch.nn.ModuleList()
        for idx, hidUnits in enumerate(n_hidden):
            if idx == 0:
                hidLayer = torch.nn.Linear(n_feature, n_hidden[0])
            else:
                hidLayer = torch.nn.Linear(n_hidden[idx-1], hidUnits)
            self.layers.append(hidLayer)

        # advantage layer
        self.advLayer1 = nn.Linear(n_hidden[-1], dueling_size)
        self.advLayer2 = nn.Linear(dueling_size, n_output)

        # value layer
        self.valLayer1 = nn.Linear(n_hidden[-1], dueling_size)
        self.valLayer2 = nn.Linear(dueling_size, 1)

        self.apply(xavier_init)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        adv = F.relu(self.advLayer1(x))
        adv = self.advLayer2(adv)

        val = F.relu(self.valLayer1(x))
        val = self.valLayer2(val)
        # here the subtraction of mean is to address the identification issue. See the original reference.
        return val + adv - adv.mean()
