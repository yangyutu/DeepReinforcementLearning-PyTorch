import torch
import torch.nn.functional as F
from utils.netInit import xavier_init
import numpy as np

class MultiLayerNetRegression(torch.nn.Module):
    """An example MLP network for Q learning. Such network can be used in a typical deep Q learning
        # Argument
        n_feature: dimensionality of input feature
        n_hidden: a list of hidden units
        n_output: dimensionality of output feature
        """
    def __init__(self, n_feature, n_hidden, n_output):
        super(MultiLayerNetRegression, self).__init__()
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
        self.predict = torch.nn.Linear(n_hidden[-1], n_output)

        self.apply(xavier_init)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        out = self.predict(x)
        return out

class MultiLayerNetLogSoftmax(torch.nn.Module):
    """An example MLP network for Q learning. Note that log softmax will apply to the last layer.
        Such network can be used as an actor network in a typical actor-critic learning
        # Argument
        n_feature: dimensionality of input feature
        n_hidden: a list of hidden units
        n_output: dimensionality of output feature
        """
    def __init__(self, n_feature, n_hidden, n_output):
        super(MultiLayerNetLogSoftmax, self).__init__()
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
        self.predict = torch.nn.Linear(n_hidden[-1], n_output)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        # each row will sum to 1
        out = F.log_softmax(self.predict(x), dim=1)
        return out

class MultiLayerNetActorCritic(torch.nn.Module):
    """An example MLP network for actor-critic learning. Note that the network outputs both action and value
        # Argument
        n_feature: dimensionality of input feature
        n_hidden: a list of hidden units
        n_output: dimensionality of output feature
        """
    def __init__(self, n_feature, n_hidden, n_output):
        super(MultiLayerNetActorCritic, self).__init__()
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


        self.actor = torch.nn.Linear(n_hidden[-1], n_output)
        self.critic = torch.nn.Linear(n_hidden[-1], 1)
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        # each row will sum to 1
        actorOut = F.log_softmax(self.actor(x), dim=1)
        criticOut = self.critic(x)
        return (actorOut, criticOut)


class MultiLayerNetRegressionWithGRU(torch.nn.Module):
    """An example MLP network for deep Q learning for recurrent units. Note that the network outputs both action and value
        # Argument
        n_feature: dimensionality of input feature
        n_hidden: a list of hidden units
        n_output: dimensionality of output feature
        """
    def __init__(self, n_feature, n_hidden, n_output, gru_size=32, device = None):
        super(MultiLayerNetRegressionWithGRU, self).__init__()

        self.n_feature = n_feature
        self.n_output = n_output
        self.gru_size = gru_size

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

        self.gru = torch.nn.GRU(n_hidden[-1], self.gru_size, num_layers=1, batch_first=True,
                          bidirectional=False)
        self.prediction = torch.nn.Linear(self.gru_size, 128)

        self.out = torch.nn.Linear(128, self.n_output)


        self.device = device if device is not None else 'cpu'

        self.hidden = None

        self.apply(xavier_init)

    def forward(self, x, hx=None):
        # initial input of x is ( batch, seq_len, feature)

        batch_size = x.size(0)
        sequence_length = x.size(1)

        # we want to reshape to (batch_size*sequence_length, n_features)
        x = x.view(-1, self.n_feature)

        for layer in self.layers:
            x = F.relu(layer(x))

        # now we reshape to ( batch, seq_length, hidden[-1])
        x = x.view(batch_size, sequence_length, -1)

        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(x, hidden)

        x = self.out(F.relu(self.prediction(hidden.squeeze())))

        self.hidden = hidden

        return x

    def get_net_state(self):
        return self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_size, device=self.device, dtype=torch.float)

    def get_zero_input(self):
        return np.zeros(self.n_feature)

class SingleGRULayerNetRegression(torch.nn.Module):
    def __init__(self, n_feature, n_output, gru_size=32, device = None):
        super(SingleGRULayerNetRegression, self).__init__()

        self.n_feature = n_feature
        self.n_output = n_output
        self.gru_size = gru_size


        self.gru = torch.nn.GRU(self.n_feature , self.gru_size, num_layers=1, batch_first=True,
                          bidirectional=False)
        self.prediction = torch.nn.Linear(self.gru_size, self.n_output)

        self.device = device if device is not None else 'cpu'

        self.hidden = None

    def forward(self, x, hx=None):
        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(x, hidden)

        x = self.prediction(hidden.squeeze())

        self.hidden = hidden

        return x

    def get_net_state(self):
        return self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_size, device=self.device, dtype=torch.float)

    def get_zero_input(self):
        return np.zeros(self.n_feature)

class MultiLayerNetRegressionWithGRUCell(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, gru_size=512, device=None):
        super(MultiLayerNetRegressionWithGRUCell, self).__init__()

        self.n_feature = n_feature
        self.n_output = n_output
        self.gru_size = gru_size

        if len(n_hidden) < 1:
            raise ValueError("number of hidden layer should be greater than 0")
        # here we need to use nn.ModuleList in order to build a list of layers.
        # we cannot use ordinary list
        self.layers = torch.nn.ModuleList()
        for idx, hidUnits in enumerate(n_hidden):
            if idx == 0:
                hidLayer = torch.nn.Linear(n_feature, n_hidden[0])
            else:
                hidLayer = torch.nn.Linear(n_hidden[idx - 1], hidUnits)
            self.layers.append(hidLayer)

        self.gru = torch.nn.GRUCell(n_hidden[-1], self.gru_size)
        self.prediction = torch.nn.Linear(self.gru_size, self.num_actions)

        self.device = device if device is not None else 'cpu'

    def forward(self, x, hx=None):
        # initial input of x is (batch, feature)

        batch_size = x.size[0]
        for layer in self.layers:
            x = F.relu(layer(x))

        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(x, hidden)

        x = self.prediction(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.gru_size, device=self.device, dtype=torch.float)