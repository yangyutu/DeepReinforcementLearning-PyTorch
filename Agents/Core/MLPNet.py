import torch
import torch.nn.functional as F


class MultiLayerNetRegression(torch.nn.Module):
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

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.predict(x)
        return x
