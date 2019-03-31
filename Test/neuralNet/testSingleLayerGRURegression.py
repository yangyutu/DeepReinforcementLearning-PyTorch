from Agents.Core.MLPNet import SingleGRULayerNetRegression

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x1 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
x2 = x1 - torch.rand_like(x1)

x = torch.cat([x1, x2], dim=1).unsqueeze(-1)
y = x2 - x1

net = SingleGRULayerNetRegression(n_feature=1, n_output=1, gru_size=32)  # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()  # something about plotting

for t in range(100):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
    print(loss)
    # grad clip is import to avoid explosion
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    print("loss at step" + str(t) + " is:" + str(loss.data))

    # if t % 50 == 0:
    #     # plot and show learning process
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
    #     plt.pause(0.1)

plt.ioff()
plt.show()
print(prediction/y - 1)
#print(y)