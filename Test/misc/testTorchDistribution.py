import torch

import torch.distributions



normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))


# get a sample from normal distribution
print(normal.sample())


# get the log_prob at a value
print(normal.log_prob(1.0))


# get the entropy of the distribution
# for normal distribution, entroy has analytical expression and only depends on variance (i.e., not related to mean)
# 0.5 + 0.5 ln(2pi) + ln(\sigma)
print(normal.entropy())




categorical = torch.distributions.Categorical(probs = torch.tensor([0.15, 0.35, 0.25, 0.25]))

categorical = torch.distributions.Categorical(logits = torch.tensor([-3.0, -1, 3, 2]))

# get a sample from normal distribution
print(categorical.sample())


# get the log_prob at a value
print(categorical.log_prob(torch.tensor(3, dtype=torch.long)))


# get the entropy of the distribution
# for normal distribution, entroy has analytical expression and only depends on variance (i.e., not related to mean)
print(categorical.entropy())