import torch 
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn as nn



class Actor(nn.Module):

    def __init__(self, input_size: int = 8, n_actions: int= 4):
        super(Actor, self).__init__()

        self.fc = nn.Linear(input_size, 16)
        self.mu = nn.Linear(16, n_actions)

        self.log_std = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor):

        x = self.fc(x)

        mu = self.mu(x)

        return mu, self.log_std.exp()


model = Actor()

H = 4

x = torch.rand(1, 8)

mu, std = model(x)

dist = Normal(mu, std)

print(dist)

act = dist.rsample()

print(act)

print(dist.log_prob(act))

#########################################

c_dist = Categorical(logits=mu)

c_act = c_dist.sample()

print(c_act)

print(c_dist.log_prob(c_act))