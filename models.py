import torch 
import torch.nn as nn

from torch.distributions.categorical import Categorical


class Actor(nn.Module):

    def __init__(self, hidden_size: int, n_actions: int, is_continous: bool = False):
        super(Actor, self).__init__()

        self.hidden_size= hidden_size
        self.n_actions= n_actions
        self.is_continous= is_continous

        self.ac = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_size, self.n_actions)

        if is_continous:
            self.log_std = nn.Parameter(torch.zeros(1))


    def forward(self, prv_base_obv: torch.Tensor, act: torch.Tensor= None):

        x = self.ac(prv_base_obv)
        mu = self.mu(x)

        if not self.is_continous:
            dist = Categorical(logits=mu)
            action = act if act is not None else dist.sample()

            log_probs = dist.log_prob(action)
            
            # log_probs = torch.tensor([
            #     dist.log_prob(action)
            # ]).float()

            # print(log_probs)

        return log_probs, action
    

class Critic(nn.Module):

    def __init__(self, hidden_size: int):
        super(Critic, self).__init__()

        self.cr = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prv_base_obv: torch.Tensor):
        return self.cr(prv_base_obv)


class ActorCriticNetwork(nn.Module):

    def __init__(
        self, base: nn.Module, actor: Actor, critic: Critic
    ):
        super(ActorCriticNetwork, self).__init__()

        self.base = base 
        self.actor = actor
        self.critic = critic

    def eval_actor(self, obs: torch.Tensor, act: torch.Tensor= None):
        x = self.base(obs)
        return self.actor(x, act=act)

    def eval_critic(self, obs: torch.Tensor):
        x = self.base(obs)
        return self.critic(x)

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor=None):
    
        x = self.base(obs)
        log_probs, action = self.actor(x, act=act)
        val = self.critic(x)

        return log_probs, action, val

