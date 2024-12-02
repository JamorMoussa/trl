import torch, torch.nn as nn


class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size):
    super().__init__()

    self.shared_module = nn.Sequential(
        nn.Linear(obs_space_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU()
    )

    self.actor_module = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_space_size)
    )

    self.critic_module = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

  def critic(self, obs):
    z = self.shared_module(obs)
    value = self.critic_module(z)
    return value

  def actor(self, obs):
    z = self.shared_module(obs)
    policy_logits = self.actor_module(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_module(obs)
    policy_logits = self.actor_module(z)
    value = self.critic_module(z)
    return policy_logits, value
