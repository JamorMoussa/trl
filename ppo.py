import torch, torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import gymnasium as gym

import numpy as np

from tensordict import TensorDict


# class ActorCriticNetwork(nn.Module):
#   def __init__(self, obs_space_size, action_space_size):
#     super().__init__()

#     self.shared_layers = nn.Sequential(
#         nn.Linear(obs_space_size, 64),
#         nn.ReLU(),
#         nn.Linear(64, 64),
#         nn.ReLU())
    
#     self.policy_layers = nn.Sequential(
#         nn.Linear(64, 64),
#         nn.ReLU(),
#         nn.Linear(64, action_space_size))
    
#     self.value_layers = nn.Sequential(
#         nn.Linear(64, 64),
#         nn.ReLU(),
#         nn.Linear(64, 1))
    
#   def value(self, obs):
#     z = self.shared_layers(obs)
#     value = self.value_layers(z)
#     return value
        
#   def policy(self, obs):
#     z = self.shared_layers(obs)
#     policy_logits = self.policy_layers(z)
#     return policy_logits

#   def forward(self, obs):
#     z = self.shared_layers(obs)
#     policy_logits = self.policy_layers(z)
#     value = self.value_layers(z)
#     return policy_logits, value
  

# class PPOTrainer():
#   def __init__(self,
#               actor_critic,
#               ppo_clip_val=0.2,
#               target_kl_div=0.01,
#               max_policy_train_iters=80,
#               value_train_iters=80,
#               policy_lr=3e-4,
#               value_lr=1e-2):
    
#     self.ac = actor_critic
#     self.ppo_clip_val = ppo_clip_val
#     self.target_kl_div = target_kl_div
#     self.max_policy_train_iters = max_policy_train_iters
#     self.value_train_iters = value_train_iters

#     policy_params = list(self.ac.shared_layers.parameters()) + \
#         list(self.ac.policy_layers.parameters())
#     self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

#     value_params = list(self.ac.shared_layers.parameters()) + \
#         list(self.ac.value_layers.parameters())
#     self.value_optim = optim.Adam(value_params, lr=value_lr)

#   def train_policy(self, obs, acts, old_log_probs, gaes):
#     for _ in range(self.max_policy_train_iters):
#       self.policy_optim.zero_grad()

#       new_logits = self.ac.policy(obs)
#       new_logits = Categorical(logits=new_logits)
#       new_log_probs = new_logits.log_prob(acts)

#       policy_ratio = torch.exp(new_log_probs - old_log_probs)
#       clipped_ratio = policy_ratio.clamp(
#           1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
#       clipped_loss = clipped_ratio * gaes
#       full_loss = policy_ratio * gaes
#       policy_loss = -torch.min(full_loss, clipped_loss).mean()

#       policy_loss.backward()
#       self.policy_optim.step()

#       kl_div = (old_log_probs - new_log_probs).mean()
#       if kl_div >= self.target_kl_div:
#         break

#   def train_value(self, obs, returns):
#     for _ in range(self.value_train_iters):
#       self.value_optim.zero_grad()

#       values = self.ac.value(obs)
#       value_loss = (returns - values) ** 2
#       value_loss = value_loss.mean()

#       value_loss.backward()
#       self.value_optim.step()

# Policy and value model



class ClipPPOLoss(torch.nn.modules.loss._Loss):

    def __init__(
        self, ppo_clip_val: float = 0.2
    ) -> None:
        super().__init__()

        self.ppo_clip_val = ppo_clip_val

    def loss_clipped(self, new_probs, old_probs, gaes):

        policy_ratio = torch.exp(new_probs - old_probs)
        clipped_ratio = policy_ratio.clamp(
            1 - self.ppo_clip_val, 1 + self.ppo_clip_val
        )
        clipped_loss = clipped_ratio * gaes
        full_loss = policy_ratio * gaes
        return - torch.min(full_loss, clipped_loss).mean()

    def forward(self, new_probs, old_probs, gaes):
        return self.loss_clipped(
            new_probs= new_probs, old_probs= old_probs, gaes= gaes
        )


def discount_rewards(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    discounted_rewards = torch.zeros_like(rewards)
    cumulative_reward = 0.0
    
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    
    return discounted_rewards


def calculate_gaes(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, decay: float = 0.98) -> torch.Tensor:
    
    next_values = torch.cat([values[1:], torch.zeros(1, dtype=values.dtype, device=values.device)])
    
    deltas = rewards + gamma * next_values - values
    
    gaes = torch.zeros_like(deltas)
    gae = 0.0
    for i in reversed(range(len(deltas))):
        gae = deltas[i] + gamma * decay * gae
        gaes[i] = gae
    
    return gaes


class PPODataLoader:

    def __init__(
        self, model, env, max_steps=1000
    ):
       
       self.model = model 
       self.env = env 
       self.max_steps = max_steps

    def __iter__(self):
       return self 
    
    def __next__(self) -> tuple[TensorDict, float]:
       return self.rollout()

    def rollout(self):
        
        train_data = []
        
        obs, _ = self.env.reset()

        ep_reward = 0

        for _ in range(self.max_steps):
            logits, val = self.model(torch.tensor(np.asarray([obs]), dtype=torch.float32))
            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            old_log_probs = act_distribution.log_prob(act).item()

            act, val = act.item(), val.item()

            next_obs, reward, done, _, _ = self.env.step(act)

            train_data.append(
                TensorDict({
                    "obs": torch.from_numpy(obs),
                    "act": torch.tensor([act]).long(),
                    "reward": torch.tensor([reward]).float(),
                    "val": torch.tensor([val]).float(),
                    "done": torch.tensor([done]).bool(),
                    "old_log_probs": torch.tensor([old_log_probs]).float()
                }).unsqueeze(0)
            )

            obs = next_obs
            ep_reward += reward
            if done:
                break

        data = torch.cat(train_data, dim=0)

        data.set("gaes",
            calculate_gaes(data["reward"].flatten(), data["val"].flatten()).float()
        )

        data.set(
           "returns",
            discount_rewards(data["reward"]).float()  
        )

        permute_idxs = np.random.permutation(data.batch_size[0])

        data = data.apply(lambda x: x.squeeze()[permute_idxs])

        return data, ep_reward


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
  

class PPOTrainer:

  def __init__(
    self,
    target_kl_div = 0.02,
    max_policy_train_iters = 40,
    value_train_iters = 40,
    gamma = 0.99,
  ):
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters
    self.gamma = gamma

    self.model = None
    self.env = None
    self.loss_fn = None
    self.actor_optim = None
    self.critic_optim = None

  def discount_rewards(self, rewards: torch.Tensor):

      rewards = rewards.flatten().tolist()

      new_rewards = [float(rewards[-1])]
      for i in reversed(range(len(rewards)-1)):
          new_rewards.append(float(rewards[i]) + self.gamma * new_rewards[-1])
      return torch.Tensor(new_rewards[::-1]).unsqueeze(-1)

  def compile(
      self, model,
      ppo_loss: ClipPPOLoss,
      value_loss: nn.MSELoss,
      actor_lr = 3e-4,
      critic_lr = 1e-3
  ):
      self.model = model
      self.ppo_loss = ppo_loss
      self.value_loss= value_loss

      self.actor_optim = optim.Adam(
        params= [*self.model.shared_module.parameters(), *self.model.actor_module.parameters()], lr=actor_lr
      )

      self.critic_optim = optim.Adam(
        params= [*self.model.shared_module.parameters(), *self.model.critic_module.parameters()], lr=critic_lr
      )

  def train_actor(
      self, data: TensorDict
  ):
      total_loss = 0

      for _ in range(self.max_policy_train_iters):
          self.actor_optim.zero_grad()

          new_logits = self.model.actor(data["obs"])
          new_logits = Categorical(logits=new_logits)
          new_log_probs = new_logits.log_prob(data["act"])

          loss = self.ppo_loss(
              new_probs= new_log_probs, old_probs= data["old_log_probs"], gaes= data["gaes"]
          )

          loss.backward()

          self.actor_optim.step()

          kl_div = (data["old_log_probs"] - new_log_probs.detach()).mean()
          if kl_div >= self.target_kl_div:
            break

      return total_loss

  def train_value(self, data: TensorDict):

      returns = data["returns"]

      for _ in range(self.value_train_iters):
        self.critic_optim.zero_grad()

        values = self.model.critic(data["obs"])

        value_loss = self.value_loss(
            returns, values.squeeze()
        )

        value_loss.backward()
        self.critic_optim.step()

      return value_loss.item(), returns

  def train(
    self, ppo_loader: PPODataLoader, max_iters = 100, verbose_iter= 10
  ):
      itr = 0

      # losses = []
      rewards = []

      for (data, esp_return)  in ppo_loader:

          loss = self.train_actor(data=data)

          val_loss, _ = self.train_value(data=data)

          rewards.append(esp_return)

          itr += 1

          if itr % verbose_iter == 0:
              print(f"Epoch {itr} | Reward: {torch.Tensor(rewards).mean().item()}")

          if itr > max_iters:
              break
