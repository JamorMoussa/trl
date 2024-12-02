import torch, torch.nn as nn
import torch.optim as optim 
from torch.distributions.categorical import Categorical

from tensordict import TensorDict


from .data import PPODataLoader
from .loss import ClipPPOLoss


__all__ = ["PPOTrainer", ]


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
        params= [*self.model.base.parameters(), *self.model.actor.parameters()], lr=actor_lr
      )

      self.critic_optim = optim.Adam(
        params= [*self.model.base.parameters(), *self.model.critic.parameters()], lr=critic_lr
      )

  def train_actor(
      self, data: TensorDict
  ):
      total_loss = 0

      for _ in range(self.max_policy_train_iters):
          self.actor_optim.zero_grad()

          # new_logits = self.model.actor(data["obs"])
          # new_logits = Categorical(logits=new_logits)
          # new_log_probs = new_logits.log_prob(data["act"])

          new_log_probs, _ = self.model.eval_actor(obs= data["obs"], act= data["act"])

          

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

        values = self.model.eval_critic(data["obs"])

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