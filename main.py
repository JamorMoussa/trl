from env import TensorDictGymEnv
import gymnasium as gym 
import torch, torch.nn as nn

from ppo import ActorCriticNetwork, PPODataLoader, PPOTrainer, discount_rewards, ClipPPOLoss
import numpy as np 

from torch.distributions.categorical import Categorical
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

DEVICE = "cpu"

# env = gym.make('CartPole-v1')

env = gym.make('Acrobot-v1', render_mode="rgb_array")

model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
# train_data, reward = rollout(model, env) # Test rollout function

ppo_loader = PPODataLoader(
    model=model,
    env=env, max_steps=1000
)

ppo_trainer = PPOTrainer(
    max_policy_train_iters = 40,
    value_train_iters = 40
)

ppo_trainer.compile(
    model= model,
    ppo_loss= ClipPPOLoss(ppo_clip_val=0.2),
    value_loss= nn.MSELoss(),
    actor_lr = 1e-4,
    critic_lr = 1e-3
)

ppo_trainer.train(
    ppo_loader= ppo_loader, max_iters=300, verbose_iter=5
)

torch.save(model.state_dict(), "./save/model.pt")
