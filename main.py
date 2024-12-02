import gymnasium as gym 
import torch, torch.nn as nn

from models import Actor, Critic, ActorCriticNetwork

import trl 

DEVICE = "cpu"

# env = gym.make('CartPole-v1')

env = gym.make("CartPole-v1", render_mode="rgb_array")

# model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
# train_data, reward = rollout(model, env) # Test rollout function

hidden_size = 16

model = ActorCriticNetwork(
    base=nn.Sequential(
        nn.Linear(env.observation_space.shape[0], hidden_size)
    ),
    actor=Actor(hidden_size, env.action_space.n),
    critic=Critic(hidden_size)
)


ppo_loader = trl.ppo.PPODataLoader(
    model=model,
    env= trl.env.TrlEnv(env=env), 
    max_steps=5000
)


ppo_trainer = trl.ppo.PPOTrainer(
    max_policy_train_iters = 40,
    value_train_iters = 40
)


ppo_trainer.compile(
    model= model,
    ppo_loss= trl.ppo.ClipPPOLoss(ppo_clip_val=0.2),
    value_loss= nn.MSELoss(),
    actor_lr = 2e-4,
    critic_lr = 1e-3
)

ppo_trainer.train(
    ppo_loader= ppo_loader, max_iters=300, verbose_iter=5
)

torch.save(model.state_dict(), "./save/model.pt")
