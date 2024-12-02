import torch
from torch.distributions.categorical import Categorical

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from ppo import ActorCriticNetwork


env = gym.make('Acrobot-v1', render_mode="rgb_array")

model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
# train_data, reward = rollout(model, env) # Test rollout function


model.load_state_dict(torch.load("./save/model.pt", weights_only=True))


env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
# env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(3):
    obs, info = env.reset()

    episode_over = False

    iiter = 0
    while not episode_over:
        log_probs, _ = model(torch.tensor(obs).float())

        # action = log_probs.argmax().item()
        action = Categorical(logits=log_probs).sample().item()

        # action, _states = model.predict(obs)
        
        obs, rewards, dones, info, _ = env.step(action)

        # obs, reward, terminated, truncated, info = env.step(action)

        episode_over = dones

        iiter += 1

        print(iiter)

        if iiter > 1000: break

        # env.render()
env.close()
