import torch
from torch.distributions.categorical import Categorical

from tensordict import TensorDict
import numpy as np 

from .utils import discount_rewards, calculate_gaes
from ..env import TrlEnv

__all__ = ["PPODataLoader", ]


class PPODataLoader:

    def __init__(
        self, model, env: TrlEnv, max_steps=1000
    ):
        if not isinstance(env, TrlEnv):
            raise ValueError("'env' must 'TrlEnv' type.")

        self.model = model 
        self.env = env 
        self.max_steps = max_steps

    def __iter__(self):
       return self 
    
    @torch.no_grad()
    def __next__(self) -> tuple[TensorDict, float]:
       return self.rollout()

    def rollout(self):
        
        train_data = []
        
        obs = self.env.reset()

        ep_reward = 0

        for _ in range(self.max_steps):

            # logits, val = self.model(torch.tensor(np.asarray([obs]), dtype=torch.float32))
            # act_distribution = Categorical(logits=logits)
            # act = act_distribution.sample()
            # old_log_probs = act_distribution.log_prob(act).item()

            old_log_probs, act , val = self.model.evaluate(obs=obs)

            # act, val = act.item(), val.item()

            next_obs, reward, act, done = self.env.step(act)

            train_data.append(
                TensorDict({
                    "obs": obs, "act": act, "reward": reward, "val": val,
                    "done": done, "old_log_probs": old_log_probs
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