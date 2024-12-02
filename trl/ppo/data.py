import torch
from torch.distributions.categorical import Categorical

from tensordict import TensorDict
import numpy as np 

from .utils import discount_rewards, calculate_gaes


__all__ = ["PPODataLoader", ]


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