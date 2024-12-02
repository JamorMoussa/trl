import gymnasium as gym 
import numpy as np
import torch 

class TrlEnv:

    def __init__(
        self, env: gym.Env, is_continous: bool= False
    ):
        self.is_continous = is_continous
        
        self.env = env
        self.action_space = self.env.action_space
        self.action_space.observation_space = self.env.observation_space

    def reset(self):
        obs, _ = self.env.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

        return obs 
    

    def step(self, action: torch.Tensor):

        if not isinstance(action, torch.Tensor):
            raise ValueError("'action' is must be 'Tensor'")
        
        if not self.is_continous:
            action = action.long().item()
        else: 
            action = action.detach().cpu().numpy()

        step_return = self.env.step(action=action)

        obs = torch.from_numpy(step_return[0]).float()

        act = torch.from_numpy(action).float() if self.is_continous else torch.tensor([action]).long()

        reward = torch.tensor([step_return[1]]).float()

        done = torch.tensor([step_return[2]]).bool()

        return obs, reward, act, done 


