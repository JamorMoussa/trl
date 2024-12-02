import gymnasium as gym
import torch 

class TensorDictGymEnv:

    def __init__(
        self, env: gym.Env
    ):
        self.env= env 

        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        state = self.env.reset()

        return {"obs": torch.from_numpy(state[0])}

    def step(self, action: int | torch.Tensor):

        if isinstance(action, int):
            action = torch.tensor([action]).long()

        if not isinstance(action, torch.Tensor):
            raise ValueError("'action' must be a 'Tensor' or 'int'")
        
        step_info = self.env.step(action=action.item())

        return {
            "obs": torch.from_numpy(step_info[0]),
            "act": action, 
            "reward": torch.tensor([step_info[1]]).float(), 
            "done": torch.tensor([step_info[2]]).bool(),
        }

        

