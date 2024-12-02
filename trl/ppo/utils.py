import torch 


__all__ = ["discount_rewards", "calculate_gaes"]


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
