import torch 
from torch.nn.modules.loss import _Loss


__all__ = ["ClipPPOLoss"]


class ClipPPOLoss(_Loss):

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

    def forward(self, new_probs, old_probs, gaes) -> torch.Tensor:
        return self.loss_clipped(
            new_probs= new_probs, old_probs= old_probs, gaes= gaes
        )
