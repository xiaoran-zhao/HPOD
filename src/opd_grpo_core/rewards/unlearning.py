from dataclasses import dataclass

import torch


@dataclass
class UnlearningRewardResult:
    rewards: torch.Tensor
    unlearning_quality: torch.Tensor
    format_valid: torch.Tensor


def build_unlearning_rewards(
    unlearning_quality_scores: torch.Tensor,
    format_validity: torch.Tensor,
) -> UnlearningRewardResult:
    """Apply the HPOD reward: S_UQ when R_FV=1, otherwise zero."""
    if unlearning_quality_scores.numel() != format_validity.numel():
        raise ValueError("S_UQ and R_FV must contain the same number of rollouts.")
    quality = unlearning_quality_scores.detach().float().reshape(-1).clamp(1.0, 5.0)
    valid = format_validity.detach().reshape(-1).to(dtype=torch.bool)
    rewards = torch.where(valid, quality, torch.zeros_like(quality))
    return UnlearningRewardResult(rewards, quality, valid)
