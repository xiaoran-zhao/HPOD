from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GRPOResult:
    loss: torch.Tensor
    valid_tokens: int
    ratio_mean: float
    zero_variance_group_rate: float
    effective_adv_abs_max: float
    reference_kl: float


def sampled_reference_kl(
    current_token_logps: torch.Tensor,
    reference_token_logps: torch.Tensor,
) -> torch.Tensor:
    """Compute the nonnegative sampled KL estimator used by GRPO."""
    if current_token_logps.shape != reference_token_logps.shape:
        raise ValueError("Current and reference token log-probabilities must match.")
    ref_log_ratio = reference_token_logps.detach().to(current_token_logps) - current_token_logps
    safe_log_ratio = torch.clamp(ref_log_ratio, min=-20.0, max=20.0)
    return torch.exp(safe_log_ratio) - safe_log_ratio - 1.0


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int = 8,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Normalize rewards independently inside each rollout group."""
    flat = rewards.detach().float().reshape(-1)
    if group_size <= 1 or flat.numel() % group_size:
        raise ValueError("Reward count must be divisible by group_size > 1.")
    groups = flat.view(-1, group_size)
    means = groups.mean(dim=-1, keepdim=True)
    stds = groups.std(dim=-1, unbiased=False, keepdim=True)
    nonzero = stds > float(eps)
    advantages = torch.where(nonzero, (groups - means) / stds.clamp_min(eps), torch.zeros_like(groups))
    zero_rate = float((~nonzero).float().mean().item())
    return advantages.reshape_as(flat).detach(), zero_rate


def token_level_grpo_loss(
    current_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float = 0.2,
    zero_variance_group_rate: float = 0.0,
    reference_kl: Optional[torch.Tensor] = None,
    reference_kl_beta: float = 0.0,
) -> GRPOResult:
    """Compute the globally token-mean PPO-style GRPO policy objective."""
    if current_token_logps.shape != old_token_logps.shape:
        raise ValueError("Current and old token log-probabilities must match.")
    if current_token_logps.shape != response_mask.shape:
        raise ValueError("response_mask must match token log-probability shape.")
    if advantages.numel() != current_token_logps.size(0):
        raise ValueError("One scalar advantage is required per trajectory.")

    mask = response_mask.to(device=current_token_logps.device, dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        zero = current_token_logps.sum() * 0.0
        return GRPOResult(zero, 0, 1.0, zero_variance_group_rate, 0.0, 0.0)

    log_ratio = current_token_logps - old_token_logps.detach()
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    token_adv = advantages.to(current_token_logps).view(-1, 1)
    token_loss = torch.maximum(-token_adv * ratio, -token_adv * clipped)
    reference_kl_mean = current_token_logps.detach().new_tensor(0.0)
    if float(reference_kl_beta) != 0.0:
        if reference_kl is None:
            raise ValueError("reference_kl is required when reference_kl_beta is nonzero.")
        if reference_kl.shape != current_token_logps.shape:
            raise ValueError("reference_kl must match token log-probability shape.")
        reference_kl_values = reference_kl.to(
            device=current_token_logps.device,
            dtype=current_token_logps.dtype,
        )
        token_loss = token_loss + float(reference_kl_beta) * reference_kl_values
        reference_kl_mean = reference_kl_values[mask].detach().mean()
    selected_ratio = ratio[mask].detach()
    return GRPOResult(
        loss=token_loss[mask].mean(),
        valid_tokens=count,
        ratio_mean=float(selected_ratio.mean().item()),
        zero_variance_group_rate=float(zero_variance_group_rate),
        effective_adv_abs_max=float(advantages.detach().abs().max().item()),
        reference_kl=float(reference_kl_mean.item()),
    )


def microbatched_token_level_grpo_loss(
    current_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    micro_batch_size: int,
    clip_range: float = 0.2,
    zero_variance_group_rate: float = 0.0,
    reference_kl: Optional[torch.Tensor] = None,
    reference_kl_beta: float = 0.0,
) -> GRPOResult:
    """Evaluate GRPO in trajectory microbatches with global token reduction."""
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive.")
    trajectory_count = current_token_logps.size(0)
    weighted_loss = current_token_logps.sum() * 0.0
    total_tokens = 0
    ratio_total = 0.0
    reference_kl_total = 0.0
    for start in range(0, trajectory_count, micro_batch_size):
        end = min(trajectory_count, start + micro_batch_size)
        result = token_level_grpo_loss(
            current_token_logps[start:end],
            old_token_logps[start:end],
            response_mask[start:end],
            advantages[start:end],
            clip_range=clip_range,
            zero_variance_group_rate=zero_variance_group_rate,
            reference_kl=None if reference_kl is None else reference_kl[start:end],
            reference_kl_beta=reference_kl_beta,
        )
        if result.valid_tokens <= 0:
            continue
        weighted_loss = weighted_loss + result.loss * result.valid_tokens
        total_tokens += result.valid_tokens
        ratio_total += result.ratio_mean * result.valid_tokens
        reference_kl_total += result.reference_kl * result.valid_tokens
    if total_tokens <= 0:
        return GRPOResult(weighted_loss, 0, 1.0, zero_variance_group_rate, 0.0, 0.0)
    return GRPOResult(
        loss=weighted_loss / total_tokens,
        valid_tokens=total_tokens,
        ratio_mean=ratio_total / total_tokens,
        zero_variance_group_rate=float(zero_variance_group_rate),
        effective_adv_abs_max=float(advantages.detach().abs().max().item()),
        reference_kl=reference_kl_total / total_tokens,
    )
