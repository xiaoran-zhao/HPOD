from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..config import CoreConfig
from ..controllers import ProgressGradBalance
from ..losses import (
    group_relative_advantages,
    microbatched_token_level_grpo_loss,
    sampled_reference_kl,
    topk_opd_loss,
)
from .batch import StrictOnPolicyBatch
from .step import coordinate_update

GradientTuple = Tuple[Optional[torch.Tensor], ...]


def _raw_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> GradientTuple:
    if not loss.requires_grad:
        raise RuntimeError("Loss is detached from trainable Student parameters.")
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


@dataclass
class StepMetrics:
    iteration: int
    retain_loss: float
    reverse_loss: float
    grpo_loss: float
    retain_grad_norm: float
    reverse_grad_norm: float
    behavior_grad_norm: float
    adaptive_lambda: float
    projection_active: bool
    final_grad_norm: float
    grpo_ratio_mean: float
    grpo_zero_variance_group_rate: float
    effective_adv_abs_max: float
    grpo_reference_kl: float
    optimizer_steps: int
    grpo_policy_epochs: int
    grpo_policy_micro_batch_size: int


class StrictOnPolicyEngine:
    """Execute the complete tensor-level Retain/Reverse OPD plus GRPO update.

    A host model generates all tensors in `StrictOnPolicyBatch` with theta_t.
    This engine consumes the batch once, computes the three raw gradient paths,
    coordinates them, performs one optimizer step, and rejects batch reuse.
    """

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        config: CoreConfig = CoreConfig(),
    ) -> None:
        self.parameters = parameters
        self.optimizer = optimizer
        self.config = config
        if config.strict_on_policy and config.grpo.policy_epochs != 1:
            raise ValueError(
                "Strict on-policy training requires GRPO policy_epochs=1; "
                "additional epochs would reuse trajectories after an update."
            )
        if config.grpo.reference_kl_mode != "sampled":
            raise ValueError("This core project implements reference_kl_mode='sampled'.")
        self.iteration = 1
        self.controller = ProgressGradBalance(
            beta=config.adaptive.ema_beta,
            alpha=config.adaptive.progress_alpha,
            eps=config.adaptive.epsilon,
        )

    def step(self, batch: StrictOnPolicyBatch) -> StepMetrics:
        batch.assert_fresh(self.iteration)
        cfg = self.config
        if batch.retain.student_logits.size(0) != cfg.retain_batch_size:
            raise ValueError(f"Retain batch must contain {cfg.retain_batch_size} samples.")
        if batch.forget.student_logits.size(0) != cfg.forget_batch_size:
            raise ValueError(f"Forget batch must contain {cfg.forget_batch_size} samples.")
        expected_trajectories = cfg.forget_batch_size * cfg.grpo.group_size
        if batch.grpo.rewards.numel() != expected_trajectories:
            raise ValueError(f"GRPO requires {expected_trajectories} trajectories per iteration.")

        retain = topk_opd_loss(
            batch.retain.student_logits,
            batch.retain.teacher_logits,
            batch.retain.response_mask,
            topk=cfg.opd.topk,
            student_temperature=cfg.opd.student_temperature,
            teacher_temperature=cfg.opd.teacher_temperature,
            reverse=False,
            sample_mean=cfg.opd.sample_mean_reduction,
        )
        reverse = topk_opd_loss(
            batch.forget.student_logits,
            batch.forget.teacher_logits,
            batch.forget.response_mask,
            topk=cfg.opd.topk,
            student_temperature=cfg.opd.student_temperature,
            teacher_temperature=cfg.opd.teacher_temperature,
            reverse=True,
            sample_mean=cfg.opd.sample_mean_reduction,
        )
        advantages, zero_variance_rate = group_relative_advantages(
            batch.grpo.rewards,
            group_size=cfg.grpo.group_size,
            eps=cfg.grpo.advantage_epsilon,
        )
        if cfg.grpo.reference_kl_enabled:
            reference_kl = sampled_reference_kl(
                batch.grpo.current_token_logps,
                batch.grpo.reference_token_logps,
            )
            reference_kl_beta = cfg.grpo.reference_kl_beta
        else:
            reference_kl = torch.zeros_like(batch.grpo.current_token_logps)
            reference_kl_beta = 0.0
        grpo = microbatched_token_level_grpo_loss(
            batch.grpo.current_token_logps,
            batch.grpo.old_token_logps,
            batch.grpo.response_mask,
            advantages,
            micro_batch_size=cfg.grpo.policy_micro_batch_size,
            clip_range=cfg.grpo.clip_range,
            zero_variance_group_rate=zero_variance_rate,
            reference_kl=reference_kl,
            reference_kl_beta=reference_kl_beta,
        )

        retain_grads = _raw_gradients(retain.loss, self.parameters, retain_graph=True)
        reverse_grads = _raw_gradients(reverse.loss, self.parameters, retain_graph=True)
        behavior_grads = _raw_gradients(grpo.loss, self.parameters, retain_graph=False)
        coordinated = coordinate_update(
            retain_grads=retain_grads,
            reverse_grads=reverse_grads,
            behavior_grads=behavior_grads,
            forget_kl=float(batch.forget_calibration_kl),
            controller=self.controller,
            parameters=self.parameters,
            optimizer=self.optimizer,
            max_grad_norm=cfg.optimizer.max_grad_norm,
        )
        batch.consume()
        metrics = StepMetrics(
            iteration=self.iteration,
            retain_loss=float(retain.loss.detach().item()),
            reverse_loss=float(reverse.loss.detach().item()),
            grpo_loss=float(grpo.loss.detach().item()),
            retain_grad_norm=coordinated.retain_grad_norm,
            reverse_grad_norm=coordinated.reverse_grad_norm,
            behavior_grad_norm=coordinated.behavior_grad_norm,
            adaptive_lambda=coordinated.adaptive_lambda,
            projection_active=coordinated.projection_active,
            final_grad_norm=coordinated.final_grad_norm,
            grpo_ratio_mean=grpo.ratio_mean,
            grpo_zero_variance_group_rate=grpo.zero_variance_group_rate,
            effective_adv_abs_max=grpo.effective_adv_abs_max,
            grpo_reference_kl=grpo.reference_kl,
            optimizer_steps=1,
            grpo_policy_epochs=cfg.grpo.policy_epochs,
            grpo_policy_micro_batch_size=cfg.grpo.policy_micro_batch_size,
        )
        self.iteration += 1
        return metrics
