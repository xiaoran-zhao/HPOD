from dataclasses import dataclass

import torch


@dataclass
class OPDBatch:
    """Differentiable student logits and frozen teacher targets for one side."""

    student_logits: torch.Tensor
    teacher_logits: torch.Tensor
    response_mask: torch.Tensor


@dataclass
class GRPOBatch:
    """Current-policy scoring data for trajectories generated before this step."""

    current_token_logps: torch.Tensor
    old_token_logps: torch.Tensor
    response_mask: torch.Tensor
    rewards: torch.Tensor
    reference_token_logps: torch.Tensor


@dataclass
class StrictOnPolicyBatch:
    """All tensors generated from one unchanged pre-update Student policy."""

    iteration_id: int
    retain: OPDBatch
    forget: OPDBatch
    grpo: GRPOBatch
    forget_calibration_kl: float
    consumed: bool = False

    def assert_fresh(self, expected_iteration: int) -> None:
        if self.consumed:
            raise RuntimeError("This rollout batch has already been consumed by an optimizer update.")
        if int(self.iteration_id) != int(expected_iteration):
            raise RuntimeError(
                f"Expected rollout iteration {expected_iteration}, received {self.iteration_id}."
            )

    def consume(self) -> None:
        if self.consumed:
            raise RuntimeError("A strict on-policy rollout batch cannot be reused.")
        self.consumed = True
