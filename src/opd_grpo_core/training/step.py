from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..controllers import ProgressGradBalance
from ..coordination import behavior_constrained_projection
from .gradients import add_gradients, apply_gradients_once, gradient_norm

GradientTuple = Tuple[Optional[torch.Tensor], ...]


@dataclass
class CoordinateResult:
    adaptive_lambda: float
    retain_grad_norm: float
    reverse_grad_norm: float
    behavior_grad_norm: float
    projection_active: bool
    final_grad_norm: float


def coordinate_update(
    *,
    retain_grads: GradientTuple,
    reverse_grads: GradientTuple,
    behavior_grads: GradientTuple,
    forget_kl: float,
    controller: ProgressGradBalance,
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> CoordinateResult:
    """Coordinate cached batch gradients and perform one optimizer step."""
    retain_norm = gradient_norm(retain_grads)
    reverse_norm = gradient_norm(reverse_grads)
    adaptive = controller.update(retain_norm, reverse_norm, forget_kl)
    knowledge_grads = add_gradients(retain_grads, reverse_grads, adaptive.adaptive_lambda)
    projection = behavior_constrained_projection(knowledge_grads, behavior_grads)
    apply_gradients_once(
        parameters,
        projection.gradients,
        optimizer,
        max_grad_norm=max_grad_norm,
    )
    return CoordinateResult(
        adaptive_lambda=adaptive.adaptive_lambda,
        retain_grad_norm=retain_norm,
        reverse_grad_norm=reverse_norm,
        behavior_grad_norm=projection.behavior_grad_norm,
        projection_active=projection.projection_active,
        final_grad_norm=projection.final_grad_norm,
    )
