from dataclasses import dataclass
from typing import Optional, Tuple

import torch

GradientTuple = Tuple[Optional[torch.Tensor], ...]


def _zeros_like_missing(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if left is None and right is None:
        raise ValueError("At least one gradient must exist.")
    if left is None:
        left = torch.zeros_like(right)
    if right is None:
        right = torch.zeros_like(left)
    return left.detach().float(), right.detach().float()


def _geometry(left: GradientTuple, right: GradientTuple, eps: float) -> tuple[float, float, float, float]:
    left_sq = right_sq = dot = 0.0
    for left_grad, right_grad in zip(left, right):
        if left_grad is None and right_grad is None:
            continue
        left_value, right_value = _zeros_like_missing(left_grad, right_grad)
        left_sq += float(left_value.square().sum().item())
        right_sq += float(right_value.square().sum().item())
        dot += float((left_value * right_value).sum().item())
    left_norm = left_sq**0.5
    right_norm = right_sq**0.5
    cosine = dot / (left_norm * right_norm + eps) if left_norm and right_norm else 0.0
    return left_norm, right_norm, dot, cosine


@dataclass
class ProjectionResult:
    gradients: GradientTuple
    knowledge_grad_norm: float
    behavior_grad_norm: float
    dot: float
    cosine: float
    projection_active: bool
    coefficient: float
    final_grad_norm: float


def behavior_constrained_projection(
    knowledge_grads: GradientTuple,
    behavior_grads: GradientTuple,
    eps: float = 1e-8,
) -> ProjectionResult:
    """Remove only the knowledge component that opposes the behavior gradient.

    The behavior gradient is not rescaled. The final update is
    projected_knowledge + behavior, matching the current formal trainer path.
    """
    if len(knowledge_grads) != len(behavior_grads):
        raise ValueError("Gradient tuples must have equal length.")
    knowledge_norm, behavior_norm, dot, cosine = _geometry(knowledge_grads, behavior_grads, eps)
    active = behavior_norm > eps and dot < 0.0
    coefficient = dot / (behavior_norm * behavior_norm + eps) if active else 0.0

    final: list[torch.Tensor | None] = []
    for knowledge_grad, behavior_grad in zip(knowledge_grads, behavior_grads):
        if knowledge_grad is None and behavior_grad is None:
            final.append(None)
            continue
        knowledge, behavior = _zeros_like_missing(knowledge_grad, behavior_grad)
        projected = knowledge - coefficient * behavior if active else knowledge
        final.append(projected + behavior)
    final_tuple = tuple(final)
    final_norm, _, _, _ = _geometry(final_tuple, tuple(None for _ in final_tuple), eps)
    return ProjectionResult(
        gradients=final_tuple,
        knowledge_grad_norm=knowledge_norm,
        behavior_grad_norm=behavior_norm,
        dot=dot,
        cosine=cosine,
        projection_active=active,
        coefficient=coefficient,
        final_grad_norm=final_norm,
    )
