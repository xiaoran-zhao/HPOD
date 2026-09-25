import torch
from typing import Optional, Tuple

GradientTuple = Tuple[Optional[torch.Tensor], ...]


def gradient_norm(grads: GradientTuple) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().float().square().sum().item())
    return total**0.5


def add_gradients(left: GradientTuple, right: GradientTuple, right_scale: float = 1.0) -> GradientTuple:
    if len(left) != len(right):
        raise ValueError("Gradient tuples must have equal length.")
    combined: list[torch.Tensor | None] = []
    for left_grad, right_grad in zip(left, right):
        if left_grad is None and right_grad is None:
            combined.append(None)
        elif left_grad is None:
            combined.append(right_grad.detach().float() * float(right_scale))
        elif right_grad is None:
            combined.append(left_grad.detach().float().clone())
        else:
            combined.append(left_grad.detach().float() + float(right_scale) * right_grad.detach().float())
    return tuple(combined)


def apply_gradients_once(
    parameters: list[torch.nn.Parameter],
    gradients: GradientTuple,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> None:
    """Assign coordinated gradients and execute exactly one optimizer update."""
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient counts must match.")
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.to(parameter)
    if float(max_grad_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
