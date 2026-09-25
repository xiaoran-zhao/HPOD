from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressState:
    d0: Optional[float] = None
    ema_retain_grad_norm: Optional[float] = None
    ema_reverse_grad_norm: Optional[float] = None


@dataclass
class ProgressMetrics:
    gradient_balance_ratio: float
    progress_factor: float
    adaptive_lambda: float


class ProgressGradBalance:
    """Adjust only the Reverse OPD weight while Retain OPD stays anchored at 1."""

    def __init__(self, beta: float = 0.9, alpha: float = 1.0, eps: float = 1e-8):
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.state = ProgressState()

    def update(
        self,
        retain_grad_norm: float,
        reverse_grad_norm: float,
        forget_kl: float,
    ) -> ProgressMetrics:
        state = self.state
        if state.d0 is None:
            state.d0 = float(forget_kl)
        if state.ema_retain_grad_norm is None:
            state.ema_retain_grad_norm = float(retain_grad_norm)
            state.ema_reverse_grad_norm = float(reverse_grad_norm)
        else:
            state.ema_retain_grad_norm = (
                self.beta * state.ema_retain_grad_norm + (1.0 - self.beta) * float(retain_grad_norm)
            )
            state.ema_reverse_grad_norm = (
                self.beta * state.ema_reverse_grad_norm + (1.0 - self.beta) * float(reverse_grad_norm)
            )

        grad_ratio = (state.ema_retain_grad_norm + self.eps) / (
            state.ema_reverse_grad_norm + self.eps
        )
        progress_factor = (state.d0 + self.eps) / (float(forget_kl) + self.eps)
        adaptive_lambda = grad_ratio * progress_factor**self.alpha
        return ProgressMetrics(grad_ratio, progress_factor, adaptive_lambda)
