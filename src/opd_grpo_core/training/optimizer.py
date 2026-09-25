import torch


def build_adamw(
    parameters: list[torch.nn.Parameter],
    *,
    learning_rate: float = 3e-5,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Build the AdamW optimizer used by the active experiment configuration."""
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=betas,
        eps=float(eps),
        weight_decay=float(weight_decay),
    )

