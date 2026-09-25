from dataclasses import dataclass, field


@dataclass(frozen=True)
class OPDConfig:
    topk: int = 16
    student_temperature: float = 1.0
    teacher_temperature: float = 1.0
    sample_mean_reduction: bool = True


@dataclass(frozen=True)
class AdaptiveConfig:
    ema_beta: float = 0.9
    progress_alpha: float = 1.0
    epsilon: float = 1e-8


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_range: float = 0.2
    advantage_epsilon: float = 1e-6
    policy_epochs: int = 1
    policy_micro_batch_size: int = 1
    reference_kl_enabled: bool = True
    reference_kl_mode: str = "sampled"
    reference_kl_beta: float = 0.02


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class CoreConfig:
    """The active, non-legacy settings used by the extracted method."""

    retain_batch_size: int = 8
    forget_batch_size: int = 8
    strict_on_policy: bool = True
    retain_weight: float = 1.0
    opd: OPDConfig = field(default_factory=OPDConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
