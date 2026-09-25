from dataclasses import dataclass

import torch


@dataclass
class OPDResult:
    loss: torch.Tensor
    kl: torch.Tensor
    valid_tokens: int


def _prediction_mask(response_mask: torch.Tensor) -> torch.Tensor:
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, sequence].")
    return response_mask[:, 1:].to(dtype=torch.bool)


def _reduce_positions(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_mean: bool,
) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    if sample_mean:
        per_sample = (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        return per_sample.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def topk_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    topk: int = 16,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    reverse: bool = False,
    sample_mean: bool = True,
) -> OPDResult:
    """Compute the active student-top-k OPD objective.

    Token rewards and student probability weights are detached, matching the
    policy-gradient-style OPD path in the parent trainer. Reverse OPD is the
    exact negative of the same forward objective.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    if student_logits.ndim != 3:
        raise ValueError("Logits must have shape [batch, sequence, vocabulary].")
    if not 0 < topk <= student_logits.size(-1):
        raise ValueError("topk must be in [1, vocabulary_size].")

    mask = _prediction_mask(response_mask).to(device=student_logits.device)
    student = student_logits[:, :-1].float() / float(student_temperature)
    teacher = teacher_logits[:, :-1].detach().float() / float(teacher_temperature)

    student_topk_logits, token_ids = torch.topk(student, k=topk, dim=-1)
    student_logps = student_topk_logits - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_selected = teacher.gather(-1, token_ids)
    teacher_logps = teacher_selected - torch.logsumexp(teacher, dim=-1, keepdim=True)

    weights = torch.softmax(student_logps, dim=-1).detach()
    token_kl = student_logps.detach() - teacher_logps.detach()
    token_advantages = (-token_kl * weights).detach()
    position_loss = (-token_advantages * student_logps).sum(dim=-1)
    position_kl = (weights * token_kl).sum(dim=-1)

    loss = _reduce_positions(position_loss, mask, sample_mean)
    if reverse:
        loss = -loss
    return OPDResult(
        loss=loss,
        kl=_reduce_positions(position_kl, mask, sample_mean=False).detach(),
        valid_tokens=int(mask.sum().item()),
    )

