"""Supervised contrastive loss for Stage 2C motif semantic projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SupervisedContrastiveLoss(nn.Module):
    """Khosla-style SupCon loss for one normalized view per sample."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)} for features {tuple(features.shape)}")
        batch_size = int(features.shape[0])
        if batch_size <= 1:
            return features.sum() * 0.0

        z = F.normalize(features.float(), dim=1, eps=1e-8)
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.t()).to(device=z.device, dtype=z.dtype)
        self_mask = torch.eye(batch_size, device=z.device, dtype=z.dtype)
        positive_mask = positive_mask * (1.0 - self_mask)

        logits = torch.matmul(z, z.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = 1.0 - self_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not bool(valid.any()):
            return features.sum() * 0.0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid].clamp_min(1.0)
        return -mean_log_prob_pos.mean()
