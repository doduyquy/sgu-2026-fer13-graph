"""Losses for D5 class-level graph retrieval."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights(
    class_counts,
    normalize_mean: bool = True,
    power: float = 1.0,
) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights.pow(float(power))
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-8)
    return weights


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, y.long(), weight=weight)


class D5RetrievalLoss(nn.Module):
    """CE plus soft-subgraph regularizers for D5A retrieval."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.lambda_cls = float(cfg.get("lambda_cls", 1.0))
        self.lambda_contrast = float(cfg.get("lambda_contrast", 0.1))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 0.01))
        self.lambda_closure = float(cfg.get("lambda_closure", 0.01))
        self.lambda_area = float(cfg.get("lambda_area", 0.01))
        self.lambda_div = float(cfg.get("lambda_div", 0.0))
        self.margin = float(cfg.get("margin", 0.2))
        self.target_area = float(cfg.get("target_area", 0.15))

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        node_attn = model_out["node_attn"]
        edge_attn = model_out["edge_attn"]
        edge_index = batch["edge_index"].long()
        y = y.long()

        loss_cls = self.ce(logits, y)
        loss_contrast = self._contrast_loss(logits, y)
        loss_smooth = self._smoothness_loss(node_attn, edge_index, y)
        loss_closure = self._closure_loss(node_attn, edge_attn, edge_index, y)
        loss_area = self._area_loss(node_attn)
        loss_div = self._diversity_loss(model_out)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_contrast * loss_contrast
            + self.lambda_smooth * loss_smooth
            + self.lambda_closure * loss_closure
            + self.lambda_area * loss_area
            + self.lambda_div * loss_div
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_contrast": loss_contrast,
            "loss_smooth": loss_smooth,
            "loss_closure": loss_closure,
            "loss_area": loss_area,
            "loss_div": loss_div,
        }

    def _contrast_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz, num_classes = logits.shape
        true_score = logits[torch.arange(bsz, device=logits.device), y]
        wrong = logits.masked_fill(F.one_hot(y, num_classes=num_classes).bool(), -1e9)
        max_wrong = wrong.max(dim=1).values
        return F.relu(self.margin - true_score + max_wrong).mean()

    @staticmethod
    def _smoothness_loss(
        node_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        true_attn = node_attn[torch.arange(bsz, device=node_attn.device), y]
        src = edge_index[0]
        dst = edge_index[1]
        return (true_attn[:, src] - true_attn[:, dst]).pow(2).mean()

    @staticmethod
    def _closure_loss(
        node_attn: torch.Tensor,
        edge_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        rows = torch.arange(bsz, device=node_attn.device)
        node_true = node_attn[rows, y]
        edge_true = edge_attn[rows, y]
        src = edge_index[0]
        dst = edge_index[1]
        endpoint_min = torch.minimum(node_true[:, src], node_true[:, dst])
        return F.relu(edge_true - endpoint_min).pow(2).mean()

    def _area_loss(self, node_attn: torch.Tensor) -> torch.Tensor:
        mass = node_attn.mean(dim=-1)
        return (mass - self.target_area).pow(2).mean()

    @staticmethod
    def _diversity_loss(model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        gate = model_out.get("class_node_gate")
        if gate is None or gate.shape[0] <= 1:
            logits = model_out["logits"]
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        flat = F.normalize(gate.float(), dim=1, eps=1e-6)
        sim = flat @ flat.t()
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()
