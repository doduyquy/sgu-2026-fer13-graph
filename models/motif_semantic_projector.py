"""Stage 2C semantic projector for frozen D8M motif representations."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


class MotifSemanticProjector(nn.Module):
    """Pool selected motif features, learn a normalized projection, and classify."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
        classifier_input: str = "z",
        projection_normalize: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if input_dim is None:
            raise ValueError("MotifSemanticProjector requires input_dim after Stage 2 feature inference")
        self.input_dim = int(input_dim)
        self.pooled_dim = self.input_dim * 3
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.num_classes = int(num_classes)
        self.classifier_input = str(classifier_input or "z").lower()
        self.projection_normalize = bool(projection_normalize)
        if self.classifier_input not in {"z", "hidden"}:
            raise ValueError(f"Unsupported classifier_input={self.classifier_input!r}")

        self.backbone = nn.Sequential(
            nn.LayerNorm(self.pooled_dim),
            nn.Linear(self.pooled_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.projection_dim),
        )
        cls_dim = self.projection_dim if self.classifier_input == "z" else self.hidden_dim
        self.classifier = nn.Linear(cls_dim, self.num_classes)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MotifSemanticProjector":
        return cls(**dict(config))

    @staticmethod
    def pool_motifs(motif_node_features: torch.Tensor, selected_weights: torch.Tensor | None = None) -> torch.Tensor:
        if motif_node_features.ndim != 3:
            raise ValueError(f"motif_node_features must be [B, M, F], got {tuple(motif_node_features.shape)}")
        if selected_weights is None:
            weights = motif_node_features.new_full(
                motif_node_features.shape[:2],
                1.0 / max(int(motif_node_features.shape[1]), 1),
            )
        else:
            weights = selected_weights.to(device=motif_node_features.device, dtype=motif_node_features.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean_pool = motif_node_features.mean(dim=1)
        max_pool = motif_node_features.max(dim=1).values
        weighted_pool = (motif_node_features * weights.unsqueeze(-1)).sum(dim=1)
        return torch.cat([mean_pool, max_pool, weighted_pool], dim=1)

    def forward(
        self,
        motif_node_features: torch.Tensor | None = None,
        selected_weights: torch.Tensor | None = None,
        pooled_repr: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if pooled_repr is None:
            if motif_node_features is None:
                raise ValueError("forward requires motif_node_features or pooled_repr")
            pooled_repr = self.pool_motifs(motif_node_features, selected_weights)
        hidden = self.backbone(pooled_repr)
        z_raw = self.projector(hidden)
        z = F.normalize(z_raw, dim=1, eps=1e-8) if self.projection_normalize else z_raw
        cls_input = z if self.classifier_input == "z" else hidden
        logits = self.classifier(cls_input)
        return {
            "logits": logits,
            "z": z,
            "z_raw": z_raw,
            "hidden": hidden,
            "pooled_repr": pooled_repr,
        }


FrozenMotifSemanticProjector = MotifSemanticProjector
