"""Stage 2A classifier over frozen Stage 1 motif graphs."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class MotifGraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 7,
        arch: str = "transformer",
        readout: str = "weighted_mean_max",
        **_: Any,
    ) -> None:
        super().__init__()
        if input_dim is None:
            raise ValueError("MotifGraphClassifier requires input_dim after Stage 2 feature inference")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.arch = str(arch or "transformer").lower()
        self.readout = str(readout)
        if self.arch == "transformer":
            self.node_proj = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=self.hidden_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            repr_dim = self.hidden_dim * 2 if self.readout == "weighted_mean_max" else self.hidden_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(repr_dim),
                nn.Dropout(float(dropout)),
                nn.Linear(repr_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        elif self.arch == "pooled_mlp":
            self.node_proj = None
            self.encoder = None
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.input_dim * 3),
                nn.Linear(self.input_dim * 3, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            raise ValueError(f"Unsupported motif graph classifier arch={self.arch!r}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MotifGraphClassifier":
        cfg = dict(config)
        return cls(**cfg)

    def forward(
        self,
        motif_node_features: torch.Tensor,
        motif_edge_features: torch.Tensor | None = None,
        selected_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del motif_edge_features
        if self.arch == "pooled_mlp":
            if selected_weights is None:
                weights = motif_node_features.new_full(
                    motif_node_features.shape[:2],
                    1.0 / max(motif_node_features.shape[1], 1),
                )
            else:
                weights = selected_weights.to(dtype=motif_node_features.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mean_pool = motif_node_features.mean(dim=1)
            max_pool = motif_node_features.max(dim=1).values
            weighted_pool = (motif_node_features * weights.unsqueeze(-1)).sum(dim=1)
            graph_repr = torch.cat([mean_pool, max_pool, weighted_pool], dim=-1)
            return self.classifier(graph_repr)

        h = self.encoder(self.node_proj(motif_node_features))
        if selected_weights is None:
            weights = h.new_full(h.shape[:2], 1.0 / max(h.shape[1], 1))
        else:
            weights = selected_weights.to(dtype=h.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted_mean = (h * weights.unsqueeze(-1)).sum(dim=1)
        if self.readout == "weighted_mean":
            graph_repr = weighted_mean
        elif self.readout == "weighted_mean_max":
            graph_repr = torch.cat([weighted_mean, h.max(dim=1).values], dim=-1)
        else:
            raise ValueError(f"Unsupported readout={self.readout!r}")
        return self.classifier(graph_repr)


FrozenMotifGraphClassifier = MotifGraphClassifier
