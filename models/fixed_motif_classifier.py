"""Fixed D5B motif-prior classifier."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FixedMotifMLPClassifier(nn.Module):
    """Pool fixed class motif priors over full-graph node features."""

    def __init__(
        self,
        node_prior: torch.Tensor,
        node_dim: int = 7,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(node_prior, dtype=torch.float32)
        if prior.ndim != 2:
            raise ValueError(f"node_prior must be [num_classes, num_nodes], got {tuple(prior.shape)}")
        if prior.shape[0] != int(num_classes):
            raise ValueError(f"Expected {num_classes} class priors, got {prior.shape[0]}")
        self.num_classes = int(num_classes)
        self.num_nodes = int(prior.shape[1])
        self.node_dim = int(node_dim)
        self.eps = float(eps)
        self.register_buffer("node_prior", prior, persistent=True)

        in_dim = self.num_classes * self.node_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, batch_or_x) -> Dict[str, torch.Tensor]:
        if isinstance(batch_or_x, dict):
            x = batch_or_x.get("x", batch_or_x.get("node_features"))
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("FixedMotifMLPClassifier needs 'x' or 'node_features'")
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")
        if x.shape[2] != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {x.shape[2]}")

        x = x.float()
        prior = self.node_prior.to(device=x.device, dtype=x.dtype)
        denom = prior.sum(dim=1).clamp_min(self.eps)

        weighted_mean = torch.einsum("cn,bnd->bcd", prior, x) / denom.view(1, -1, 1)
        centered = x.unsqueeze(1) - weighted_mean.unsqueeze(2)
        weighted_var = (
            prior.view(1, self.num_classes, self.num_nodes, 1) * centered.pow(2)
        ).sum(dim=2) / denom.view(1, -1, 1)
        weighted_std = weighted_var.clamp_min(0.0).sqrt()
        weighted_max = (
            prior.view(1, self.num_classes, self.num_nodes, 1) * x.unsqueeze(1)
        ).amax(dim=2)
        weighted_energy = torch.einsum("cn,bnd->bcd", prior, x.pow(2)) / denom.view(1, -1, 1)

        motif_features = torch.cat(
            [weighted_mean, weighted_std, weighted_max, weighted_energy],
            dim=-1,
        )
        logits = self.mlp(motif_features.flatten(start_dim=1))
        return {
            "logits": logits,
            "motif_features": motif_features,
            "diagnostics": {
                "prior_min": prior.detach().amin(),
                "prior_max": prior.detach().amax(),
                "prior_mean": prior.detach().mean(),
                "motif_feature_mean": motif_features.detach().mean(),
                "motif_feature_std": motif_features.detach().std(unbiased=False),
            },
        }
