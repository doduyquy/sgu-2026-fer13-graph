"""Small edge-aware GNN encoder used by D5A."""

from __future__ import annotations

import torch
from torch import nn


class EdgeAwarePixelGNNLayer(nn.Module):
    """Message passing over a shared directed pixel graph.

    Input shapes:
        h: ``[B, N, hidden_dim]``
        edge_index: ``[2, E]``
        edge_attr: ``[B, E, edge_attr_dim]`` or ``[E, edge_attr_dim]``
        node_mask: ``[B, N]``
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        edge_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, num_nodes, _ = h.shape
        src = edge_index[0].long()
        dst = edge_index[1].long()
        num_edges = int(src.numel())
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(bsz, -1, -1)

        encoded_edges = self.edge_encoder(edge_attr)
        src_h = h[:, src, :]
        messages = self.message_mlp(torch.cat([src_h, encoded_edges], dim=-1))

        aggregated = torch.zeros_like(h, dtype=messages.dtype)
        degree = torch.zeros(bsz, num_nodes, 1, device=h.device, dtype=messages.dtype)
        ones = torch.ones(num_edges, 1, device=h.device, dtype=messages.dtype)
        for b in range(bsz):
            aggregated[b].index_add_(0, dst, messages[b])
            degree[b].index_add_(0, dst, ones)
        aggregated = aggregated / degree.clamp_min(1.0)

        h = self.msg_norm(h + self.dropout(aggregated))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return h


class EdgeAwarePixelGNNEncoder(nn.Module):
    """A stack of ``EdgeAwarePixelGNNLayer`` blocks."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        edge_attr_dim: int,
        edge_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EdgeAwarePixelGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        return h
