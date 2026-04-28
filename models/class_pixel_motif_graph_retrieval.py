"""D5A class-level pixel motif graph retrieval model."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.edge_gnn import EdgeAwarePixelGNNEncoder


class ClassPixelMotifGraphRetrieval(nn.Module):
    """Class-level soft subgraph retrieval over the full pixel graph.

    The classifier score is the weighted matching energy between encoded
    image nodes/edges and learnable class prototypes. There is no CNN branch,
    global image pooling shortcut, hard top-K selection, or legacy motif bank.
    """

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        num_edges: int = 17860,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        edge_hidden_dim: int = 64,
        gnn_layers: int = 1,
        use_edge_gnn: bool = True,
        dropout: float = 0.2,
        temperature: float = 0.2,
        edge_score_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.num_edges = int(num_edges)
        self.hidden_dim = int(hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.temperature = float(temperature)
        self.edge_score_weight = float(edge_score_weight)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        gnn_count = int(gnn_layers) if use_edge_gnn else 0
        self.gnn = EdgeAwarePixelGNNEncoder(
            num_layers=gnn_count,
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout,
        )

        self.class_node_proto = nn.Parameter(
            torch.empty(self.num_classes, self.num_nodes, hidden_dim)
        )
        self.class_edge_proto = nn.Parameter(
            torch.empty(self.num_classes, self.num_edges, edge_hidden_dim)
        )
        self.class_node_gate_logits = nn.Parameter(torch.zeros(self.num_classes, self.num_nodes))
        self.class_edge_gate_logits = nn.Parameter(torch.zeros(self.num_classes, self.num_edges))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_node_proto, mean=0.0, std=0.02)
        nn.init.normal_(self.class_edge_proto, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClassPixelMotifGraphRetrieval":
        cfg = dict(config)
        height = int(cfg.pop("height", 48))
        width = int(cfg.pop("width", 48))
        connectivity = int(cfg.pop("connectivity", 8))
        cfg.setdefault("num_nodes", height * width)
        cfg.setdefault("num_edges", _expected_edge_count(height, width, connectivity))
        return cls(**cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | Dict]:
        x = batch.get("x", batch.get("node_features"))
        if x is None:
            raise KeyError("Batch must contain 'x' or 'node_features'")
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        node_mask = batch.get("node_mask")

        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")
        if edge_attr.shape[1] != self.num_edges:
            raise ValueError(f"Expected {self.num_edges} edges, got {edge_attr.shape[1]}")

        h = self.node_encoder(x.float())
        e = self.edge_encoder(edge_attr.float())
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        h = self.gnn(h, edge_index=edge_index, edge_attr=e, node_mask=node_mask)

        node_sim = self._node_similarity(h)
        edge_sim = self._edge_similarity(e)

        class_node_gate = torch.sigmoid(self.class_node_gate_logits)
        class_edge_gate = torch.sigmoid(self.class_edge_gate_logits)
        node_attn = torch.sigmoid(node_sim / max(self.temperature, 1e-6))
        edge_attn = torch.sigmoid(edge_sim / max(self.temperature, 1e-6))
        node_attn = node_attn * class_node_gate.unsqueeze(0)
        edge_attn = edge_attn * class_edge_gate.unsqueeze(0)
        if node_mask is not None:
            node_attn = node_attn * node_mask.unsqueeze(1).to(node_attn.dtype)

        node_score = self._weighted_score(node_attn, node_sim)
        edge_score = self._weighted_score(edge_attn, edge_sim)
        logits = node_score + self.edge_score_weight * edge_score

        return {
            "logits": logits,
            "node_attn": node_attn,
            "edge_attn": edge_attn,
            "class_node_gate": class_node_gate,
            "class_edge_gate": class_edge_gate,
            "node_sim": node_sim,
            "edge_sim": edge_sim,
            "aux_loss": torch.zeros((), device=logits.device, dtype=logits.dtype),
            "aux": {},
            "diagnostics": self._diagnostics(
                logits=logits,
                node_attn=node_attn,
                edge_attn=edge_attn,
                class_node_gate=class_node_gate,
                class_edge_gate=class_edge_gate,
            ),
        }

    def _node_similarity(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, dim=-1, eps=1e-6)
        proto = F.normalize(self.class_node_proto, dim=-1, eps=1e-6)
        return torch.einsum("bnh,cnh->bcn", h_norm, proto)

    def _edge_similarity(self, e: torch.Tensor) -> torch.Tensor:
        e_norm = F.normalize(e, dim=-1, eps=1e-6)
        proto = F.normalize(self.class_edge_proto, dim=-1, eps=1e-6)
        return torch.einsum("beh,ceh->bce", e_norm, proto)

    @staticmethod
    def _weighted_score(attn: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        denom = attn.sum(dim=-1).clamp_min(1e-6)
        return (attn * sim).sum(dim=-1) / denom

    @staticmethod
    def _diagnostics(
        logits: torch.Tensor,
        node_attn: torch.Tensor,
        edge_attn: torch.Tensor,
        class_node_gate: torch.Tensor,
        class_edge_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        top2 = torch.topk(logits.detach(), k=min(2, logits.shape[1]), dim=1).values
        if top2.shape[1] == 2:
            margin = (top2[:, 0] - top2[:, 1]).mean()
        else:
            margin = torch.zeros((), device=logits.device)
        return {
            "node_attn_mass_mean": node_attn.detach().mean(),
            "node_attn_mass_min": node_attn.detach().amin(),
            "node_attn_mass_max": node_attn.detach().amax(),
            "edge_attn_mass_mean": edge_attn.detach().mean(),
            "edge_attn_mass_min": edge_attn.detach().amin(),
            "edge_attn_mass_max": edge_attn.detach().amax(),
            "class_node_gate_mean": class_node_gate.detach().mean(),
            "class_node_gate_min": class_node_gate.detach().amin(),
            "class_node_gate_max": class_node_gate.detach().amax(),
            "class_edge_gate_mean": class_edge_gate.detach().mean(),
            "class_edge_gate_min": class_edge_gate.detach().amin(),
            "class_edge_gate_max": class_edge_gate.detach().amax(),
            "class_score_margin": margin,
            "node_attn_nan_count": torch.isnan(node_attn).sum(),
            "edge_attn_nan_count": torch.isnan(edge_attn).sum(),
            "logits_nan_count": torch.isnan(logits).sum(),
        }


def _expected_edge_count(height: int, width: int, connectivity: int) -> int:
    if connectivity == 4:
        return 2 * height * (width - 1) + 2 * (height - 1) * width
    if connectivity == 8:
        return 2 * height * (width - 1) + 2 * (height - 1) * width + 4 * (height - 1) * (width - 1)
    raise ValueError(f"Unsupported connectivity: {connectivity}")
