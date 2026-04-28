"""Full graph dataset and collate function for D5A."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from fer_d5.data.graph_repository import ChunkedGraphDataset
from fer_d5.data.graph_types import ResolvedPixelGraph


class FullGraphDataset(Dataset):
    """Return one resolved 48x48 pixel graph per FER-2013 sample."""

    def __init__(
        self,
        repo_root: str | Path,
        split: str,
        graph_cache_chunks: int = 1,
    ) -> None:
        self.dataset = ChunkedGraphDataset(
            repo_root=repo_root,
            split=split,
            resolve=True,
            graph_cache_chunks=graph_cache_chunks,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        graph = self.dataset[int(idx)]
        if not isinstance(graph, ResolvedPixelGraph):
            raise TypeError("FullGraphDataset expected a resolved graph")
        num_nodes = int(graph.node_features.shape[0])
        return {
            "graph_id": torch.tensor(int(graph.graph_id), dtype=torch.long),
            "node_features": graph.node_features.float(),
            "x": graph.node_features.float(),
            "edge_index": graph.edge_index.long(),
            "edge_attr": graph.edge_attr.float(),
            "node_mask": torch.ones(num_nodes, dtype=torch.bool),
            "y": torch.tensor(int(graph.label), dtype=torch.long),
            "label": torch.tensor(int(graph.label), dtype=torch.long),
        }


def collate_fn_full_graph(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate resolved graphs into the D5A batch contract."""

    x = torch.stack([item["node_features"] for item in batch], dim=0)
    edge_attr = torch.stack([item["edge_attr"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    node_mask = torch.stack([item["node_mask"] for item in batch], dim=0)
    graph_id = torch.stack([item["graph_id"] for item in batch], dim=0)
    return {
        "graph_id": graph_id,
        "x": x,
        "node_features": x,
        "edge_index": batch[0]["edge_index"],
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "y": y,
        "label": y,
    }
