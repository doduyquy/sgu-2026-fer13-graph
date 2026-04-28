"""Resolve stored graph samples into full graph tensors."""

from __future__ import annotations

import torch

from data.graph_types import PixelGraphSample, ResolvedPixelGraph, SharedGraphStructure


class GraphResolver:
    """Join shared topology/static features with per-image dynamic features."""

    def __init__(self, shared: SharedGraphStructure) -> None:
        self.shared = shared
        self.edge_count = int(shared.edge_index.shape[1])
        self.height = int(shared.height)
        self.width = int(shared.width)

    def resolve(self, sample: PixelGraphSample) -> ResolvedPixelGraph:
        if int(sample.height) != self.height or int(sample.width) != self.width:
            raise ValueError(
                f"Sample size mismatch: expected {(self.height, self.width)}, "
                f"got {(sample.height, sample.width)}"
            )
        if sample.edge_attr_dynamic.shape[0] != self.edge_count:
            raise ValueError(
                f"Dynamic edge count mismatch: expected {self.edge_count}, "
                f"got {sample.edge_attr_dynamic.shape[0]}"
            )
        if sample.node_features.shape[0] != self.height * self.width:
            raise ValueError("Node count mismatch")

        edge_attr = torch.cat(
            [
                self.shared.edge_attr_static.float(),
                sample.edge_attr_dynamic.float(),
            ],
            dim=1,
        )
        if not torch.isfinite(sample.node_features).all():
            raise ValueError(f"Non-finite node features for graph_id={sample.graph_id}")
        if not torch.isfinite(edge_attr).all():
            raise ValueError(f"Non-finite edge attrs for graph_id={sample.graph_id}")

        return ResolvedPixelGraph(
            graph_id=int(sample.graph_id),
            label=int(sample.label),
            split=sample.split,
            node_features=sample.node_features.float(),
            edge_index=self.shared.edge_index.long(),
            edge_attr=edge_attr.float(),
            node_feature_names=list(sample.node_feature_names),
            edge_feature_names=list(self.shared.static_feature_names)
            + list(sample.dynamic_feature_names),
            metadata=dict(sample.metadata),
        )
