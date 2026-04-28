"""Dataclasses for canonical FER-2013 pixel graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class SharedGraphStructure:
    """Topology and static edge attributes shared by every 48x48 image."""

    height: int
    width: int
    connectivity: int
    edge_index: torch.Tensor
    edge_attr_static: torch.Tensor
    static_feature_names: List[str] = field(default_factory=list)
    config_dict: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PixelGraphSample:
    """Per-image graph payload stored in graph repository chunks."""

    graph_id: int
    label: int
    split: str
    usage: str
    height: int
    width: int
    node_features: torch.Tensor
    edge_attr_dynamic: torch.Tensor
    node_feature_names: List[str] = field(default_factory=list)
    dynamic_feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedPixelGraph:
    """Full graph after joining shared topology with per-image features."""

    graph_id: int
    label: int
    split: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_feature_names: List[str] = field(default_factory=list)
    edge_feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
