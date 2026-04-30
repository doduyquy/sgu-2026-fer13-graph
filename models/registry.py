"""Model registry for the clean D5/D6 project."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval
from models.slot_pixel_part_graph_motif import SlotPixelPartGraphMotif


def build_model(config: Dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    name = cfg.pop("name", "class_pixel_motif_graph_retrieval")
    if name == "class_pixel_motif_graph_retrieval":
        return ClassPixelMotifGraphRetrieval.from_config(cfg)
    if name == "slot_pixel_part_graph_motif":
        return SlotPixelPartGraphMotif.from_config(cfg)
    raise ValueError(f"Unknown model: {name}")
