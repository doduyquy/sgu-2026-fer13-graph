"""Model registry for the clean D5 project."""

from __future__ import annotations

from typing import Any, Dict

from fer_d5.models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval


def build_model(config: Dict[str, Any]) -> ClassPixelMotifGraphRetrieval:
    cfg = dict(config)
    name = cfg.pop("name", "class_pixel_motif_graph_retrieval")
    if name != "class_pixel_motif_graph_retrieval":
        raise ValueError(f"Unknown D5 model: {name}")
    return ClassPixelMotifGraphRetrieval.from_config(cfg)
