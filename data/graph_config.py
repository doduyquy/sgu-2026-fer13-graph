"""Graph construction configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class GraphConfig:
    """Feature contract for FER-2013 48x48 pixel graphs."""

    height: int = 48
    width: int = 48
    connectivity: int = 8
    normalize_pixels: bool = True
    node_feature_names: List[str] = field(
        default_factory=lambda: [
            "intensity",
            "x_norm",
            "y_norm",
            "gx",
            "gy",
            "grad_mag",
            "local_contrast",
        ]
    )
    edge_static_feature_names: List[str] = field(
        default_factory=lambda: ["dx", "dy", "dist"]
    )
    edge_dynamic_feature_names: List[str] = field(
        default_factory=lambda: ["delta_intensity", "intensity_similarity"]
    )
    intensity_similarity_alpha: float = 1.0
    chunk_size: int = 500
    version: str = "d5_graph_v1"

    @property
    def num_nodes(self) -> int:
        return int(self.height * self.width)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "GraphConfig":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def expected_edge_count(self) -> int:
        h, w = int(self.height), int(self.width)
        if self.connectivity == 4:
            return 2 * h * (w - 1) + 2 * (h - 1) * w
        if self.connectivity == 8:
            return 2 * h * (w - 1) + 2 * (h - 1) * w + 4 * (h - 1) * (w - 1)
        raise ValueError(f"Unsupported connectivity: {self.connectivity}")
