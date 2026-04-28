"""D5 model package."""

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from fer_d5.models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval
from fer_d5.models.registry import build_model

__all__ = ["ClassPixelMotifGraphRetrieval", "build_model"]
