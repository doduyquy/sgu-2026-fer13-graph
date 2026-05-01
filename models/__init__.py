"""D5/D6 model package."""

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval
from models.dual_branch_graph_swin_motif import DualBranchGraphSwinMotifD7
from models.fixed_motif_classifier import FixedMotifMLPClassifier
from models.registry import build_model
from models.slot_pixel_part_graph_motif import SlotPixelPartGraphMotif

__all__ = [
    "ClassPixelMotifGraphRetrieval",
    "DualBranchGraphSwinMotifD7",
    "FixedMotifMLPClassifier",
    "SlotPixelPartGraphMotif",
    "build_model",
]
