"""Classification metrics for D5."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data.labels import EMOTION_NAMES


def compute_metrics(y_true, y_pred) -> Dict:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
    }


def classification_report_dict(y_true, y_pred, target_names: List[str] | None = None) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=target_names or EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_array(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(7)))
