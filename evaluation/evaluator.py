"""Evaluation loop and output writers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from fer_d5.data.labels import EMOTION_NAMES, NUM_CLASSES
from fer_d5.evaluation.metrics import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_array,
)
from fer_d5.training.trainer import move_to_device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict:
    model.eval()
    y_true = []
    y_pred = []
    graph_ids = []
    scores = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        pred = logits.argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        graph_ids.extend(batch["graph_id"].detach().cpu().tolist())
        scores.extend(logits.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report_dict(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix_array(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["graph_id"] = graph_ids
    metrics["scores"] = scores
    return metrics


def save_confusion_matrix(cm: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_predictions_csv(metrics: Dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "y_true", "y_pred"] + [f"score_{i}" for i in range(NUM_CLASSES)])
        for gid, y_true, y_pred, score in zip(
            metrics["graph_id"],
            metrics["y_true"],
            metrics["y_pred"],
            metrics["scores"],
        ):
            writer.writerow([gid, y_true, y_pred] + list(score))
