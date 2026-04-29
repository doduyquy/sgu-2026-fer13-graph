"""Evaluation loop and output writers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.labels import EMOTION_NAMES, NUM_CLASSES
from evaluation.metrics import (
    classification_report_dict,
    compute_metrics,
    confusion_matrix_array,
)
from training.trainer import move_to_device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
    collect_examples: int = 10,
) -> Dict:
    model.eval()
    y_true = []
    y_pred = []
    graph_ids = []
    scores = []
    correct_examples = []
    wrong_examples = []
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
        if len(correct_examples) < collect_examples or len(wrong_examples) < collect_examples:
            x_cpu = batch["x"].detach().cpu()
            y_cpu = batch["y"].detach().cpu()
            pred_cpu = pred.detach().cpu()
            gid_cpu = batch["graph_id"].detach().cpu()
            prob_cpu = torch.softmax(logits.detach().cpu().float(), dim=1)
            for i in range(x_cpu.shape[0]):
                label = int(y_cpu[i].item())
                pred_label = int(pred_cpu[i].item())
                item = {
                    "graph_id": int(gid_cpu[i].item()),
                    "y_true": label,
                    "y_pred": pred_label,
                    "confidence": float(prob_cpu[i, pred_label].item()),
                    "image": x_cpu[i, :, 0].float().reshape(48, 48).numpy(),
                }
                if label == pred_label and len(correct_examples) < collect_examples:
                    correct_examples.append(item)
                elif label != pred_label and len(wrong_examples) < collect_examples:
                    wrong_examples.append(item)

    metrics = compute_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report_dict(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix_array(y_true, y_pred)
    metrics["pred_count"] = np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=NUM_CLASSES).tolist()
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["graph_id"] = graph_ids
    metrics["scores"] = scores
    metrics["correct_examples"] = correct_examples
    metrics["wrong_examples"] = wrong_examples
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


def save_example_grid(examples, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        return
    cols = 5
    rows = int(math.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, item in zip(axes, examples):
        image = np.asarray(item["image"], dtype=np.float32)
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        y_true = int(item["y_true"])
        y_pred = int(item["y_pred"])
        ax.set_title(
            f"id={item['graph_id']}\n"
            f"y={EMOTION_NAMES[y_true]}\n"
            f"p={EMOTION_NAMES[y_pred]} {item['confidence']:.2f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
