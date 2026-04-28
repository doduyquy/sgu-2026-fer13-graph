"""Visualization for D5 class-level gates and sample attentions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.labels import EMOTION_NAMES, NUM_CLASSES
from training.trainer import move_to_device


def node_map(vec, height: int = 48, width: int = 48) -> np.ndarray:
    return torch.as_tensor(vec).detach().cpu().float().reshape(height, width).numpy()


def save_class_gate_heatmaps(
    model: torch.nn.Module,
    out_dir: str | Path,
    height: int = 48,
    width: int = 48,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        gate = torch.sigmoid(model.class_node_gate_logits).detach().cpu()
    for c in range(NUM_CLASSES):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(node_map(gate[c], height, width), cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"{c} {EMOTION_NAMES[c]} gate")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = out_dir / f"class_{c}_{EMOTION_NAMES[c].lower()}_gate.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)


def save_attention_grid(
    image,
    node_attn,
    label: int,
    pred: int,
    out_path: str | Path,
    height: int = 48,
    width: int = 48,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = node_map(image, height, width)
    attn = torch.as_tensor(node_attn).detach().cpu().float()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    axes[0].imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"image y={label} pred={pred}")
    axes[0].axis("off")
    for c in range(NUM_CLASSES):
        ax = axes[c + 1]
        ax.imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(node_map(attn[c], height, width), cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
        ax.set_title(f"{c} {EMOTION_NAMES[c]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_true_pred_attention(
    image,
    true_attn,
    pred_attn,
    label: int,
    pred: int,
    out_path: str | Path,
    height: int = 48,
    width: int = 48,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = node_map(image, height, width)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image")
    axes[1].imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(node_map(true_attn, height, width), cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"true {label} {EMOTION_NAMES[label]}")
    axes[2].imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(node_map(pred_attn, height, width), cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[2].set_title(f"pred {pred} {EMOTION_NAMES[pred]}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_top_edge_attention(
    image,
    edge_index,
    edge_attn,
    out_path: str | Path,
    height: int = 48,
    width: int = 48,
    top_k: int = 120,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = node_map(image, height, width)
    edge_index = torch.as_tensor(edge_index).detach().cpu().long()
    edge_attn = torch.as_tensor(edge_attn).detach().cpu().float()
    k = min(int(top_k), int(edge_attn.numel()))
    values, idx = torch.topk(edge_attn, k=k)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    vmax = float(values.max().item()) if values.numel() else 1.0
    for value, edge_id in zip(values.tolist(), idx.tolist()):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        sx, sy = src % width, src // width
        dx, dy = dst % width, dst // width
        alpha = 0.15 + 0.85 * (float(value) / max(vmax, 1e-6))
        ax.plot([sx, dx], [sy, dy], color="yellow", linewidth=0.6, alpha=alpha)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_sample_attention_maps(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: str | Path,
    max_samples: int = 16,
    height: int = 48,
    width: int = 48,
    include_edges: bool = True,
) -> None:
    out_dir = Path(out_dir)
    grid_dir = out_dir / "all_class_grids"
    pair_dir = out_dir / "true_pred"
    edge_dir = out_dir / "top_edges"
    grid_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    if include_edges:
        edge_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        preds = logits.argmax(dim=1)
        x = batch["x"]
        y = batch["y"]
        graph_id = batch["graph_id"]
        for i in range(x.shape[0]):
            if saved >= int(max_samples):
                return
            label = int(y[i].item())
            pred = int(preds[i].item())
            gid = int(graph_id[i].item())
            image = x[i, :, 0]
            node_attn = out["node_attn"][i]
            save_attention_grid(
                image=image,
                node_attn=node_attn,
                label=label,
                pred=pred,
                out_path=grid_dir / f"sample_{gid}_all_classes.png",
                height=height,
                width=width,
            )
            save_true_pred_attention(
                image=image,
                true_attn=node_attn[label],
                pred_attn=node_attn[pred],
                label=label,
                pred=pred,
                out_path=pair_dir / f"sample_{gid}_true_pred.png",
                height=height,
                width=width,
            )
            if include_edges:
                save_top_edge_attention(
                    image=image,
                    edge_index=batch["edge_index"],
                    edge_attn=out["edge_attn"][i, pred],
                    out_path=edge_dir / f"sample_{gid}_pred_edges.png",
                    height=height,
                    width=width,
                )
            saved += 1
