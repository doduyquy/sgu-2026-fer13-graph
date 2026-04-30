"""Visualize D6A soft part slots and part-to-part attention."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    apply_cli_overrides,
    build_dataloader,
    load_checkpoint_model,
    load_config,
    output_root_from_checkpoint,
    resolve_path,
)
from data.labels import EMOTION_NAMES  # noqa: E402
from training.trainer import move_to_device  # noqa: E402


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _plot_part_mask_grid(
    image: np.ndarray,
    masks: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    k = masks.shape[0]
    cols = int(math.ceil(math.sqrt(k + 1)))
    rows = int(math.ceil((k + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.1))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=8)
    axes[0].axis("off")
    for idx in range(k):
        ax = axes[idx + 1]
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(masks[idx], cmap="magma", alpha=0.65)
        ax.set_title(f"slot {idx:02d}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("slot")
    ax.set_ylabel("slot")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_avg_slots(avg_masks: np.ndarray, out_dir: Path) -> None:
    k = avg_masks.shape[0]
    for idx in range(k):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(avg_masks[idx], cmap="magma")
        ax.set_title(f"avg slot {idx:02d}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_slot_{idx:02d}.png", dpi=160)
        plt.close(fig)

    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx in range(k):
        axes[idx].imshow(avg_masks[idx], cmap="magma")
        axes[idx].set_title(f"slot {idx:02d}", fontsize=8)
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_slot_grid.png", dpi=160)
    plt.close(fig)


def _cosine_similarity(flat_masks: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(flat_masks, axis=1, keepdims=True).clip(min=1e-8)
    normed = flat_masks / denom
    return normed @ normed.T


def _save_slot_area(area_values: list[np.ndarray], out_dir: Path) -> None:
    if not area_values:
        return
    areas = np.concatenate(area_values, axis=0)
    mean = areas.mean(axis=0)
    std = areas.std(axis=0)
    with (out_dir / "slot_area.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot", "mean", "std"])
        for idx, (m, s) in enumerate(zip(mean, std)):
            writer.writerow([idx, float(m), float(s)])
    with (out_dir / "slot_area.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(mean)), mean, yerr=std, capsize=2)
    ax.set_xlabel("slot")
    ax.set_ylabel("area mass")
    ax.set_title("Slot area distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "slot_area.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_visualize(
    config,
    checkpoint=None,
    split: str = "test",
    max_samples: int = 16,
    max_batches: int | None = None,
) -> None:
    paths = config.get("paths", {})
    if checkpoint is not None:
        output_root = output_root_from_checkpoint(checkpoint) or resolve_path(
            paths.get("resolved_output_root") or paths.get("output_root", "outputs")
        )
    else:
        output_root = resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"

    model, device, _ = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split=split, shuffle=False)
    graph_cfg = config.get("graph", {})
    height = int(graph_cfg.get("height", 48))
    width = int(graph_cfg.get("width", 48))

    fig_root = output_root / "figures"
    masks_dir = _ensure_dir(fig_root / "d6_part_masks")
    attn_dir = _ensure_dir(fig_root / "d6_part_attention")
    summary_dir = _ensure_dir(fig_root / "d6_slot_summary")

    mask_sum = None
    sample_count = 0
    saved_samples = 0
    saved_correct = 0
    saved_wrong = 0
    saved_attn = 0
    area_values: list[np.ndarray] = []

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        probs = torch.softmax(logits.float(), dim=1)
        pred = logits.argmax(dim=1)
        masks = out["part_masks"].detach().float()
        area_values.append(_to_numpy(out["slot_area"]))

        batch_mask_sum = masks.sum(dim=0)
        mask_sum = batch_mask_sum if mask_sum is None else mask_sum + batch_mask_sum
        sample_count += int(masks.shape[0])

        for i in range(masks.shape[0]):
            if saved_samples >= int(max_samples) and saved_correct >= int(max_samples // 2) and saved_wrong >= int(max_samples // 2):
                continue
            image = _to_numpy(batch["x"][i, :, 0]).reshape(height, width)
            item_masks = _to_numpy(masks[i]).reshape(masks.shape[1], height, width)
            y_true = int(batch["y"][i].item())
            y_pred = int(pred[i].item())
            confidence = float(probs[i, y_pred].item())
            graph_id = int(batch["graph_id"][i].item())
            title = (
                f"id={graph_id} true={EMOTION_NAMES[y_true]} "
                f"pred={EMOTION_NAMES[y_pred]} conf={confidence:.2f}"
            )
            if saved_samples < int(max_samples):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"sample_{saved_samples:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_samples += 1
            if y_true == y_pred and saved_correct < int(max_samples // 2):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"correct_{saved_correct:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_correct += 1
            elif y_true != y_pred and saved_wrong < int(max_samples // 2):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"wrong_{saved_wrong:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_wrong += 1

            part_attn = out.get("part_attn")
            if part_attn is not None and saved_attn < int(max_samples):
                attn = part_attn[i].detach().float()
                if attn.ndim == 3:
                    attn = attn.mean(dim=0)
                _plot_heatmap(
                    _to_numpy(attn),
                    attn_dir / f"part_attn_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title="part-to-part attention",
                    cmap="Blues",
                )
                saved_attn += 1

    if mask_sum is None or sample_count == 0:
        print("No samples were visualized.")
        return

    avg_masks = _to_numpy(mask_sum / float(sample_count)).reshape(-1, height, width)
    _plot_avg_slots(avg_masks, summary_dir)
    sim = _cosine_similarity(avg_masks.reshape(avg_masks.shape[0], -1))
    _plot_heatmap(sim, summary_dir / "slot_similarity.png", "Average slot mask cosine similarity", cmap="magma")
    _save_slot_area(area_values, summary_dir)

    print(f"D6 part masks: {masks_dir}")
    print(f"D6 part attention: {attn_dir}")
    print(f"D6 slot summary: {summary_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d6a_slot_pixel_part_graph_motif.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_visualize(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
