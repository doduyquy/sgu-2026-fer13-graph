"""Visualize D8A Graph-Swin pre-part motif outputs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
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


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _warn_missing(field: str) -> None:
    warnings.warn(f"Missing field {field}; skipping related D8A visualization.", stacklevel=2)


def _plot_mask_grid(image: np.ndarray, masks: np.ndarray, out_path: Path, title: str) -> None:
    k = masks.shape[0]
    cols = int(math.ceil(math.sqrt(k + 1)))
    rows = int(math.ceil((k + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.1))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=8)
    for idx in range(k):
        axes[idx + 1].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx + 1].imshow(masks[idx], cmap="magma", alpha=0.65)
        axes[idx + 1].set_title(f"slot {idx:02d}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_top_masks(image: np.ndarray, masks: np.ndarray, out_path: Path, title: str, top_k: int = 8) -> None:
    area = masks.reshape(masks.shape[0], -1).mean(axis=1)
    top = np.argsort(area)[-min(top_k, masks.shape[0]) :][::-1]
    cols = len(top)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.0, 2.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, slot in zip(axes, top):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(masks[slot], cmap="magma", alpha=0.65)
        ax.set_title(f"slot {int(slot):02d} a={area[slot]:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_class_motifs(image: np.ndarray, maps: np.ndarray, out_path: Path, title: str) -> None:
    cols = 4
    rows = int(math.ceil(maps.shape[0] / cols))
    vmax = max(float(np.nanmax(maps)), 1e-8)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.4))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx in range(maps.shape[0]):
        axes[idx].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].imshow(maps[idx], cmap="magma", alpha=0.65, vmin=0.0, vmax=vmax)
        axes[idx].set_title(EMOTION_NAMES[idx], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_context_maps(
    image: np.ndarray,
    context_norm: np.ndarray,
    delta_norm: np.ndarray,
    out_path: Path,
    title: str,
    alpha: float,
    ratio: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=9)
    im1 = axes[1].imshow(context_norm, cmap="viridis")
    axes[1].set_title("context norm", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(delta_norm, cmap="magma")
    axes[2].set_title("enhanced - pixel", fontsize=9)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title} alpha={alpha:.4f} ratio={ratio:.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_slot_area_summary(area_values: list[np.ndarray], out_path: Path) -> None:
    if not area_values:
        _warn_missing("slot_area")
        return
    areas = np.concatenate(area_values, axis=0)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot", "mean_area", "std_area", "min_area", "max_area"])
        for idx in range(areas.shape[1]):
            col = areas[:, idx]
            writer.writerow([idx, float(col.mean()), float(col.std()), float(col.min()), float(col.max())])


def _write_slot_similarity(mask_sum: torch.Tensor | None, sample_count: int, out_path: Path) -> None:
    if mask_sum is None or sample_count <= 0:
        _warn_missing("part_masks")
        return
    avg_masks = _np(mask_sum / float(sample_count)).reshape(mask_sum.shape[0], -1)
    denom = np.linalg.norm(avg_masks, axis=1, keepdims=True).clip(min=1e-8)
    sim = (avg_masks / denom) @ (avg_masks / denom).T
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_i", "slot_j", "cosine_similarity"])
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                writer.writerow([i, j, float(sim[i, j])])


def _write_context_diagnostics(rows: list[dict], out_path: Path) -> None:
    if not rows:
        _warn_missing("h_context/enhanced_h_pixel")
        return
    keys = ["graph_id", "y_true", "y_pred", "context_alpha", "h_pixel_norm", "context_norm", "enhanced_norm", "context_to_pixel_ratio"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_class_part_attention(attn_sum: np.ndarray | None, count: int, out_path: Path) -> None:
    if attn_sum is None or count <= 0:
        _warn_missing("class_part_attn")
        return
    avg = attn_sum / float(count)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "slot", "attention"])
        for c in range(avg.shape[0]):
            for k in range(avg.shape[1]):
                writer.writerow([c, EMOTION_NAMES[c], k, float(avg[c, k])])


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
    graph_cfg = dict(config.get("graph", {}) or {})
    height = int(graph_cfg.get("height", 48))
    width = int(graph_cfg.get("width", 48))
    out_dir = _ensure_dir(output_root / "figures" / "d8a_prepart")

    area_values: list[np.ndarray] = []
    context_rows: list[dict] = []
    mask_sum = None
    sample_count = 0
    class_attn_sum = None
    class_attn_count = 0
    saved = 0

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits.float(), dim=1)
        bsz = logits.shape[0]

        part_masks = out.get("part_masks")
        if torch.is_tensor(part_masks):
            masks = part_masks.detach().float()
            batch_mask_sum = masks.sum(dim=0)
            mask_sum = batch_mask_sum if mask_sum is None else mask_sum + batch_mask_sum
            sample_count += int(masks.shape[0])
        else:
            masks = None
            _warn_missing("part_masks")

        slot_area = out.get("slot_area")
        if torch.is_tensor(slot_area):
            area_values.append(_np(slot_area))

        class_part_attn = out.get("class_part_attn")
        if torch.is_tensor(class_part_attn):
            cpa = _np(class_part_attn)
            class_attn_sum = cpa.sum(axis=0) if class_attn_sum is None else class_attn_sum + cpa.sum(axis=0)
            class_attn_count += int(cpa.shape[0])

        for i in range(bsz):
            graph_id = int(batch["graph_id"][i].item())
            y_true = int(batch["y"][i].item())
            y_pred = int(pred[i].item())
            image = _np(batch["x"][i, :, 0]).reshape(height, width)
            confidence = float(probs[i, y_pred].item())
            title = f"id={graph_id} true={EMOTION_NAMES[y_true]} pred={EMOTION_NAMES[y_pred]} conf={confidence:.2f}"

            h_pixel = out.get("h_pixel")
            h_context = out.get("h_context")
            enhanced = out.get("enhanced_h_pixel")
            alpha = out.get("context_alpha")
            if torch.is_tensor(h_pixel) and torch.is_tensor(h_context) and torch.is_tensor(enhanced):
                hp = h_pixel[i].detach().float()
                hc = h_context[i].detach().float()
                eh = enhanced[i].detach().float()
                pixel_norm = float(hp.norm(dim=-1).mean().cpu())
                context_norm = float(hc.norm(dim=-1).mean().cpu())
                enhanced_norm = float(eh.norm(dim=-1).mean().cpu())
                ratio = context_norm / max(pixel_norm, 1e-8)
                alpha_value = float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float("nan")
                context_rows.append(
                    {
                        "graph_id": graph_id,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "context_alpha": alpha_value,
                        "h_pixel_norm": pixel_norm,
                        "context_norm": context_norm,
                        "enhanced_norm": enhanced_norm,
                        "context_to_pixel_ratio": ratio,
                    }
                )
                if saved < int(max_samples):
                    _plot_context_maps(
                        image=image,
                        context_norm=_np(hc.norm(dim=-1)).reshape(height, width),
                        delta_norm=_np((eh - hp).norm(dim=-1)).reshape(height, width),
                        out_path=out_dir / f"context_sample_{saved:03d}_id_{graph_id}.png",
                        title=title,
                        alpha=alpha_value,
                        ratio=ratio,
                    )

            if saved < int(max_samples) and masks is not None:
                item_masks = _np(masks[i]).reshape(masks.shape[1], height, width)
                _plot_mask_grid(image, item_masks, out_dir / f"part_masks_sample_{saved:03d}_id_{graph_id}.png", title)
                _plot_top_masks(image, item_masks, out_dir / f"top_part_masks_sample_{saved:03d}_id_{graph_id}.png", title)

            class_pixel_motif = out.get("class_pixel_motif")
            if saved < int(max_samples) and torch.is_tensor(class_pixel_motif):
                maps = _np(class_pixel_motif[i]).reshape(-1, height, width)
                _plot_class_motifs(image, maps, out_dir / f"class_pixel_motif_sample_{saved:03d}_id_{graph_id}.png", title)

            part_attn = out.get("part_attn")
            if saved < int(max_samples) and torch.is_tensor(part_attn):
                attn = part_attn[i].detach().float()
                if attn.ndim == 3:
                    attn = attn.mean(dim=0)
                _plot_heatmap(_np(attn), out_dir / f"part_to_part_attention_sample_{saved:03d}_id_{graph_id}.png", title, cmap="Blues")

            if saved < int(max_samples) and torch.is_tensor(class_part_attn):
                _plot_heatmap(
                    _np(class_part_attn[i]),
                    out_dir / f"class_to_part_attention_sample_{saved:03d}_id_{graph_id}.png",
                    title,
                    cmap="viridis",
                )
            saved += 1
            if saved >= int(max_samples) and max_batches is None:
                break
        if saved >= int(max_samples) and max_batches is None:
            break

    _write_slot_area_summary(area_values, out_dir / "d8a_slot_area_summary.csv")
    _write_slot_similarity(mask_sum, sample_count, out_dir / "d8a_slot_similarity.csv")
    _write_context_diagnostics(context_rows, out_dir / "d8a_context_diagnostics.csv")
    _write_class_part_attention(class_attn_sum, class_attn_count, out_dir / "d8a_class_part_attention.csv")
    print(f"D8A pre-part figures and CSVs: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8a_graph_swin_prepart_d6b.yaml")
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
