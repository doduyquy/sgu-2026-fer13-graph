"""Visualize D7 Graph-Swin region attention and class-level fusion gates."""

from __future__ import annotations

import argparse
import csv
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


def _plot_matrix(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _square_grid_shape(num_regions: int) -> tuple[int, int] | None:
    side = int(math.sqrt(int(num_regions)))
    if side * side == int(num_regions):
        return side, side
    return None


def _plot_class_region_grid(avg_attn: np.ndarray, out_path: Path, title: str) -> None:
    grid_shape = _square_grid_shape(avg_attn.shape[1])
    if grid_shape is None:
        cols = 2
        rows = int(math.ceil(avg_attn.shape[0] / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.2))
        axes = np.asarray(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for idx in range(avg_attn.shape[0]):
            axes[idx].bar(np.arange(avg_attn.shape[1]), avg_attn[idx])
            axes[idx].set_title(EMOTION_NAMES[idx], fontsize=8)
            axes[idx].set_xlabel("region")
            axes[idx].set_ylim(0.0, max(float(np.nanmax(avg_attn)), 1e-8))
            axes[idx].axis("on")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return
    cols = 4
    rows = int(math.ceil(avg_attn.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    vmax = max(float(np.nanmax(avg_attn)), 1e-8)
    for idx in range(avg_attn.shape[0]):
        axes[idx].imshow(avg_attn[idx].reshape(grid_shape), cmap="magma", vmin=0.0, vmax=vmax)
        axes[idx].set_title(EMOTION_NAMES[idx], fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_region_token_norm(region_norm: np.ndarray, out_path: Path) -> None:
    grid_shape = _square_grid_shape(region_norm.shape[0])
    if grid_shape is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(np.arange(region_norm.shape[0]), region_norm)
        ax.set_title("Average region token norm")
        ax.set_xlabel("region")
        ax.set_ylabel("norm")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(region_norm.reshape(grid_shape), cmap="cividis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Average region token norm")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_gate_by_class(gate: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(gate.shape[0]), gate)
    ax.set_xticks(np.arange(gate.shape[0]))
    ax.set_xticklabels(EMOTION_NAMES[: gate.shape[0]], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("D6 branch gate")
    ax.set_title("Average fusion gate by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_region_csvs(avg_attn: np.ndarray, out_dir: Path, top_k: int = 4) -> None:
    eps = 1e-8
    entropy = -(avg_attn * np.log(np.clip(avg_attn, eps, None))).sum(axis=1)
    grid_shape = _square_grid_shape(avg_attn.shape[1])
    grid_cols = grid_shape[1] if grid_shape is not None else None
    with (out_dir / "class_region_attention_entropy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "entropy", "max_prob"])
        for idx, name in enumerate(EMOTION_NAMES[: avg_attn.shape[0]]):
            writer.writerow([idx, name, float(entropy[idx]), float(avg_attn[idx].max())])
    with (out_dir / "top_regions_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "rank", "region", "row", "col", "attention"])
        for idx, name in enumerate(EMOTION_NAMES[: avg_attn.shape[0]]):
            top = np.argsort(avg_attn[idx])[-int(top_k):][::-1]
            for rank, region in enumerate(top, start=1):
                if grid_cols is None:
                    row, col = "", ""
                else:
                    row, col = int(region // grid_cols), int(region % grid_cols)
                writer.writerow([idx, name, rank, int(region), row, col, float(avg_attn[idx, region])])


def _write_gate_csvs(gate_by_class: np.ndarray, gate_samples: list[list[float]], out_dir: Path) -> None:
    with (out_dir / "fusion_gate_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "mean_gate_d6"])
        for idx, name in enumerate(EMOTION_NAMES[: gate_by_class.shape[0]]):
            writer.writerow([idx, name, float(gate_by_class[idx])])
    with (out_dir / "fusion_gate_by_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx"] + [f"gate_{name}" for name in EMOTION_NAMES[: gate_by_class.shape[0]]])
        for idx, row in enumerate(gate_samples):
            writer.writerow([idx] + [float(v) for v in row])


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
    out_dir = _ensure_dir(output_root / "figures" / "d7_graph_swin")

    class_region_sum = None
    class_region_true_sum = None
    class_true_count = np.zeros(len(EMOTION_NAMES), dtype=np.float64)
    region_norm_sum = None
    gate_sum = None
    gate_samples: list[list[float]] = []
    class_part_sum = None
    count = 0
    saved_samples = 0

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        class_region_attn = out.get("class_region_attn")
        region_tokens = out.get("region_tokens")
        fusion_gate = out.get("fusion_gate")
        class_part_attn = out.get("class_part_attn")
        if not torch.is_tensor(class_region_attn):
            continue
        attn_np = _to_numpy(class_region_attn)
        class_region_sum = attn_np.sum(axis=0) if class_region_sum is None else class_region_sum + attn_np.sum(axis=0)
        if class_region_true_sum is None:
            class_region_true_sum = np.zeros_like(attn_np[0])
        labels = batch["y"].detach().cpu().numpy().astype(int)
        for i, label in enumerate(labels):
            if 0 <= label < class_region_true_sum.shape[0]:
                class_region_true_sum[label] += attn_np[i, label]
                class_true_count[label] += 1.0

        if torch.is_tensor(region_tokens):
            norms = _to_numpy(region_tokens.norm(dim=-1))
            region_norm_sum = norms.sum(axis=0) if region_norm_sum is None else region_norm_sum + norms.sum(axis=0)

        if torch.is_tensor(fusion_gate):
            gate = _to_numpy(fusion_gate)
            if gate.ndim == 3:
                gate = gate.mean(axis=2)
            gate_sum = gate.sum(axis=0) if gate_sum is None else gate_sum + gate.sum(axis=0)
            remaining = max(0, int(max_samples) - len(gate_samples))
            gate_samples.extend(gate[:remaining].tolist())

        if torch.is_tensor(class_part_attn):
            cpa = _to_numpy(class_part_attn)
            class_part_sum = cpa.sum(axis=0) if class_part_sum is None else class_part_sum + cpa.sum(axis=0)

        if saved_samples < int(max_samples):
            probs = torch.softmax(out["logits"].detach().float(), dim=1)
            pred = out["logits"].argmax(dim=1).detach().cpu().numpy().astype(int)
            for i in range(min(attn_np.shape[0], int(max_samples) - saved_samples)):
                true_label = int(labels[i])
                pred_label = int(pred[i])
                title = (
                    f"id={int(batch['graph_id'][i].detach().cpu().item())} "
                    f"true={EMOTION_NAMES[true_label]} pred={EMOTION_NAMES[pred_label]} "
                    f"conf={float(probs[i, pred_label].detach().cpu().item()):.2f}"
                )
                _plot_class_region_grid(
                    attn_np[i],
                    out_dir / f"class_region_attention_sample_{saved_samples:03d}.png",
                    title,
                )
                saved_samples += 1

        count += int(attn_np.shape[0])

    if class_region_sum is None or count == 0:
        print("No D7 Graph-Swin attention tensors were found.")
        return

    avg_region = class_region_sum / float(count)
    _plot_matrix(
        avg_region,
        out_dir / "class_region_attention_grid.png",
        "Average class-to-region attention",
        xlabel="region",
        ylabel="emotion",
        cmap="viridis",
    )
    _plot_class_region_grid(avg_region, out_dir / "region_attention_maps.png", "Average region maps by class")
    true_den = np.maximum(class_true_count[:, None], 1.0)
    avg_by_true = class_region_true_sum / true_den
    _plot_matrix(
        avg_by_true,
        out_dir / "class_region_attention_avg_by_true_class.png",
        "Class-region attention by true class",
        xlabel="region",
        ylabel="emotion",
        cmap="magma",
    )
    _write_region_csvs(avg_region, out_dir)

    if region_norm_sum is not None:
        _plot_region_token_norm(region_norm_sum / float(count), out_dir / "region_token_grid.png")

    if gate_sum is not None:
        gate_by_class = gate_sum / float(count)
        _plot_gate_by_class(gate_by_class, out_dir / "fusion_gate_by_class.png")
        if gate_samples:
            _plot_matrix(
                np.asarray(gate_samples, dtype=np.float64),
                out_dir / "fusion_gate_by_sample.png",
                "Fusion gate by sample",
                xlabel="emotion",
                ylabel="sample",
                cmap="coolwarm",
            )
        _write_gate_csvs(gate_by_class, gate_samples, out_dir)

    if class_part_sum is not None:
        avg_class_part = class_part_sum / float(count)
        _plot_matrix(
            avg_class_part,
            out_dir / "d6_class_part_attention_in_d7.png",
            "D6 branch class-to-part attention inside D7",
            xlabel="part slot",
            ylabel="emotion",
            cmap="viridis",
        )

    print(f"D7 Graph-Swin visualizations: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d7c_dual_branch_gated_fusion.yaml")
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
