"""Visualize D8B face-aware Graph-Swin gates and region attention."""

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
    warnings.warn(f"Missing field {field}; skipping related D8B visualization.", stacklevel=2)


def _grid_shape(count: int) -> tuple[int, int] | None:
    side = int(math.sqrt(int(count)))
    if side * side == int(count):
        return side, side
    return None


def _plot_heatmap(matrix: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_gate_overlay(image: np.ndarray, gate: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=9)
    im1 = axes[1].imshow(gate, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("pixel gate", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(gate, cmap="magma", alpha=0.65, vmin=0.0, vmax=1.0)
    axes[2].set_title("overlay", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_class_region_grid(avg_attn: np.ndarray, out_path: Path, title: str) -> None:
    grid_shape = _grid_shape(avg_attn.shape[1])
    if grid_shape is None:
        _plot_heatmap(avg_attn, out_path, title, cmap="magma")
        return
    cols = 4
    rows = int(math.ceil(avg_attn.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.2))
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


def _write_summary(rows: list[dict], out_path: Path, keys: list[str]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_gate_grid_csv(values: np.ndarray, out_path: Path, prefix: str) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([prefix, "mean", "std", "min", "max"])
        flat = values.reshape(values.shape[0], -1)
        for idx in range(flat.shape[1]):
            col = flat[:, idx]
            writer.writerow([idx, float(col.mean()), float(col.std()), float(col.min()), float(col.max())])


def _write_class_region_attention(avg_attn: np.ndarray, out_dir: Path, top_k: int = 4) -> None:
    eps = 1e-8
    entropy = -(avg_attn * np.log(np.clip(avg_attn, eps, None))).sum(axis=1)
    grid_shape = _grid_shape(avg_attn.shape[1])
    grid_cols = grid_shape[1] if grid_shape is not None else None
    with (out_dir / "d8b_class_region_attention.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "region", "row", "col", "attention"])
        for c, name in enumerate(EMOTION_NAMES[: avg_attn.shape[0]]):
            for r in range(avg_attn.shape[1]):
                row = "" if grid_cols is None else int(r // grid_cols)
                col = "" if grid_cols is None else int(r % grid_cols)
                writer.writerow([c, name, r, row, col, float(avg_attn[c, r])])
    with (out_dir / "d8b_class_region_attention_entropy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "entropy", "max_prob"])
        for c, name in enumerate(EMOTION_NAMES[: avg_attn.shape[0]]):
            writer.writerow([c, name, float(entropy[c]), float(avg_attn[c].max())])
    with (out_dir / "d8b_top_regions_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "rank", "region", "row", "col", "attention"])
        for c, name in enumerate(EMOTION_NAMES[: avg_attn.shape[0]]):
            top = np.argsort(avg_attn[c])[-int(top_k) :][::-1]
            for rank, region in enumerate(top, start=1):
                row = "" if grid_cols is None else int(region // grid_cols)
                col = "" if grid_cols is None else int(region % grid_cols)
                writer.writerow([c, name, rank, int(region), row, col, float(avg_attn[c, region])])


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
    model_cfg = dict(config.get("model", {}) or {})
    swin_cfg = dict(model_cfg.get("graph_swin", {}) or {})
    height = int(model_cfg.get("height", graph_cfg.get("height", 48)))
    width = int(model_cfg.get("width", graph_cfg.get("width", 48)))
    window_size = int(swin_cfg.get("window_size", 6))
    win_h = height // window_size
    win_w = width // window_size
    out_dir = _ensure_dir(output_root / "figures" / "d8b_face_aware")

    diag_rows: list[dict] = []
    pixel_rows: list[dict] = []
    window_values: list[np.ndarray] = []
    region_values: list[np.ndarray] = []
    class_region_sum = None
    class_region_count = 0
    saved = 0
    correct_saved = 0
    wrong_saved = 0

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

        pixel_gate = out.get("pixel_gate")
        window_gate = out.get("window_gate")
        region_gate = out.get("region_gate")
        class_region_attn = out.get("class_region_attn")

        if torch.is_tensor(window_gate):
            window_values.append(_np(window_gate.squeeze(-1)))
        else:
            _warn_missing("window_gate")
        if torch.is_tensor(region_gate):
            region_values.append(_np(region_gate.squeeze(-1)))
        else:
            _warn_missing("region_gate")
        if torch.is_tensor(class_region_attn):
            attn_np = _np(class_region_attn)
            class_region_sum = attn_np.sum(axis=0) if class_region_sum is None else class_region_sum + attn_np.sum(axis=0)
            class_region_count += int(attn_np.shape[0])
        else:
            _warn_missing("class_region_attn")

        diag = out.get("diagnostics", {}) or {}
        for i in range(bsz):
            graph_id = int(batch["graph_id"][i].item())
            y_true = int(batch["y"][i].item())
            y_pred = int(pred[i].item())
            row = {"graph_id": graph_id, "y_true": y_true, "y_pred": y_pred}
            for key in (
                "pixel_gate_mean",
                "pixel_gate_std",
                "pixel_gate_border_mean",
                "pixel_gate_center_mean",
                "window_gate_mean",
                "region_gate_mean",
                "class_region_entropy_mean",
                "region_token_norm",
            ):
                value = diag.get(key)
                row[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float("nan")
            diag_rows.append(row)

            image = _np(batch["x"][i, :, 0]).reshape(height, width)
            title = (
                f"id={graph_id} true={EMOTION_NAMES[y_true]} pred={EMOTION_NAMES[y_pred]} "
                f"conf={float(probs[i, y_pred].item()):.2f}"
            )
            if torch.is_tensor(pixel_gate):
                gate = _np(pixel_gate[i, :, 0]).reshape(height, width)
                bw = int(config.get("loss", {}).get("pixel_gate_border_width", 3))
                border_mask = np.zeros((height, width), dtype=bool)
                if bw > 0:
                    border_mask[:bw, :] = True
                    border_mask[-bw:, :] = True
                    border_mask[:, :bw] = True
                    border_mask[:, -bw:] = True
                pixel_rows.append(
                    {
                        "graph_id": graph_id,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "mean": float(gate.mean()),
                        "std": float(gate.std()),
                        "min": float(gate.min()),
                        "max": float(gate.max()),
                        "border_mean": float(gate[border_mask].mean()) if border_mask.any() else float("nan"),
                        "center_mean": float(gate[~border_mask].mean()) if (~border_mask).any() else float("nan"),
                    }
                )
                if saved < int(max_samples):
                    _plot_gate_overlay(gate=gate, image=image, out_path=out_dir / f"pixel_gate_sample_{saved:03d}_id_{graph_id}.png", title=title)
            elif saved < int(max_samples):
                _warn_missing("pixel_gate")

            if saved < int(max_samples) and torch.is_tensor(window_gate):
                wg = _np(window_gate[i, :, 0])
                shape = (win_h, win_w) if wg.size == win_h * win_w else _grid_shape(wg.size)
                if shape is not None:
                    _plot_heatmap(wg.reshape(shape), out_dir / f"window_gate_grid_sample_{saved:03d}_id_{graph_id}.png", title, cmap="viridis")
            if saved < int(max_samples) and torch.is_tensor(region_gate):
                rg = _np(region_gate[i, :, 0])
                shape = _grid_shape(rg.size)
                if shape is not None:
                    _plot_heatmap(rg.reshape(shape), out_dir / f"region_gate_grid_sample_{saved:03d}_id_{graph_id}.png", title, cmap="cividis")
            if saved < int(max_samples) and torch.is_tensor(class_region_attn):
                _plot_class_region_grid(
                    _np(class_region_attn[i]),
                    out_dir / f"class_region_attention_sample_{saved:03d}_id_{graph_id}.png",
                    title,
                )

            if torch.is_tensor(pixel_gate) and (correct_saved < 4 or wrong_saved < 4):
                is_correct = y_true == y_pred
                if is_correct and correct_saved < 4:
                    _plot_gate_overlay(gate=gate, image=image, out_path=out_dir / f"correct_gate_sample_{correct_saved:03d}_id_{graph_id}.png", title=title)
                    correct_saved += 1
                elif (not is_correct) and wrong_saved < 4:
                    _plot_gate_overlay(gate=gate, image=image, out_path=out_dir / f"wrong_gate_sample_{wrong_saved:03d}_id_{graph_id}.png", title=title)
                    wrong_saved += 1
            saved += 1

    _write_summary(
        diag_rows,
        out_dir / "d8b_gate_diagnostics.csv",
        [
            "graph_id",
            "y_true",
            "y_pred",
            "pixel_gate_mean",
            "pixel_gate_std",
            "pixel_gate_border_mean",
            "pixel_gate_center_mean",
            "window_gate_mean",
            "region_gate_mean",
            "class_region_entropy_mean",
            "region_token_norm",
        ],
    )
    _write_summary(
        pixel_rows,
        out_dir / "d8b_pixel_gate_summary.csv",
        ["graph_id", "y_true", "y_pred", "mean", "std", "min", "max", "border_mean", "center_mean"],
    )
    if window_values:
        values = np.concatenate(window_values, axis=0)
        _write_gate_grid_csv(values, out_dir / "d8b_window_gate_summary.csv", "window")
        avg = values.mean(axis=0)
        shape = (win_h, win_w) if avg.size == win_h * win_w else _grid_shape(avg.size)
        if shape is not None:
            _plot_heatmap(avg.reshape(shape), out_dir / "window_gate_grid.png", "Average window gate", cmap="viridis")
    if region_values:
        values = np.concatenate(region_values, axis=0)
        _write_gate_grid_csv(values, out_dir / "d8b_region_gate_summary.csv", "region")
        avg = values.mean(axis=0)
        shape = _grid_shape(avg.size)
        if shape is not None:
            _plot_heatmap(avg.reshape(shape), out_dir / "region_gate_grid.png", "Average region gate", cmap="cividis")
    if class_region_sum is not None and class_region_count > 0:
        avg_attn = class_region_sum / float(class_region_count)
        _plot_heatmap(avg_attn, out_dir / "class_region_attention_grid.png", "Average class-region attention", cmap="magma")
        _plot_class_region_grid(avg_attn, out_dir / "class_region_attention_maps.png", "Average class-region maps")
        _write_class_region_attention(avg_attn, out_dir)
    print(f"D8B face-aware figures and CSVs: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8b_face_aware_graph_swin.yaml")
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
