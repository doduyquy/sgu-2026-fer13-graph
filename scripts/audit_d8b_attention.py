"""Standalone attention/gate audit for D8B face-aware Graph-Swin.

This script is inference-only. It reads already exposed D8B output tensors,
projects them to 48x48 maps where possible, computes spatial audit metrics,
and writes CSV/figure artifacts for D8B-R1 decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_checkpoint_model, load_config, resolve_path, save_config  # noqa: E402
from data.labels import EMOTION_NAMES, NUM_CLASSES  # noqa: E402
from evaluation.metrics import classification_report_dict, compute_metrics, confusion_matrix_array  # noqa: E402
from training.trainer import move_to_device  # noqa: E402


METRIC_FIELDS = [
    "confidence",
    "border_mass",
    "outer_border_mass",
    "center_mass",
    "upper_mass",
    "middle_mass",
    "lower_mass",
    "upper_middle_lower_sum",
    "entropy",
    "normalized_entropy",
    "effective_attention_count",
    "top1_mass",
    "top5_percent_mass",
    "top20_percent_mass",
    "top10_border_mass",
    "top20_border_mass",
    "top20_outer_border_mass",
    "top20_upper_mass",
    "top20_middle_mass",
    "top20_lower_mass",
    "center_of_mass_x",
    "center_of_mass_y",
    "attention_std_x",
    "attention_std_y",
]

SUMMARY_FIELDS = [
    "sample_id",
    "label",
    "label_name",
    "pred",
    "pred_name",
    "correct",
    "confidence",
    "map_name",
    "border_mass",
    "outer_border_mass",
    "center_mass",
    "upper_mass",
    "middle_mass",
    "lower_mass",
    "upper_middle_lower_sum",
    "entropy",
    "normalized_entropy",
    "effective_attention_count",
    "top1_mass",
    "top5_percent_mass",
    "top20_percent_mass",
    "top10_border_mass",
    "top20_border_mass",
    "top20_outer_border_mass",
    "top20_upper_mass",
    "top20_middle_mass",
    "top20_lower_mass",
    "center_of_mass_x",
    "center_of_mass_y",
    "attention_std_x",
    "attention_std_y",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _parse_map_names(value: Optional[str], audit_cfg: Dict[str, Any]) -> Optional[list[str]]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    names = audit_cfg.get("map_names")
    if isinstance(names, str):
        return [item.strip() for item in names.split(",") if item.strip()]
    if isinstance(names, (list, tuple)):
        return [str(item).strip() for item in names if str(item).strip()]
    return None


def _make_masks(height: int, width: int, border_width: int, outer_border_width: int, tau: float) -> Dict[str, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width)
    border = np.zeros((height, width), dtype=np.float32)
    bw = max(0, int(border_width))
    if bw > 0:
        border[:bw, :] = 1.0
        border[-bw:, :] = 1.0
        border[:, :bw] = 1.0
        border[:, -bw:] = 1.0
    outer = np.zeros((height, width), dtype=np.float32)
    obw = max(0, int(outer_border_width))
    if obw > 0:
        outer[:obw, :] = 1.0
        outer[-obw:, :] = 1.0
        outer[:, :obw] = 1.0
        outer[:, -obw:] = 1.0
    tau = max(float(tau), 1e-8)
    upper = 1.0 / (1.0 + np.exp(-((0.42 - y) / tau)))
    middle = (1.0 / (1.0 + np.exp(-((y - 0.30) / tau)))) * (1.0 / (1.0 + np.exp(-((0.70 - y) / tau))))
    lower = 1.0 / (1.0 + np.exp(-((y - 0.55) / tau)))
    left = 1.0 / (1.0 + np.exp(-((0.42 - x) / tau)))
    center_x = (1.0 / (1.0 + np.exp(-((x - 0.30) / tau)))) * (1.0 / (1.0 + np.exp(-((0.70 - x) / tau))))
    right = 1.0 / (1.0 + np.exp(-((x - 0.55) / tau)))
    return {
        "border": border,
        "outer_border": outer,
        "center": 1.0 - border,
        "upper": np.repeat(upper, width, axis=1).astype(np.float32),
        "middle": np.repeat(middle, width, axis=1).astype(np.float32),
        "lower": np.repeat(lower, width, axis=1).astype(np.float32),
        "left": np.repeat(left, height, axis=0).astype(np.float32),
        "center_x": np.repeat(center_x, height, axis=0).astype(np.float32),
        "right": np.repeat(right, height, axis=0).astype(np.float32),
    }


def _grid_shape(count: int) -> Optional[tuple[int, int]]:
    side = int(math.sqrt(int(count)))
    if side * side == int(count):
        return side, side
    return None


def _resize_grid_map(values: torch.Tensor, height: int, width: int, mode: str = "nearest") -> torch.Tensor:
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim != 4:
        raise ValueError(f"Expected [B,1,H,W] map, got {tuple(values.shape)}")
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(values.float(), size=(height, width), mode=mode, **kwargs).squeeze(1)


def _vector_to_spatial(values: torch.Tensor, height: int, width: int, mode: str = "nearest") -> Optional[torch.Tensor]:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2:
        return None
    shape = _grid_shape(int(values.shape[1]))
    if shape is None:
        return None
    grid = values.reshape(values.shape[0], 1, shape[0], shape[1])
    return _resize_grid_map(grid, height=height, width=width, mode=mode)


def _pixel_vector_to_spatial(values: torch.Tensor, height: int, width: int) -> Optional[torch.Tensor]:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2 or values.shape[1] != height * width:
        return None
    return values.reshape(values.shape[0], height, width)


def _window_attn_to_spatial(values: torch.Tensor, height: int, width: int, window_size: int, shift_size: int = 0) -> Optional[torch.Tensor]:
    if values.ndim != 3:
        return None
    bsz, num_windows, window_area = values.shape
    if int(window_size) <= 0 or window_area != int(window_size) * int(window_size):
        return None
    win_h = height // int(window_size)
    win_w = width // int(window_size)
    if win_h * win_w != num_windows:
        return None
    grid = values.reshape(bsz, win_h, win_w, window_size, window_size)
    spatial = grid.permute(0, 1, 3, 2, 4).contiguous().reshape(bsz, height, width)
    if int(shift_size) != 0:
        spatial = torch.roll(spatial, shifts=(int(shift_size), int(shift_size)), dims=(1, 2))
    return spatial


def _extract_maps(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pred: torch.Tensor,
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    requested: Optional[Iterable[str]] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    requested_set = set(requested) if requested else None
    maps: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    def want(name: str) -> bool:
        return requested_set is None or name in requested_set

    def add(name: str, value: Optional[torch.Tensor], source: str, method: str) -> None:
        if not want(name) or value is None:
            return
        maps[name] = value
        meta[name] = {"source": source, "source_shape": list(value.shape), "spatial_method": method}

    pixel_gate = out.get("pixel_gate")
    if torch.is_tensor(pixel_gate):
        add("pixel_gate", _pixel_vector_to_spatial(pixel_gate, height, width), "pixel_gate", "reshape 2304 nodes to 48x48")

    window_gate = out.get("window_gate")
    if torch.is_tensor(window_gate):
        spatial = _vector_to_spatial(window_gate, height, width, mode="nearest")
        add("window_gate", spatial, "window_gate", "sqrt grid nearest-upsample to 48x48")

    region_gate = out.get("region_gate")
    if torch.is_tensor(region_gate):
        spatial = _vector_to_spatial(region_gate, height, width, mode="nearest")
        add("region_gate", spatial, "region_gate", "sqrt grid nearest-upsample to 48x48")

    class_region_attn = out.get("class_region_attn")
    if torch.is_tensor(class_region_attn):
        labels = batch["y"].long()
        batch_idx = torch.arange(class_region_attn.shape[0], device=class_region_attn.device)
        true_vec = class_region_attn[batch_idx, labels]
        pred_vec = class_region_attn[batch_idx, pred.long()]
        mean_vec = class_region_attn.mean(dim=1)
        add(
            "class_region_attn_true",
            _vector_to_spatial(true_vec, height, width, mode="nearest"),
            "class_region_attn[label]",
            "true-class region vector nearest-upsample to 48x48",
        )
        add(
            "class_region_attn_pred",
            _vector_to_spatial(pred_vec, height, width, mode="nearest"),
            "class_region_attn[pred]",
            "predicted-class region vector nearest-upsample to 48x48",
        )
        add(
            "class_region_attn_mean",
            _vector_to_spatial(mean_vec, height, width, mode="nearest"),
            "class_region_attn.mean(class)",
            "mean class-region vector nearest-upsample to 48x48",
        )

    regular_window_attn = out.get("regular_window_attn")
    if torch.is_tensor(regular_window_attn):
        add(
            "regular_window_attn",
            _window_attn_to_spatial(regular_window_attn, height, width, window_size),
            "regular_window_attn",
            "reassemble window-local attention to 48x48",
        )

    shifted_window_attn = out.get("shifted_window_attn")
    if torch.is_tensor(shifted_window_attn):
        add(
            "shifted_window_attn",
            _window_attn_to_spatial(shifted_window_attn, height, width, window_size, shift_size=shift_size),
            "shifted_window_attn",
            "reassemble shifted windows then roll back to original frame",
        )

    return maps, meta


def _normalize_map(raw_map: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.asarray(raw_map, dtype=np.float32)
    method = "nonnegative raw map"
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        method = "nan/inf replaced with 0"
    if float(values.min(initial=0.0)) < 0.0:
        values = np.abs(values)
        method = "absolute value because map contained negatives"
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    p = values / (total + eps)
    vmin = float(values.min(initial=0.0))
    vmax = float(values.max(initial=0.0))
    vis = (values - vmin) / max(vmax - vmin, eps)
    return p, vis, method


def _top_region_mass(p: np.ndarray, raw: np.ndarray, mask: np.ndarray, fraction: float, eps: float = 1e-8) -> float:
    flat_raw = raw.reshape(-1)
    flat_p = p.reshape(-1)
    flat_mask = mask.reshape(-1)
    k = max(1, int(math.ceil(float(fraction) * flat_raw.size)))
    top_idx = np.argpartition(flat_raw, flat_raw.size - k)[flat_raw.size - k :]
    denom = float(flat_p[top_idx].sum())
    return float((flat_p[top_idx] * flat_mask[top_idx]).sum() / max(denom, eps))


def _top_mass(p: np.ndarray, fraction: float) -> float:
    flat = p.reshape(-1)
    k = max(1, int(math.ceil(float(fraction) * flat.size)))
    top_idx = np.argpartition(flat, flat.size - k)[flat.size - k :]
    return float(flat[top_idx].sum())


def _compute_map_metrics(p: np.ndarray, raw: np.ndarray, masks: Dict[str, np.ndarray], eps: float = 1e-8) -> Dict[str, float]:
    height, width = p.shape
    entropy = float(-(p * np.log(np.clip(p, eps, None))).sum())
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    cx = float((p * xx).sum())
    cy = float((p * yy).sum())
    std_x = float(np.sqrt(max(float((p * (xx - cx) ** 2).sum()), 0.0)))
    std_y = float(np.sqrt(max(float((p * (yy - cy) ** 2).sum()), 0.0)))
    upper = float((p * masks["upper"]).sum())
    middle = float((p * masks["middle"]).sum())
    lower = float((p * masks["lower"]).sum())
    return {
        "border_mass": float((p * masks["border"]).sum()),
        "outer_border_mass": float((p * masks["outer_border"]).sum()),
        "center_mass": float((p * masks["center"]).sum()),
        "upper_mass": upper,
        "middle_mass": middle,
        "lower_mass": lower,
        "upper_middle_lower_sum": upper + middle + lower,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(height * width),
        "effective_attention_count": float(math.exp(entropy)),
        "top1_mass": float(p.max(initial=0.0)),
        "top5_percent_mass": _top_mass(p, 0.05),
        "top20_percent_mass": _top_mass(p, 0.20),
        "top10_border_mass": _top_region_mass(p, raw, masks["border"], 0.10),
        "top20_border_mass": _top_region_mass(p, raw, masks["border"], 0.20),
        "top20_outer_border_mass": _top_region_mass(p, raw, masks["outer_border"], 0.20),
        "top20_upper_mass": _top_region_mass(p, raw, masks["upper"], 0.20),
        "top20_middle_mass": _top_region_mass(p, raw, masks["middle"], 0.20),
        "top20_lower_mass": _top_region_mass(p, raw, masks["lower"], 0.20),
        "center_of_mass_x": cx,
        "center_of_mass_y": cy,
        "attention_std_x": std_x,
        "attention_std_y": std_y,
    }


def _write_csv(rows: list[dict], out_path: Path, fields: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _aggregate(rows: list[dict], group_fields: list[str]) -> list[dict]:
    groups: Dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    out_rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        item = {field: value for field, value in zip(group_fields, key)}
        item["count"] = len(group)
        item["accuracy"] = float(np.mean([1.0 if _bool(row.get("correct")) else 0.0 for row in group]))
        for field in METRIC_FIELDS:
            values = np.asarray([float(row.get(field, float("nan"))) for row in group], dtype=np.float64)
            item[f"{field}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
            item[f"{field}_std"] = float(np.nanstd(values)) if values.size else float("nan")
        out_rows.append(item)
    return out_rows


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stats_fields(group_fields: list[str]) -> list[str]:
    fields = list(group_fields) + ["count", "accuracy"]
    for field in METRIC_FIELDS:
        fields.extend([f"{field}_mean", f"{field}_std"])
    return fields


def _image_minmax(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    return (image - lo) / max(hi - lo, 1e-8)


def _plot_heatmap(heatmap: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overlay(image: np.ndarray, heatmap: np.ndarray, out_path: Path, title: str, top20: bool = False) -> None:
    image = _image_minmax(image)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(heatmap, cmap="magma", alpha=0.60, vmin=0.0, vmax=1.0)
    if top20:
        flat = heatmap.reshape(-1)
        k = max(1, int(math.ceil(0.20 * flat.size)))
        idx = np.argpartition(flat, flat.size - k)[flat.size - k :]
        top = np.zeros_like(flat, dtype=np.float32)
        top[idx] = 1.0
        top = top.reshape(heatmap.shape)
        ax.contour(top, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_combined_grid(items: list[dict], out_path: Path) -> None:
    if not items:
        return
    rows = len(items)
    fig, axes = plt.subplots(rows, 4, figsize=(10.5, max(2.2 * rows, 3.0)))
    axes = np.asarray(axes).reshape(rows, 4)
    for row_idx, item in enumerate(items):
        image = _image_minmax(item["image"])
        heatmap = item["vis_map"]
        top = np.zeros_like(heatmap, dtype=np.float32)
        flat = heatmap.reshape(-1)
        k = max(1, int(math.ceil(0.20 * flat.size)))
        idx = np.argpartition(flat, flat.size - k)[flat.size - k :]
        top.reshape(-1)[idx] = 1.0
        title = (
            f"id={item['sample_id']} {item['map_name']}\n"
            f"y={EMOTION_NAMES[item['label']]} p={EMOTION_NAMES[item['pred']]} "
            f"conf={item['confidence']:.2f}"
        )
        axes[row_idx, 0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(title, fontsize=7)
        axes[row_idx, 1].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title("heatmap", fontsize=7)
        axes[row_idx, 2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].imshow(heatmap, cmap="magma", alpha=0.60, vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title("overlay", fontsize=7)
        axes[row_idx, 3].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 3].contour(top, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[row_idx, 3].set_title("top20", fontsize=7)
        for ax in axes[row_idx]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _select_visual_items(rows: list[dict], visual_store: Dict[tuple[int, str], dict], max_visualizations: int) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    used: set[tuple[int, str]] = set()

    def add_candidates(candidates: Iterable[dict], reason: str, limit: int) -> None:
        added = 0
        for row in candidates:
            key = (int(row["sample_id"]), str(row["map_name"]))
            if key in used or key not in visual_store:
                continue
            item = dict(visual_store[key])
            item["reason"] = reason
            selected.append(item)
            used.add(key)
            added += 1
            if len(selected) >= int(max_visualizations) or added >= int(limit):
                return

    pixel_rows = [row for row in rows if row["map_name"] == "pixel_gate"] or rows
    add_candidates((row for row in pixel_rows if _bool(row["correct"])), "first_correct", 6)
    add_candidates((row for row in pixel_rows if not _bool(row["correct"])), "first_wrong", 6)
    add_candidates(sorted(rows, key=lambda r: float(r["top20_border_mass"]), reverse=True), "high_top20_border", 8)
    add_candidates(
        sorted((row for row in rows if not _bool(row["correct"])), key=lambda r: float(r["confidence"]), reverse=True),
        "high_conf_wrong",
        8,
    )
    for cls in range(NUM_CLASSES):
        add_candidates((row for row in pixel_rows if int(row["label"]) == cls), f"class_{EMOTION_NAMES[cls]}", 1)
    return selected[: int(max_visualizations)], _make_hard_cases(rows)


def _make_hard_cases(rows: list[dict], max_rows: int = 160) -> list[dict]:
    hard: list[dict] = []
    key_seen: set[tuple] = set()

    def add(row: dict, reason: str) -> None:
        key = (row["sample_id"], row["map_name"], reason)
        if key in key_seen:
            return
        item = dict(row)
        item["reason"] = reason
        hard.append(item)
        key_seen.add(key)

    expressive = {0, 2, 4, 5}
    for row in sorted(rows, key=lambda r: float(r["confidence"]), reverse=True):
        if not _bool(row["correct"]) and float(row["confidence"]) >= 0.50:
            add(row, "wrong_high_confidence")
    for row in sorted(rows, key=lambda r: float(r["top20_border_mass"]), reverse=True):
        if float(row["top20_border_mass"]) >= 0.35:
            add(row, "high_top20_border")
    for row in sorted(rows, key=lambda r: float(r["border_mass"]), reverse=True):
        if float(row["border_mass"]) >= 0.20:
            add(row, "high_border_mass")
    for row in sorted(rows, key=lambda r: float(r["top20_upper_mass"])):
        if int(row["label"]) in expressive and float(row["top20_upper_mass"]) < 0.25:
            add(row, "low_upper_expressive_class")
    return hard[:max_rows]


def _save_visualizations(selected: list[dict], figures_dir: Path) -> None:
    if not selected:
        return
    for idx, item in enumerate(selected):
        prefix = f"sample_{int(item['sample_id']):06d}_{item['map_name']}_{idx:03d}"
        title = (
            f"{item['reason']} id={item['sample_id']} {item['map_name']}\n"
            f"true={EMOTION_NAMES[item['label']]} pred={EMOTION_NAMES[item['pred']]} "
            f"conf={item['confidence']:.2f}"
        )
        _plot_heatmap(item["vis_map"], figures_dir / f"{prefix}_heatmap.png", title)
        _plot_overlay(item["image"], item["vis_map"], figures_dir / f"{prefix}_overlay.png", title)
        _plot_overlay(item["image"], item["vis_map"], figures_dir / f"{prefix}_top20_overlay.png", title, top20=True)
    _plot_combined_grid(selected[: min(len(selected), 20)], figures_dir / "combined_grid.png")


def _write_metrics_json(rows: list[dict], output_dir: Path) -> None:
    first_by_sample: Dict[int, dict] = {}
    for row in rows:
        sid = int(row["sample_id"])
        first_by_sample.setdefault(sid, row)
    ordered = [first_by_sample[key] for key in sorted(first_by_sample)]
    y_true = [int(row["label"]) for row in ordered]
    y_pred = [int(row["pred"]) for row in ordered]
    metrics = compute_metrics(y_true, y_pred)
    data = {
        **metrics,
        "num_samples": len(ordered),
        "pred_count": np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=NUM_CLASSES).tolist(),
        "classification_report": classification_report_dict(y_true, y_pred),
        "confusion_matrix": confusion_matrix_array(y_true, y_pred).tolist(),
    }
    with (output_dir / "audit_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_map_metadata(meta: Dict[str, Dict[str, Any]], normalizers: Dict[str, str], output_dir: Path, args: argparse.Namespace) -> None:
    data = {
        "maps": {name: {**dict(meta.get(name, {})), "normalization": normalizers.get(name, "")} for name in sorted(meta)},
        "topk_definition": "Top-k region metrics use attention mass fraction inside the selected top-value pixels.",
        "border_width": int(args.border_width),
        "outer_border_width": int(args.outer_border_width),
        "region_tau": float(args.region_tau),
    }
    with (output_dir / "map_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@torch.no_grad()
def run_audit(config: Dict[str, Any], args: argparse.Namespace) -> None:
    audit_cfg = dict(config.get("audit", {}) or {})
    checkpoint = args.checkpoint or audit_cfg.get("checkpoint")
    if checkpoint is None:
        checkpoint = "output/d8b_face_aware_graph_swin_border020/checkpoints/best.pth"
    graph_repo_path = args.graph_repo_path or audit_cfg.get("graph_repo_path") or "artifacts/graph_repo"
    output_dir = resolve_path(args.output_dir or audit_cfg.get("output_dir") or "outputs/d8b_attention_audit_border020")
    if output_dir is None:
        raise ValueError("output_dir resolved to None")
    split = args.split or audit_cfg.get("split") or "val"
    max_batches = args.max_batches if args.max_batches is not None else audit_cfg.get("max_batches")
    max_visualizations = (
        args.max_visualizations if args.max_visualizations is not None else int(audit_cfg.get("max_visualizations", 32))
    )
    map_names = _parse_map_names(args.map_names, audit_cfg)

    config = dict(config)
    config.setdefault("paths", {})["graph_repo_path"] = str(graph_repo_path)
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    if args.device is not None:
        config.setdefault("training", {})["device"] = str(args.device)

    output_dir = _ensure_dir(output_dir)
    figures_dir = _ensure_dir(output_dir / "figures")
    save_config(config, output_dir)

    model, device, checkpoint_data = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split=split, shuffle=False)

    graph_cfg = dict(config.get("graph", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    swin_cfg = dict(model_cfg.get("graph_swin", {}) or {})
    height = int(model_cfg.get("height", graph_cfg.get("height", 48)))
    width = int(model_cfg.get("width", graph_cfg.get("width", 48)))
    window_size = int(swin_cfg.get("window_size", 6))
    shift_size = int(swin_cfg.get("shift_size", window_size // 2))
    masks = _make_masks(height, width, int(args.border_width), int(args.outer_border_width), float(args.region_tau))

    rows: list[dict] = []
    visual_store: Dict[tuple[int, str], dict] = {}
    all_meta: Dict[str, Dict[str, Any]] = {}
    normalizers: Dict[str, str] = {}
    npz_maps: Dict[str, list[np.ndarray]] = defaultdict(list)
    npz_ids: list[int] = []
    total_samples = 0

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        probs = torch.softmax(logits.detach().float(), dim=1)
        pred = logits.argmax(dim=1)
        maps, meta = _extract_maps(
            out=out,
            batch=batch,
            pred=pred,
            height=height,
            width=width,
            window_size=window_size,
            shift_size=shift_size,
            requested=map_names,
        )
        all_meta.update(meta)
        bsz = int(logits.shape[0])
        total_samples += bsz

        images = _to_numpy(batch["x"][:, :, 0]).reshape(bsz, height, width)
        labels = batch["y"].detach().cpu().long().numpy()
        preds = pred.detach().cpu().long().numpy()
        graph_ids = batch["graph_id"].detach().cpu().long().numpy()
        confs = probs[torch.arange(bsz, device=probs.device), pred].detach().cpu().numpy()

        if args.save_npz:
            npz_ids.extend([int(x) for x in graph_ids.tolist()])

        for map_name, map_tensor in maps.items():
            map_np = _to_numpy(map_tensor)
            if args.save_npz:
                npz_maps[map_name].append(map_np.astype(np.float32))
            for i in range(bsz):
                p, vis, norm_method = _normalize_map(map_np[i])
                normalizers.setdefault(map_name, norm_method)
                metrics = _compute_map_metrics(p, map_np[i], masks)
                sid = int(graph_ids[i])
                label = int(labels[i])
                y_pred = int(preds[i])
                row = {
                    "sample_id": sid,
                    "label": label,
                    "label_name": EMOTION_NAMES[label] if 0 <= label < len(EMOTION_NAMES) else str(label),
                    "pred": y_pred,
                    "pred_name": EMOTION_NAMES[y_pred] if 0 <= y_pred < len(EMOTION_NAMES) else str(y_pred),
                    "correct": bool(label == y_pred),
                    "confidence": float(confs[i]),
                    "map_name": map_name,
                    **metrics,
                }
                rows.append(row)
                visual_store[(sid, map_name)] = {
                    "sample_id": sid,
                    "label": label,
                    "pred": y_pred,
                    "correct": bool(label == y_pred),
                    "confidence": float(confs[i]),
                    "map_name": map_name,
                    "image": images[i],
                    "vis_map": vis,
                    "raw_map": map_np[i],
                }

        if (batch_idx + 1) % 5 == 0:
            print(f"[Audit] processed_batches={batch_idx + 1} samples={total_samples} rows={len(rows)}")

    if not rows:
        raise RuntimeError("No attention/gate maps were audited. Check D8B output fields or --map_names.")

    _write_csv(rows, output_dir / "audit_summary.csv", SUMMARY_FIELDS)
    classwise = _aggregate(rows, ["map_name", "label", "label_name"])
    correct_wrong = _aggregate(rows, ["map_name", "correct"])
    pred_stats = _aggregate(rows, ["map_name", "pred", "pred_name"])
    _write_csv(classwise, output_dir / "classwise_stats.csv", _stats_fields(["map_name", "label", "label_name"]))
    _write_csv(correct_wrong, output_dir / "correct_vs_wrong_stats.csv", _stats_fields(["map_name", "correct"]))
    _write_csv(pred_stats, output_dir / "pred_class_stats.csv", _stats_fields(["map_name", "pred", "pred_name"]))

    selected, hard_cases = _select_visual_items(rows, visual_store, int(max_visualizations))
    hard_fields = SUMMARY_FIELDS + ["reason"]
    _write_csv(hard_cases, output_dir / "hard_cases.csv", hard_fields)
    _save_visualizations(selected, figures_dir)
    _write_metrics_json(rows, output_dir)
    _write_map_metadata(all_meta, normalizers, output_dir, args)

    if args.save_npz:
        arrays = {name: np.concatenate(parts, axis=0) for name, parts in npz_maps.items() if parts}
        arrays["sample_id"] = np.asarray(npz_ids, dtype=np.int64)
        np.savez_compressed(output_dir / "attention_maps.npz", **arrays)

    print(f"[Audit] checkpoint={checkpoint}")
    print(f"[Audit] split={split} samples={total_samples} maps={sorted({row['map_name'] for row in rows})}")
    print(f"[Audit] output_dir={output_dir}")
    if isinstance(checkpoint_data, dict):
        print(f"[Audit] checkpoint_epoch={checkpoint_data.get('epoch', 'unknown')} best_metric={checkpoint_data.get('best_metric', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8b_face_aware_graph_swin_border020.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_visualizations", type=int, default=None)
    parser.add_argument("--map_names", default=None, help="Comma-separated map names to audit.")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--border_width", type=int, default=4)
    parser.add_argument("--outer_border_width", type=int, default=6)
    parser.add_argument("--region_tau", type=float, default=0.05)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_audit(config, args)


if __name__ == "__main__":
    main()
