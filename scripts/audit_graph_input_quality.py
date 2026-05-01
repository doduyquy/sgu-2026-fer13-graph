"""Audit FER-2013 full pixel graph input quality.

This script reads an existing graph repository and writes statistics, figures,
and a markdown report. It does not train models, rebuild graphs, or modify the
graph builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_config, resolve_path
from data.full_graph_dataset import FullGraphDataset, collate_fn_full_graph
from data.labels import EMOTION_NAMES, NUM_CLASSES


NODE_FEATURE_NAMES = [
    "intensity",
    "x_norm",
    "y_norm",
    "gx",
    "gy",
    "grad_mag",
    "local_contrast",
]
EDGE_FEATURE_NAMES = [
    "dx",
    "dy",
    "dist",
    "delta_intensity",
    "intensity_similarity",
]
SPLITS = ("train", "val", "test")
PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
EPS = 1e-12


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def format_float(value: Any, digits: int = 6) -> str:
    value = to_float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}"


def safe_torch_load_shape(shape: Sequence[int]) -> str:
    return "x".join(str(int(v)) for v in shape)


def choose_indices(length: int, max_samples: Optional[int], seed: int) -> Optional[List[int]]:
    if max_samples is None or max_samples <= 0 or max_samples >= length:
        return None
    rng = random.Random(seed)
    indices = list(range(length))
    rng.shuffle(indices)
    return sorted(indices[: int(max_samples)])


class FeatureStatsAccumulator:
    """Streaming feature stats with bounded value reservoirs for percentiles."""

    def __init__(
        self,
        feature_names: Sequence[str],
        rng: np.random.Generator,
        reservoir_size: int = 200_000,
        batch_sample_size: int = 20_000,
    ) -> None:
        self.feature_names = list(feature_names)
        self.dim = len(self.feature_names)
        self.rng = rng
        self.reservoir_size = int(reservoir_size)
        self.batch_sample_size = int(batch_sample_size)

        self.total_count = np.zeros(self.dim, dtype=np.int64)
        self.count = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.abs_sum = np.zeros(self.dim, dtype=np.float64)
        self.zero_count = np.zeros(self.dim, dtype=np.int64)
        self.negative_count = np.zeros(self.dim, dtype=np.int64)
        self.nan_count = np.zeros(self.dim, dtype=np.int64)
        self.inf_count = np.zeros(self.dim, dtype=np.int64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.reservoirs: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(self.dim)]

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        if values is None:
            return
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        if values.numel() == 0:
            return
        values = values.detach().cpu().reshape(-1, values.shape[-1]).to(torch.float64)
        if values.shape[-1] != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {values.shape[-1]}")

        finite = torch.isfinite(values)
        isnan = torch.isnan(values)
        isinf = torch.isinf(values)
        self.total_count += np.full(self.dim, int(values.shape[0]), dtype=np.int64)
        self.nan_count += isnan.sum(dim=0).numpy().astype(np.int64)
        self.inf_count += isinf.sum(dim=0).numpy().astype(np.int64)
        self.count += finite.sum(dim=0).numpy().astype(np.int64)

        clean = torch.where(finite, values, torch.zeros_like(values))
        self.sum += clean.sum(dim=0).numpy()
        self.sumsq += (clean * clean).sum(dim=0).numpy()
        self.abs_sum += clean.abs().sum(dim=0).numpy()
        self.zero_count += ((values == 0) & finite).sum(dim=0).numpy().astype(np.int64)
        self.negative_count += ((values < 0) & finite).sum(dim=0).numpy().astype(np.int64)

        min_vals = torch.where(finite, values, torch.full_like(values, float("inf"))).min(dim=0).values.numpy()
        max_vals = torch.where(finite, values, torch.full_like(values, float("-inf"))).max(dim=0).values.numpy()
        self.min = np.minimum(self.min, min_vals)
        self.max = np.maximum(self.max, max_vals)

        self._update_reservoirs(values.to(torch.float32), finite)

    def _update_reservoirs(self, values: torch.Tensor, finite: torch.Tensor) -> None:
        for feature_idx in range(self.dim):
            col = values[:, feature_idx][finite[:, feature_idx]]
            if col.numel() == 0:
                continue
            if col.numel() > self.batch_sample_size:
                take = self.rng.choice(int(col.numel()), size=self.batch_sample_size, replace=False)
                col_np = col[take].numpy()
            else:
                col_np = col.numpy()
            merged = np.concatenate([self.reservoirs[feature_idx], col_np.astype(np.float32, copy=False)])
            if merged.size > self.reservoir_size:
                keep = self.rng.choice(merged.size, size=self.reservoir_size, replace=False)
                merged = merged[keep]
            self.reservoirs[feature_idx] = merged

    def final_rows(self, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, name in enumerate(self.feature_names):
            count = int(self.count[idx])
            mean = self.sum[idx] / max(count, 1)
            variance = max(self.sumsq[idx] / max(count, 1) - mean * mean, 0.0)
            std = math.sqrt(variance)
            sample = self.reservoirs[idx]
            pct = percentile_dict(sample)
            outlier_z3 = outlier_ratio(sample, mean, std, 3.0)
            outlier_z5 = outlier_ratio(sample, mean, std, 5.0)
            row = {
                **scope,
                "feature": name,
                "count": count,
                "mean": mean,
                "std": std,
                "min": self.min[idx] if np.isfinite(self.min[idx]) else float("nan"),
                "max": self.max[idx] if np.isfinite(self.max[idx]) else float("nan"),
                **pct,
                "zero_ratio": self.zero_count[idx] / max(count, 1),
                "negative_ratio": self.negative_count[idx] / max(count, 1),
                "abs_mean": self.abs_sum[idx] / max(count, 1),
                "outlier_ratio_z3": outlier_z3,
                "outlier_ratio_z5": outlier_z5,
                "nan_count": int(self.nan_count[idx]),
                "inf_count": int(self.inf_count[idx]),
                "reservoir_count": int(sample.size),
            }
            rows.append(row)
        return rows

    def mean_std_by_feature(self) -> Dict[str, Tuple[float, float]]:
        rows = self.final_rows({})
        return {str(row["feature"]): (to_float(row["mean"]), to_float(row["std"])) for row in rows}


class RowReservoir:
    """Bounded row reservoir for correlation analysis."""

    def __init__(
        self,
        dim: int,
        rng: np.random.Generator,
        reservoir_size: int = 200_000,
        batch_sample_size: int = 10_000,
    ) -> None:
        self.dim = int(dim)
        self.rng = rng
        self.reservoir_size = int(reservoir_size)
        self.batch_sample_size = int(batch_sample_size)
        self.rows = np.empty((0, self.dim), dtype=np.float32)

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        if values is None:
            return
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        if values.numel() == 0:
            return
        values = values.detach().cpu().reshape(-1, values.shape[-1])
        if values.shape[-1] != self.dim:
            return
        finite = torch.isfinite(values).all(dim=1)
        values = values[finite]
        if values.numel() == 0:
            return
        if values.shape[0] > self.batch_sample_size:
            take = self.rng.choice(int(values.shape[0]), size=self.batch_sample_size, replace=False)
            values = values[take]
        arr = values.to(torch.float32).numpy()
        merged = np.concatenate([self.rows, arr], axis=0)
        if merged.shape[0] > self.reservoir_size:
            keep = self.rng.choice(merged.shape[0], size=self.reservoir_size, replace=False)
            merged = merged[keep]
        self.rows = merged

    def corr(self) -> np.ndarray:
        if self.rows.shape[0] < 2:
            return np.full((self.dim, self.dim), np.nan, dtype=np.float64)
        return np.corrcoef(self.rows.astype(np.float64), rowvar=False)


class SampleReservoir:
    """Keep a small random sample of complete graph tensors for figures."""

    def __init__(self, per_class: int, rng: np.random.Generator) -> None:
        self.per_class = int(per_class)
        self.rng = rng
        self.counts = Counter()
        self.samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def update(self, split: str, graph_id: int, label: int, x: torch.Tensor) -> None:
        label = int(label)
        self.counts[label] += 1
        seen = self.counts[label]
        item = {
            "split": split,
            "graph_id": int(graph_id),
            "label": label,
            "x": x.detach().cpu().to(torch.float32).clone(),
        }
        bucket = self.samples[label]
        if len(bucket) < self.per_class:
            bucket.append(item)
            return
        j = int(self.rng.integers(0, seen))
        if j < self.per_class:
            bucket[j] = item

    def all_samples(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for label in sorted(self.samples):
            result.extend(self.samples[label])
        return result


def percentile_dict(sample: np.ndarray) -> Dict[str, float]:
    if sample.size == 0:
        return {percentile_key(p): float("nan") for p in PERCENTILES}
    values = np.percentile(sample.astype(np.float64), PERCENTILES)
    return {percentile_key(p): float(v) for p, v in zip(PERCENTILES, values)}


def percentile_key(p: float) -> str:
    if float(p).is_integer():
        return f"p{int(p)}"
    return f"p{p}"


def outlier_ratio(sample: np.ndarray, mean: float, std: float, z: float) -> float:
    if sample.size == 0 or not np.isfinite(std) or std <= EPS:
        return 0.0
    return float(np.mean(np.abs((sample.astype(np.float64) - mean) / std) > z))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def make_masks(height: int = 48, width: int = 48, border_width: int = 3) -> Dict[str, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    border = (yy < border_width) | (yy >= height - border_width) | (xx < border_width) | (xx >= width - border_width)
    corner = (
        ((yy < border_width) & (xx < border_width))
        | ((yy < border_width) & (xx >= width - border_width))
        | ((yy >= height - border_width) & (xx < border_width))
        | ((yy >= height - border_width) & (xx >= width - border_width))
    )
    upper = yy < height // 2
    lower = yy >= height // 2
    left = xx < width // 2
    right = xx >= width // 2

    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ry = height * 0.30
    rx = width * 0.30
    ellipse = ((yy.float() - cy) / ry) ** 2 + ((xx.float() - cx) / rx) ** 2 <= 1.0
    rectangle = (
        (yy >= int(height * 0.25))
        & (yy < int(height * 0.75))
        & (xx >= int(width * 0.25))
        & (xx < int(width * 0.75))
    )
    center = ellipse | rectangle
    return {
        "border": border.reshape(-1),
        "center": center.reshape(-1),
        "corner": corner.reshape(-1),
        "upper": upper.reshape(-1),
        "lower": lower.reshape(-1),
        "left": left.reshape(-1),
        "right": right.reshape(-1),
    }


def safe_mask_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.full((values.shape[0],), float("nan"), dtype=values.dtype)
    return values[:, mask].mean(dim=1)


def compute_border_rows(
    split: str,
    batch: Dict[str, torch.Tensor],
    border_widths: Sequence[int],
    height: int = 48,
    width: int = 48,
) -> List[Dict[str, Any]]:
    x = batch["x"].detach().cpu().to(torch.float32)
    labels = batch["y"].detach().cpu().long()
    graph_ids = batch.get("graph_id", torch.arange(x.shape[0])).detach().cpu().long()
    rows: List[Dict[str, Any]] = []
    intensity = x[:, :, 0]
    grad_mag = x[:, :, 5]
    contrast = x[:, :, 6]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x_norm = (xx.reshape(-1).float() / max(width - 1, 1)).unsqueeze(0)
    y_norm = (yy.reshape(-1).float() / max(height - 1, 1)).unsqueeze(0)
    center_prior = torch.exp(-(((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2) / (2 * 0.22**2))).squeeze(0)

    activation = torch.clamp(intensity + grad_mag + contrast, min=0.0)
    activation_sum = activation.sum(dim=1).clamp_min(EPS)
    com_x = (activation * x_norm).sum(dim=1) / activation_sum
    com_y = (activation * y_norm).sum(dim=1) / activation_sum
    center_prior_mass = (activation * center_prior.unsqueeze(0)).sum(dim=1) / activation_sum

    for width_value in border_widths:
        masks = make_masks(height=height, width=width, border_width=int(width_value))
        border = masks["border"]
        center = masks["center"]
        corner = masks["corner"]
        high_contrast_threshold = contrast.mean(dim=1, keepdim=True) + contrast.std(dim=1, keepdim=True)
        border_high_contrast_ratio = ((contrast[:, border] > high_contrast_threshold).float()).mean(dim=1)

        values = {
            "mean_intensity_border": safe_mask_mean(intensity, border),
            "mean_intensity_center": safe_mask_mean(intensity, center),
            "mean_grad_mag_border": safe_mask_mean(grad_mag, border),
            "mean_grad_mag_center": safe_mask_mean(grad_mag, center),
            "mean_local_contrast_border": safe_mask_mean(contrast, border),
            "mean_local_contrast_center": safe_mask_mean(contrast, center),
            "mean_intensity_corner": safe_mask_mean(intensity, corner),
            "mean_intensity_upper": safe_mask_mean(intensity, masks["upper"]),
            "mean_intensity_lower": safe_mask_mean(intensity, masks["lower"]),
            "mean_intensity_left": safe_mask_mean(intensity, masks["left"]),
            "mean_intensity_right": safe_mask_mean(intensity, masks["right"]),
            "border_high_contrast_ratio": border_high_contrast_ratio,
            "center_prior_mass": center_prior_mass,
            "face_center_of_mass_x": com_x,
            "face_center_of_mass_y": com_y,
        }
        for sample_idx in range(x.shape[0]):
            row = {
                "split": split,
                "graph_id": int(graph_ids[sample_idx]),
                "label": int(labels[sample_idx]),
                "class_name": EMOTION_NAMES[int(labels[sample_idx])] if 0 <= int(labels[sample_idx]) < NUM_CLASSES else "INVALID",
                "border_width": int(width_value),
            }
            for key, tensor in values.items():
                row[key] = float(tensor[sample_idx])
            row["border_center_intensity_diff"] = row["mean_intensity_border"] - row["mean_intensity_center"]
            row["border_center_grad_mag_diff"] = row["mean_grad_mag_border"] - row["mean_grad_mag_center"]
            row["border_center_local_contrast_diff"] = (
                row["mean_local_contrast_border"] - row["mean_local_contrast_center"]
            )
            rows.append(row)
    return rows


def aggregate_border_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        key
        for key in rows[0].keys()
        if key
        not in {
            "split",
            "graph_id",
            "label",
            "class_name",
            "border_width",
        }
    ] if rows else []
    grouped: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), int(row["label"]), int(row["border_width"]), str(row["class_name"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (split, label, border_width, class_name), group in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "split": split,
            "label": label,
            "class_name": class_name,
            "border_width": border_width,
            "num_samples": len(group),
        }
        for metric in metrics:
            arr = np.asarray([to_float(row[metric]) for row in group], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            out[f"{metric}_std"] = float(np.nanstd(arr)) if arr.size else float("nan")
            out[f"{metric}_p05"] = float(np.nanpercentile(arr, 5)) if arr.size else float("nan")
            out[f"{metric}_p50"] = float(np.nanpercentile(arr, 50)) if arr.size else float("nan")
            out[f"{metric}_p95"] = float(np.nanpercentile(arr, 95)) if arr.size else float("nan")
        result.append(out)
    return result


def update_mean_maps(
    sums: Dict[int, Dict[str, torch.Tensor]],
    counts: Counter,
    x: torch.Tensor,
    labels: torch.Tensor,
    height: int,
    width: int,
) -> None:
    x = x.detach().cpu().to(torch.float32)
    labels = labels.detach().cpu().long()
    for class_idx in range(NUM_CLASSES):
        mask = labels == class_idx
        if not bool(mask.any()):
            continue
        selected = x[mask]
        counts[class_idx] += int(selected.shape[0])
        if class_idx not in sums:
            sums[class_idx] = {
                "intensity": torch.zeros((height, width), dtype=torch.float64),
                "grad_mag": torch.zeros((height, width), dtype=torch.float64),
                "local_contrast": torch.zeros((height, width), dtype=torch.float64),
            }
        sums[class_idx]["intensity"] += selected[:, :, 0].reshape(-1, height, width).sum(dim=0).double()
        sums[class_idx]["grad_mag"] += selected[:, :, 5].reshape(-1, height, width).sum(dim=0).double()
        sums[class_idx]["local_contrast"] += selected[:, :, 6].reshape(-1, height, width).sum(dim=0).double()


def shape_tuple(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def inspect_batch_integrity(
    integrity: Dict[str, Any],
    split: str,
    batch: Dict[str, torch.Tensor],
    expected_edge_index: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    x = batch["x"].detach().cpu()
    edge_attr = batch["edge_attr"].detach().cpu()
    edge_index = batch["edge_index"].detach().cpu()
    labels = batch["y"].detach().cpu().long()

    split_info = integrity["splits"][split]
    split_info["num_samples"] += int(x.shape[0])
    split_info["class_counts"].update(int(v) for v in labels.tolist())
    split_info["labels_invalid"] += int(((labels < 0) | (labels >= NUM_CLASSES)).sum().item())
    split_info["node_feature_shape_unique"].add(safe_torch_load_shape(x.shape[1:]))
    split_info["num_nodes_unique"].add(int(x.shape[1]))
    split_info["edge_attr_shape_unique"].add(safe_torch_load_shape(edge_attr.shape[1:]))
    split_info["num_edges_unique"].add(int(edge_attr.shape[-2]))
    split_info["edge_dim_unique"].add(int(edge_attr.shape[-1]))

    if edge_index.ndim == 2:
        split_info["edge_index_shape_unique"].add(safe_torch_load_shape(edge_index.shape))
        edge_index_for_compare = edge_index
    elif edge_index.ndim == 3:
        split_info["edge_index_shape_unique"].add(safe_torch_load_shape(edge_index.shape[1:]))
        edge_index_for_compare = edge_index[0]
    else:
        split_info["edge_index_shape_unique"].add(safe_torch_load_shape(edge_index.shape))
        edge_index_for_compare = edge_index.reshape(2, -1) if edge_index.numel() else edge_index

    if expected_edge_index is None:
        return edge_index_for_compare.clone()
    if edge_index_for_compare.shape != expected_edge_index.shape or not torch.equal(edge_index_for_compare, expected_edge_index):
        split_info["edge_index_mismatch_batches"] += 1
    return expected_edge_index


def build_dataloaders(
    config: Dict[str, Any],
    graph_repo_path: Path,
    batch_size: int,
    num_workers: int,
    max_samples_per_split: Optional[int],
    seed: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    data_cfg = dict(config.get("data", {}))
    chunk_cache_size = int(data_cfg.get("chunk_cache_size", data_cfg.get("graph_cache_chunks", 0)) or 0)
    for split_idx, split in enumerate(SPLITS):
        dataset = FullGraphDataset(repo_root=graph_repo_path, split=split, chunk_cache_size=chunk_cache_size)
        indices = choose_indices(len(dataset), max_samples=max_samples_per_split, seed=seed + 1009 * split_idx)
        used_dataset = Subset(dataset, indices) if indices is not None else dataset
        loaders[split] = DataLoader(
            used_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_fn_full_graph,
            pin_memory=False,
        )
    return loaders


def empty_integrity(max_samples_per_split: Optional[int] = None) -> Dict[str, Any]:
    return {
        "audit_options": {
            "max_samples_per_split": max_samples_per_split,
            "sampled_subset": max_samples_per_split is not None and int(max_samples_per_split) > 0,
        },
        "splits": {
            split: {
                "num_samples": 0,
                "class_counts": Counter(),
                "labels_invalid": 0,
                "node_feature_shape_unique": set(),
                "edge_attr_shape_unique": set(),
                "edge_index_shape_unique": set(),
                "num_nodes_unique": set(),
                "num_edges_unique": set(),
                "edge_dim_unique": set(),
                "edge_index_mismatch_batches": 0,
                "notes": [],
            }
            for split in SPLITS
        },
        "node_feature_nan_counts": {name: 0 for name in NODE_FEATURE_NAMES},
        "node_feature_inf_counts": {name: 0 for name in NODE_FEATURE_NAMES},
        "edge_feature_nan_counts": {name: 0 for name in EDGE_FEATURE_NAMES},
        "edge_feature_inf_counts": {name: 0 for name in EDGE_FEATURE_NAMES},
        "warnings": [],
    }


def finalize_integrity(
    integrity: Dict[str, Any],
    node_split_accs: Dict[str, FeatureStatsAccumulator],
    edge_split_accs: Dict[str, FeatureStatsAccumulator],
) -> Dict[str, Any]:
    warnings = integrity["warnings"]
    sampled_subset = bool(integrity.get("audit_options", {}).get("sampled_subset", False))
    for split, info in integrity["splits"].items():
        for key in (
            "node_feature_shape_unique",
            "edge_attr_shape_unique",
            "edge_index_shape_unique",
            "num_nodes_unique",
            "num_edges_unique",
            "edge_dim_unique",
        ):
            info[key] = sorted(info[key])
        info["num_classes"] = len([c for c, n in info["class_counts"].items() if n > 0])
        info["class_counts"] = {str(k): int(v) for k, v in sorted(info["class_counts"].items())}
        if "2304" not in [str(v) for v in info["num_nodes_unique"]]:
            warnings.append(f"{split}: num_nodes unique is {info['num_nodes_unique']}, expected 2304")
        if info["edge_dim_unique"] != [5]:
            warnings.append(f"{split}: edge_dim unique is {info['edge_dim_unique']}, expected 5")
        if not any(str(shape).endswith("7") or shape == "2304x7" for shape in info["node_feature_shape_unique"]):
            warnings.append(f"{split}: node feature shapes {info['node_feature_shape_unique']} do not match [2304,7]")
        if info["labels_invalid"] > 0:
            warnings.append(f"{split}: found {info['labels_invalid']} labels outside 0..6")
        if info["edge_index_mismatch_batches"] > 0:
            warnings.append(f"{split}: edge_index changed in {info['edge_index_mismatch_batches']} batches")
        counts = [int(info["class_counts"].get(str(c), 0)) for c in range(NUM_CLASSES)]
        nonzero = [v for v in counts if v > 0]
        if len(nonzero) < NUM_CLASSES:
            message = f"{split}: sampled subset is missing classes in class counts {counts}"
            if sampled_subset:
                info["notes"].append(message)
            else:
                warnings.append(f"{split}: missing classes in split class counts {counts}")
        elif max(nonzero) / max(min(nonzero), 1) > 20:
            warnings.append(f"{split}: strong class imbalance max/min={max(nonzero)}/{min(nonzero)}")

    node_nan = np.zeros(len(NODE_FEATURE_NAMES), dtype=np.int64)
    node_inf = np.zeros(len(NODE_FEATURE_NAMES), dtype=np.int64)
    edge_nan = np.zeros(len(EDGE_FEATURE_NAMES), dtype=np.int64)
    edge_inf = np.zeros(len(EDGE_FEATURE_NAMES), dtype=np.int64)
    for acc in node_split_accs.values():
        node_nan += acc.nan_count
        node_inf += acc.inf_count
    for acc in edge_split_accs.values():
        edge_nan += acc.nan_count
        edge_inf += acc.inf_count
    integrity["node_feature_nan_counts"] = {name: int(node_nan[idx]) for idx, name in enumerate(NODE_FEATURE_NAMES)}
    integrity["node_feature_inf_counts"] = {name: int(node_inf[idx]) for idx, name in enumerate(NODE_FEATURE_NAMES)}
    integrity["edge_feature_nan_counts"] = {name: int(edge_nan[idx]) for idx, name in enumerate(EDGE_FEATURE_NAMES)}
    integrity["edge_feature_inf_counts"] = {name: int(edge_inf[idx]) for idx, name in enumerate(EDGE_FEATURE_NAMES)}

    for name, count in integrity["node_feature_nan_counts"].items():
        if count > 0:
            warnings.append(f"node feature {name}: NaN count={count}")
    for name, count in integrity["node_feature_inf_counts"].items():
        if count > 0:
            warnings.append(f"node feature {name}: Inf count={count}")
    for name, count in integrity["edge_feature_nan_counts"].items():
        if count > 0:
            warnings.append(f"edge feature {name}: NaN count={count}")
    for name, count in integrity["edge_feature_inf_counts"].items():
        if count > 0:
            warnings.append(f"edge feature {name}: Inf count={count}")

    add_range_warnings(warnings, node_split_accs, edge_split_accs)
    integrity["warnings"] = sorted(set(warnings))
    return integrity


def add_range_warnings(
    warnings: List[str],
    node_split_accs: Dict[str, FeatureStatsAccumulator],
    edge_split_accs: Dict[str, FeatureStatsAccumulator],
) -> None:
    expected_node = {
        "intensity": (0.0, 1.0),
        "x_norm": (0.0, 1.0),
        "y_norm": (0.0, 1.0),
        "gx": (-1.0, 1.0),
        "gy": (-1.0, 1.0),
        "grad_mag": (0.0, 1.0),
        "local_contrast": (0.0, 1.0),
    }
    expected_edge = {
        "dx": (-1.0, 1.0),
        "dy": (-1.0, 1.0),
        "dist": (1.0, math.sqrt(2.0)),
        "delta_intensity": (0.0, 1.0),
        "intensity_similarity": (0.0, 1.0),
    }
    for split, acc in node_split_accs.items():
        for idx, name in enumerate(NODE_FEATURE_NAMES):
            lo, hi = expected_node[name]
            if acc.min[idx] < lo - 1e-4 or acc.max[idx] > hi + 1e-4:
                warnings.append(
                    f"{split}: node {name} range [{acc.min[idx]:.6g}, {acc.max[idx]:.6g}] outside expected [{lo}, {hi}]"
                )
            mean, std = acc.mean_std_by_feature()[name]
            if std < 1e-7:
                warnings.append(f"{split}: node {name} has near-zero std={std:.6g}")
            if name in {"gx", "gy", "grad_mag", "local_contrast"} and std > 0.5:
                warnings.append(f"{split}: node {name} has unusually large std={std:.6g}")
    for split, acc in edge_split_accs.items():
        for idx, name in enumerate(EDGE_FEATURE_NAMES):
            lo, hi = expected_edge[name]
            if acc.min[idx] < lo - 1e-4 or acc.max[idx] > hi + 1e-4:
                warnings.append(
                    f"{split}: edge {name} range [{acc.min[idx]:.6g}, {acc.max[idx]:.6g}] outside expected [{lo:.6g}, {hi:.6g}]"
                )
            mean, std = acc.mean_std_by_feature()[name]
            if name in {"dx", "dy", "dist"} and std < 1e-7:
                warnings.append(f"{split}: edge static feature {name} has near-zero std={std:.6g}")


def make_drift_rows(
    node_split_accs: Dict[str, FeatureStatsAccumulator],
    edge_split_accs: Dict[str, FeatureStatsAccumulator],
) -> Tuple[List[Dict[str, Any]], str]:
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        scipy_note = "scipy available: ks_stat and wasserstein_distance were computed on bounded reservoirs."
    except Exception:
        ks_2samp = None
        wasserstein_distance = None
        scipy_note = "scipy unavailable: ks_stat and wasserstein_distance columns are left blank."

    rows: List[Dict[str, Any]] = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for kind, accs, names in (
        ("node", node_split_accs, NODE_FEATURE_NAMES),
        ("edge", edge_split_accs, EDGE_FEATURE_NAMES),
    ):
        train_std = accs.get("train").mean_std_by_feature() if "train" in accs else {}
        for split_a, split_b in pairs:
            if split_a not in accs or split_b not in accs:
                continue
            rows_a = {row["feature"]: row for row in accs[split_a].final_rows({})}
            rows_b = {row["feature"]: row for row in accs[split_b].final_rows({})}
            for idx, feature in enumerate(names):
                mean_a = to_float(rows_a[feature]["mean"])
                mean_b = to_float(rows_b[feature]["mean"])
                std_a = to_float(rows_a[feature]["std"])
                std_b = to_float(rows_b[feature]["std"])
                denom = to_float(train_std.get(feature, (float("nan"), std_a))[1])
                normalized = abs(mean_a - mean_b) / (denom + EPS) if np.isfinite(denom) else float("nan")
                sample_a = accs[split_a].reservoirs[idx]
                sample_b = accs[split_b].reservoirs[idx]
                ks_stat = ""
                wasserstein = ""
                if ks_2samp is not None and sample_a.size > 0 and sample_b.size > 0:
                    ks_stat = float(ks_2samp(sample_a, sample_b).statistic)
                    wasserstein = float(wasserstein_distance(sample_a, sample_b))
                rows.append(
                    {
                        "feature_type": kind,
                        "feature": feature,
                        "split_a": split_a,
                        "split_b": split_b,
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "std_a": std_a,
                        "std_b": std_b,
                        "mean_diff": mean_a - mean_b,
                        "std_diff": std_a - std_b,
                        "normalized_mean_diff": normalized,
                        "ks_stat": ks_stat,
                        "wasserstein_distance": wasserstein,
                    }
                )
    return rows, scipy_note


def rows_for_percentiles(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [percentile_key(p) for p in PERCENTILES]
    result = []
    for row in rows:
        result.append({k: row.get(k) for k in ["split", "feature", *keys, "reservoir_count"]})
    return result


def summarize_top_drift(drift_rows: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    valid = [row for row in drift_rows if np.isfinite(to_float(row.get("normalized_mean_diff")))]
    return sorted(valid, key=lambda row: to_float(row["normalized_mean_diff"]), reverse=True)[:limit]


def write_correlation_csv(path: Path, corr: np.ndarray, names: Sequence[str]) -> None:
    rows = []
    for i, name in enumerate(names):
        row = {"feature": name}
        for j, other in enumerate(names):
            row[other] = float(corr[i, j]) if np.isfinite(corr[i, j]) else ""
        rows.append(row)
    write_csv(path, rows, ["feature", *names])


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:
        return None, str(exc)


def save_histograms(
    output_dir: Path,
    node_split_accs: Dict[str, FeatureStatsAccumulator],
    edge_split_accs: Dict[str, FeatureStatsAccumulator],
) -> Optional[str]:
    plt, err = try_import_matplotlib()
    if err:
        return err
    assert plt is not None
    for kind, accs, names, subdir in (
        ("node", node_split_accs, NODE_FEATURE_NAMES, output_dir / "figures" / "node_feature_histograms"),
        ("edge", edge_split_accs, EDGE_FEATURE_NAMES, output_dir / "figures" / "edge_feature_histograms"),
    ):
        ensure_dir(subdir)
        for split, acc in accs.items():
            for idx, name in enumerate(names):
                sample = acc.reservoirs[idx]
                if sample.size == 0:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(sample, bins=80, color="#31688e", alpha=0.85)
                ax.set_title(f"{kind} {split}: {name}")
                ax.set_xlabel(name)
                ax.set_ylabel("reservoir count")
                fig.tight_layout()
                fig.savefig(subdir / f"{split}_{name}.png", dpi=140)
                plt.close(fig)
    return None


def save_feature_map_samples(output_dir: Path, sample_reservoir: SampleReservoir, height: int, width: int) -> Optional[str]:
    plt, err = try_import_matplotlib()
    if err:
        return err
    assert plt is not None
    out_dir = ensure_dir(output_dir / "figures" / "feature_maps_samples")
    panels = [
        ("intensity", 0, "gray"),
        ("gx", 3, "coolwarm"),
        ("gy", 4, "coolwarm"),
        ("grad_mag", 5, "magma"),
        ("local_contrast", 6, "viridis"),
    ]
    for item in sample_reservoir.all_samples():
        x = item["x"].reshape(height, width, -1).numpy()
        fig, axes = plt.subplots(1, len(panels), figsize=(14, 3))
        for ax, (name, idx, cmap) in zip(axes, panels):
            im = ax.imshow(x[:, :, idx], cmap=cmap)
            ax.set_title(name)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        class_name = EMOTION_NAMES[item["label"]] if 0 <= item["label"] < NUM_CLASSES else "INVALID"
        fig.suptitle(f"{item['split']} class={item['label']} {class_name} graph_id={item['graph_id']}")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"sample_{item['split']}_{item['label']}_{item['graph_id']}_feature_maps.png",
            dpi=150,
        )
        plt.close(fig)
    return None


def save_mean_maps(
    output_dir: Path,
    mean_map_sums: Dict[int, Dict[str, torch.Tensor]],
    mean_map_counts: Counter,
) -> Optional[str]:
    plt, err = try_import_matplotlib()
    if err:
        return err
    assert plt is not None
    out_dir = ensure_dir(output_dir / "figures" / "feature_mean_maps_by_class")
    features = ["intensity", "grad_mag", "local_contrast"]
    means: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in features}
    for class_idx, feature_sums in mean_map_sums.items():
        count = max(int(mean_map_counts[class_idx]), 1)
        for feature in features:
            means[feature][class_idx] = (feature_sums[feature] / count).numpy()
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(means[feature][class_idx], cmap="gray" if feature == "intensity" else "viridis")
            ax.set_title(f"class {class_idx} {EMOTION_NAMES[class_idx]} mean {feature}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / f"class_{class_idx}_mean_{feature}.png", dpi=150)
            plt.close(fig)

    for feature in features:
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(18, 3))
        for class_idx, ax in enumerate(axes):
            if class_idx in means[feature]:
                im = ax.imshow(means[feature][class_idx], cmap="gray" if feature == "intensity" else "viridis")
                ax.set_title(f"{class_idx} {EMOTION_NAMES[class_idx]}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.set_title(f"{class_idx} missing")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"class_mean_{feature}_grid.png", dpi=150)
        plt.close(fig)
    return None


def save_border_figures(output_dir: Path, border_rows: Sequence[Dict[str, Any]], border_widths: Sequence[int]) -> Optional[str]:
    plt, err = try_import_matplotlib()
    if err:
        return err
    assert plt is not None
    out_dir = ensure_dir(output_dir / "figures" / "border_center_maps")
    for width in border_widths:
        masks = make_masks(border_width=int(width))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, name in zip(axes, ["border", "center", "corner", "upper"]):
            ax.imshow(masks[name].reshape(48, 48).numpy(), cmap="gray")
            ax.set_title(f"{name} w={width}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"masks_border_width_{width}.png", dpi=140)
        plt.close(fig)

    if border_rows:
        primary_width = int(border_widths[0])
        grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for row in border_rows:
            if int(row["border_width"]) == primary_width:
                grouped[(str(row["split"]), int(row["label"]))].append(to_float(row["border_center_intensity_diff"]))
        for split in SPLITS:
            values = [float(np.nanmean(grouped[(split, c)])) if grouped[(split, c)] else np.nan for c in range(NUM_CLASSES)]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(NUM_CLASSES), values, color="#35b779")
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xticks(range(NUM_CLASSES), EMOTION_NAMES, rotation=30, ha="right")
            ax.set_ylabel("border - center intensity")
            ax.set_title(f"{split} border/center intensity diff w={primary_width}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{split}_class_border_center_intensity_diff.png", dpi=150)
            plt.close(fig)
    return None


def save_correlation_heatmap(output_dir: Path, corr: np.ndarray, names: Sequence[str], filename: str, title: str) -> Optional[str]:
    plt, err = try_import_matplotlib()
    if err:
        return err
    assert plt is not None
    out_dir = ensure_dir(output_dir / "figures" / "correlation_heatmaps")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    return None


def write_report(
    output_dir: Path,
    integrity: Dict[str, Any],
    node_split_rows: Sequence[Dict[str, Any]],
    edge_split_rows: Sequence[Dict[str, Any]],
    drift_rows: Sequence[Dict[str, Any]],
    border_class_rows: Sequence[Dict[str, Any]],
    scipy_note: str,
    figure_note: str,
) -> None:
    warnings = integrity.get("warnings", [])
    node_by_feature = {(row["split"], row["feature"]): row for row in node_split_rows}
    edge_by_feature = {(row["split"], row["feature"]): row for row in edge_split_rows}
    top_drift = summarize_top_drift(drift_rows)

    severe = [
        w
        for w in warnings
        if any(token in w.lower() for token in ("nan", "inf", "outside", "invalid", "mismatch", "expected"))
    ]
    scale_ok = "No critical range warning was found." if not severe else "Range/integrity warnings were found; inspect the warning list."
    v2_recommendation = "Hold input-v2 until after D6C unless the warnings below match a training failure pattern."
    if severe:
        v2_recommendation = "Fix normalization/integrity first before adding graph input v2 features."
    elif top_drift and to_float(top_drift[0]["normalized_mean_diff"]) > 0.25:
        v2_recommendation = "Audit split construction/preprocessing before input-v2; drift is more urgent than adding features."

    class_border_notes = []
    primary_width_rows = [row for row in border_class_rows if int(row.get("border_width", -1)) == 3]
    if primary_width_rows:
        ranked = sorted(
            primary_width_rows,
            key=lambda row: abs(to_float(row.get("border_center_intensity_diff_mean"))),
            reverse=True,
        )[:5]
        for row in ranked:
            class_border_notes.append(
                f"- {row['split']} {row['class_name']} w={row['border_width']}: "
                f"border-center intensity diff mean={format_float(row.get('border_center_intensity_diff_mean'))}"
            )

    lines: List[str] = []
    lines.append("# Graph Input Audit Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- Critical warnings: {len(severe)}; total warnings: {len(warnings)}.")
    lines.append(f"- Feature scale: {scale_ok}")
    lines.append("- NaN/Inf/outlier checks are included in CSV/JSON; percentiles/outlier ratios use bounded reservoirs.")
    lines.append(f"- Drift analysis: {scipy_note}")
    lines.append(f"- Recommendation: {v2_recommendation}")
    lines.append("")
    lines.append("## 2. Graph Integrity")
    for split, info in integrity["splits"].items():
        lines.append(
            f"- {split}: samples={info['num_samples']}, classes={info['num_classes']}, "
            f"node_shapes={info['node_feature_shape_unique']}, edge_shapes={info['edge_attr_shape_unique']}, "
            f"num_edges={info['num_edges_unique']}"
        )
        counts = ", ".join(f"{idx}:{info['class_counts'].get(str(idx), 0)}" for idx in range(NUM_CLASSES))
        lines.append(f"  class counts: {counts}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in warnings[:80])
        if len(warnings) > 80:
            lines.append(f"- ... {len(warnings) - 80} more warnings in graph_integrity_report.json")
    else:
        lines.append("")
        lines.append("No integrity warnings were generated.")
    lines.append("")
    lines.append("## 3. Node Feature Distribution")
    lines.append("| split | feature | mean | std | min | max | p1 | p99 | zero_ratio | outlier_z3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        for feature in NODE_FEATURE_NAMES:
            row = node_by_feature.get((split, feature), {})
            lines.append(
                f"| {split} | {feature} | {format_float(row.get('mean'))} | {format_float(row.get('std'))} | "
                f"{format_float(row.get('min'))} | {format_float(row.get('max'))} | "
                f"{format_float(row.get('p1'))} | {format_float(row.get('p99'))} | "
                f"{format_float(row.get('zero_ratio'))} | {format_float(row.get('outlier_ratio_z3'))} |"
            )
    lines.append("")
    lines.append(
        "Node feature checks target intensity/x/y ranges, gx/gy/grad scale, grad_mag tails, local_contrast scale, and near-zero std."
    )
    lines.append("")
    lines.append("## 4. Edge Feature Distribution")
    lines.append("| split | feature | mean | std | min | max | p1 | p99 | zero_ratio | outlier_z3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        for feature in EDGE_FEATURE_NAMES:
            row = edge_by_feature.get((split, feature), {})
            lines.append(
                f"| {split} | {feature} | {format_float(row.get('mean'))} | {format_float(row.get('std'))} | "
                f"{format_float(row.get('min'))} | {format_float(row.get('max'))} | "
                f"{format_float(row.get('p1'))} | {format_float(row.get('p99'))} | "
                f"{format_float(row.get('zero_ratio'))} | {format_float(row.get('outlier_ratio_z3'))} |"
            )
    lines.append("")
    lines.append("dx/dy/dist should be static grid-topology features; delta_intensity and intensity_similarity should vary per image.")
    lines.append("")
    lines.append("## 5. Split Drift Analysis")
    if top_drift:
        lines.append("| type | feature | pair | norm_mean_diff | ks | wasserstein |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in top_drift:
            lines.append(
                f"| {row['feature_type']} | {row['feature']} | {row['split_a']} vs {row['split_b']} | "
                f"{format_float(row['normalized_mean_diff'])} | {format_float(row.get('ks_stat'))} | "
                f"{format_float(row.get('wasserstein_distance'))} |"
            )
    else:
        lines.append("No drift rows were generated.")
    lines.append("")
    lines.append("## 6. Class-wise Feature Bias")
    lines.append(
        "Class-wise node/edge statistics are written to `node_feature_stats_by_class.csv` and `edge_feature_stats_by_class.csv`."
    )
    lines.append("Inspect Disgust/Fear rows first because those classes are usually low-count and shortcut-prone in FER-2013.")
    lines.append("")
    lines.append("## 7. Border and Center Bias")
    if class_border_notes:
        lines.extend(class_border_notes)
    else:
        lines.append("No border/class aggregate rows were generated.")
    lines.append(
        "Positive border-center intensity diff means the crop border is brighter than the central face region; large class-specific values are shortcut risk."
    )
    lines.append("")
    lines.append("## 8. Feature Map Visual Inspection")
    lines.append(f"- Figure generation note: {figure_note}")
    lines.append("- Review `figures/feature_maps_samples/` for sample-level intensity/gx/gy/grad_mag/local_contrast maps.")
    lines.append("- Review `figures/feature_mean_maps_by_class/` for class mean intensity/gradient/contrast maps.")
    lines.append("")
    lines.append("## 9. Correlation and Redundancy")
    lines.append(
        "Correlation CSV/heatmaps are written under `figures/correlation_heatmaps/` and duplicated at the audit root for convenience."
    )
    lines.append("High absolute correlation suggests redundancy; low correlation in dynamic edge features suggests useful complementary signal.")
    lines.append("")
    lines.append("## 10. Recommendations for Graph Input V2")
    lines.append("A. Safe changes")
    lines.append("- Normalize/re-scale existing features only if this audit reports range or std warnings.")
    lines.append("- Add `grad_ori_cos` and `grad_ori_sin` if gx/gy/grad_mag are clean and not degenerate.")
    lines.append("- Add `local_mean_5x5` and `local_std_5x5` if local_contrast is useful but noisy.")
    lines.append("- Add a soft `border_distance` feature if border-center bias is measurable but not class-hardcoded.")
    lines.append("")
    lines.append("B. Medium-risk changes")
    lines.append("- `center_prior_soft`, because it may help alignment but can also encode crop assumptions.")
    lines.append("- Sparse multi-scale edges and patch context features, because they change model capacity and IO cost.")
    lines.append("- `patch_mean_5x5` / `patch_std_5x5`, because they may duplicate local_mean/std unless ablated carefully.")
    lines.append("")
    lines.append("C. Not recommended")
    lines.append("- Dense `[2304,2304,3]` edge tensors.")
    lines.append("- Hard face masks, hard landmarks, or hard-coded eyes/nose/mouth priors.")
    lines.append("")
    lines.append("## 11. Next Steps")
    lines.append("- If audit is clean: keep current input for D6C, then consider D6D input-v2.")
    lines.append("- If scale errors appear: fix normalization before new architecture work.")
    lines.append("- If border bias is strong: consider soft border-distance features or border-aware loss diagnostics.")
    lines.append("- If features look too weak/noisy: implement D6D-input-v2 with small, ablated feature additions.")
    lines.append("")
    (output_dir / "graph_input_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    config = load_config(args.config)
    graph_repo_path = Path(args.graph_repo_path) if args.graph_repo_path else resolve_path(
        config.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo")
    )
    if graph_repo_path is None:
        raise FileNotFoundError("Could not resolve graph_repo_path")
    output_dir = ensure_dir(Path(args.output_dir))
    figures_root = output_dir / "figures"
    for rel in (
        "node_feature_histograms",
        "edge_feature_histograms",
        "feature_maps_samples",
        "feature_mean_maps_by_class",
        "border_center_maps",
        "correlation_heatmaps",
    ):
        ensure_dir(figures_root / rel)

    height = int(config.get("graph", {}).get("height", 48))
    width = int(config.get("graph", {}).get("width", 48))
    print(f"[Audit] graph_repo_path={graph_repo_path}")
    print(f"[Audit] output_dir={output_dir}")
    print(f"[Audit] max_samples_per_split={args.max_samples_per_split}")

    loaders = build_dataloaders(
        config=config,
        graph_repo_path=graph_repo_path,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_samples_per_split=args.max_samples_per_split,
        seed=int(args.seed),
    )

    rng = np.random.default_rng(int(args.seed))
    node_split_accs = {
        split: FeatureStatsAccumulator(NODE_FEATURE_NAMES, rng=np.random.default_rng(int(args.seed) + i))
        for i, split in enumerate(SPLITS)
    }
    edge_split_accs = {
        split: FeatureStatsAccumulator(EDGE_FEATURE_NAMES, rng=np.random.default_rng(int(args.seed) + 101 + i))
        for i, split in enumerate(SPLITS)
    }
    node_class_accs = {
        (split, class_idx): FeatureStatsAccumulator(
            NODE_FEATURE_NAMES,
            rng=np.random.default_rng(int(args.seed) + 1000 + 31 * i + class_idx),
            reservoir_size=80_000,
            batch_sample_size=8_000,
        )
        for i, split in enumerate(SPLITS)
        for class_idx in range(NUM_CLASSES)
    }
    edge_class_accs = {
        (split, class_idx): FeatureStatsAccumulator(
            EDGE_FEATURE_NAMES,
            rng=np.random.default_rng(int(args.seed) + 2000 + 31 * i + class_idx),
            reservoir_size=80_000,
            batch_sample_size=8_000,
        )
        for i, split in enumerate(SPLITS)
        for class_idx in range(NUM_CLASSES)
    }
    node_corr = RowReservoir(len(NODE_FEATURE_NAMES), rng=np.random.default_rng(int(args.seed) + 3001))
    edge_corr = RowReservoir(len(EDGE_FEATURE_NAMES), rng=np.random.default_rng(int(args.seed) + 3002))
    sample_reservoir = SampleReservoir(per_class=3, rng=np.random.default_rng(int(args.seed) + 4001))
    mean_map_sums: Dict[int, Dict[str, torch.Tensor]] = {}
    mean_map_counts: Counter = Counter()
    border_rows: List[Dict[str, Any]] = []
    integrity = empty_integrity(args.max_samples_per_split)
    expected_edge_index: Optional[torch.Tensor] = None

    for split in SPLITS:
        print(f"[Audit] reading split={split}")
        for batch_idx, batch in enumerate(loaders[split]):
            x = batch["x"].detach().cpu()
            edge_attr = batch["edge_attr"].detach().cpu()
            labels = batch["y"].detach().cpu().long()
            graph_ids = batch.get("graph_id", torch.arange(x.shape[0])).detach().cpu().long()

            expected_edge_index = inspect_batch_integrity(integrity, split, batch, expected_edge_index)
            node_split_accs[split].update(x)
            edge_split_accs[split].update(edge_attr)

            for class_idx in range(NUM_CLASSES):
                mask = labels == class_idx
                if not bool(mask.any()):
                    continue
                node_class_accs[(split, class_idx)].update(x[mask])
                edge_class_accs[(split, class_idx)].update(edge_attr[mask])

            if split == "train":
                node_corr.update(x)
                edge_corr.update(edge_attr)
                update_mean_maps(mean_map_sums, mean_map_counts, x, labels, height=height, width=width)
                for sample_idx in range(x.shape[0]):
                    sample_reservoir.update(
                        split=split,
                        graph_id=int(graph_ids[sample_idx]),
                        label=int(labels[sample_idx]),
                        x=x[sample_idx],
                    )

            border_rows.extend(
                compute_border_rows(
                    split=split,
                    batch=batch,
                    border_widths=[int(v) for v in args.border_widths],
                    height=height,
                    width=width,
                )
            )
            if (batch_idx + 1) % 25 == 0:
                print(f"[Audit] {split}: processed {batch_idx + 1} batches")

    node_split_rows = []
    edge_split_rows = []
    node_class_rows = []
    edge_class_rows = []
    for split in SPLITS:
        node_split_rows.extend(node_split_accs[split].final_rows({"split": split}))
        edge_split_rows.extend(edge_split_accs[split].final_rows({"split": split}))
        for class_idx in range(NUM_CLASSES):
            node_class_rows.extend(
                node_class_accs[(split, class_idx)].final_rows(
                    {"split": split, "label": class_idx, "class_name": EMOTION_NAMES[class_idx]}
                )
            )
            edge_class_rows.extend(
                edge_class_accs[(split, class_idx)].final_rows(
                    {"split": split, "label": class_idx, "class_name": EMOTION_NAMES[class_idx]}
                )
            )

    border_class_rows = aggregate_border_rows(border_rows)
    drift_rows, scipy_note = make_drift_rows(node_split_accs, edge_split_accs)
    integrity = finalize_integrity(integrity, node_split_accs, edge_split_accs)

    node_corr_matrix = node_corr.corr()
    edge_corr_matrix = edge_corr.corr()

    write_csv(output_dir / "node_feature_stats_by_split.csv", node_split_rows)
    write_csv(output_dir / "node_feature_stats_by_class.csv", node_class_rows)
    write_csv(output_dir / "edge_feature_stats_by_split.csv", edge_split_rows)
    write_csv(output_dir / "edge_feature_stats_by_class.csv", edge_class_rows)
    write_csv(output_dir / "node_feature_percentiles_by_split.csv", rows_for_percentiles(node_split_rows))
    write_csv(output_dir / "edge_feature_percentiles_by_split.csv", rows_for_percentiles(edge_split_rows))
    write_csv(output_dir / "feature_drift_train_val_test.csv", drift_rows)
    write_csv(output_dir / "border_center_statistics.csv", border_rows)
    write_csv(output_dir / "class_border_center_statistics.csv", border_class_rows)
    write_json(output_dir / "graph_integrity_report.json", integrity)

    corr_dir = ensure_dir(output_dir / "figures" / "correlation_heatmaps")
    write_correlation_csv(corr_dir / "node_feature_correlation.csv", node_corr_matrix, NODE_FEATURE_NAMES)
    write_correlation_csv(corr_dir / "edge_feature_correlation.csv", edge_corr_matrix, EDGE_FEATURE_NAMES)
    write_correlation_csv(output_dir / "node_feature_correlation.csv", node_corr_matrix, NODE_FEATURE_NAMES)
    write_correlation_csv(output_dir / "edge_feature_correlation.csv", edge_corr_matrix, EDGE_FEATURE_NAMES)

    figure_notes: List[str] = []
    if args.make_figures:
        for err in (
            save_histograms(output_dir, node_split_accs, edge_split_accs),
            save_feature_map_samples(output_dir, sample_reservoir, height=height, width=width),
            save_mean_maps(output_dir, mean_map_sums, mean_map_counts),
            save_border_figures(output_dir, border_rows, args.border_widths),
            save_correlation_heatmap(
                output_dir,
                node_corr_matrix,
                NODE_FEATURE_NAMES,
                "node_feature_correlation.png",
                "Train node feature correlation",
            ),
            save_correlation_heatmap(
                output_dir,
                edge_corr_matrix,
                EDGE_FEATURE_NAMES,
                "edge_feature_correlation.png",
                "Train edge feature correlation",
            ),
        ):
            if err:
                figure_notes.append(err)
    else:
        figure_notes.append("make_figures=false; figures were skipped.")
    figure_note = "OK" if not figure_notes else "; ".join(sorted(set(figure_notes)))

    write_report(
        output_dir=output_dir,
        integrity=integrity,
        node_split_rows=node_split_rows,
        edge_split_rows=edge_split_rows,
        drift_rows=drift_rows,
        border_class_rows=border_class_rows,
        scipy_note=scipy_note,
        figure_note=figure_note,
    )
    print(f"[Audit] report written: {output_dir / 'graph_input_audit_report.md'}")
    if integrity.get("warnings"):
        print(f"[Audit] warnings={len(integrity['warnings'])}; see graph_integrity_report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit FER full pixel graph input quality.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--graph_repo_path", default=None, help="Optional graph repository path override.")
    parser.add_argument("--output_dir", default="output/graph_input_audit")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_figures", type=parse_bool, default=True)
    parser.add_argument("--border_widths", type=int, nargs="+", default=[3, 5])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
