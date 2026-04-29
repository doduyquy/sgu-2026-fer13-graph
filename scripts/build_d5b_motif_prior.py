"""Build the D5B-1 offline discriminative full-graph motif prior."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, load_config, resolve_path
from data.graph_repository import GraphRepositoryReader, torch_load
from data.labels import EMOTION_NAMES
from visualize_d5b_prior import save_prior_figures

DEFAULT_FEATURE_NAMES = [
    "intensity",
    "x_norm",
    "y_norm",
    "gx",
    "gy",
    "grad_mag",
    "local_contrast",
]


def _normalise_per_class(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mins = values.amin(dim=1, keepdim=True)
    maxs = values.amax(dim=1, keepdim=True)
    return (values - mins) / (maxs - mins).clamp_min(eps)


def _smooth_prior(raw_score: torch.Tensor, height: int, width: int, kernel_size: int) -> torch.Tensor:
    if int(kernel_size) <= 1:
        return raw_score
    if int(kernel_size) % 2 == 0:
        raise ValueError(f"smooth_kernel must be odd, got {kernel_size}")
    kernel = torch.ones(1, 1, int(kernel_size), int(kernel_size), dtype=raw_score.dtype)
    kernel = kernel / kernel.sum()
    image = raw_score.reshape(raw_score.shape[0], 1, height, width)
    smoothed = F.conv2d(image, kernel, padding=int(kernel_size) // 2)
    return smoothed.reshape(raw_score.shape[0], height * width)


def _activation(node_features: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor:
    if node_features.ndim != 3 or node_features.shape[-1] < 7:
        raise ValueError(f"node_features must be [B, 2304, >=7], got {tuple(node_features.shape)}")
    return (
        float(weights.get("intensity", 0.2)) * node_features[:, :, 0]
        + float(weights.get("grad_mag", 0.4)) * node_features[:, :, 5]
        + float(weights.get("local_contrast", 0.4)) * node_features[:, :, 6]
    )


def collect_train_activations(
    repo_root: str | Path,
    split: str,
    activation_weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
    reader = GraphRepositoryReader(repo_root)
    num_samples = reader.split_size(split)
    shared = reader.load_shared()
    num_nodes = int(shared.height) * int(shared.width)
    activations = torch.empty(num_samples, num_nodes, dtype=torch.float32)
    labels = torch.empty(num_samples, dtype=torch.long)
    feature_names = list(DEFAULT_FEATURE_NAMES)

    cursor = 0
    for chunk_path in tqdm(reader.chunk_paths(split), desc=f"collect {split} activations"):
        chunk = torch_load(chunk_path)
        if not chunk:
            continue
        node_features = torch.stack([sample.node_features.float() for sample in chunk], dim=0)
        if node_features.shape[1:] != (num_nodes, 7):
            raise ValueError(
                f"Expected chunk node_features [B, {num_nodes}, 7], got {tuple(node_features.shape)}"
            )
        if chunk[0].node_feature_names:
            feature_names = list(chunk[0].node_feature_names)
        act = _activation(node_features, activation_weights)
        batch_size = int(act.shape[0])
        activations[cursor : cursor + batch_size] = act
        labels[cursor : cursor + batch_size] = torch.tensor([int(sample.label) for sample in chunk])
        cursor += batch_size

    if cursor != num_samples:
        activations = activations[:cursor]
        labels = labels[:cursor]
    if not torch.isfinite(activations).all():
        raise ValueError("Non-finite activation values encountered")
    return activations, labels, feature_names


def compute_node_prior(
    activations: torch.Tensor,
    labels: torch.Tensor,
    config: Dict[str, Any],
    height: int = 48,
    width: int = 48,
    num_classes: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if activations.ndim != 2:
        raise ValueError(f"activations must be [S, N], got {tuple(activations.shape)}")
    if activations.shape[1] != height * width:
        raise ValueError(f"Expected {height * width} nodes, got {activations.shape[1]}")
    if labels.ndim != 1 or labels.shape[0] != activations.shape[0]:
        raise ValueError("labels must be [S] and align with activations")

    prior_cfg = dict(config.get("prior", {}))
    eps = float(prior_cfg.get("eps", 1e-6))
    percentile = float(prior_cfg.get("support_percentile", 75))
    alpha = float(prior_cfg.get("alpha_effect", 0.45))
    beta = float(prior_cfg.get("beta_fisher", 0.30))
    gamma = float(prior_cfg.get("gamma_support", 0.20))
    delta = float(prior_cfg.get("delta_commonness", 0.15))
    smooth_kernel = int(prior_cfg.get("smooth_kernel", 3))
    clamp_min = float(prior_cfg.get("clamp_min", 0.05))
    clamp_max = float(prior_cfg.get("clamp_max", 0.95))

    labels = labels.long()
    sample_count = int(activations.shape[0])
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    if (class_counts <= 0).any():
        raise ValueError(f"Each class needs support in train split, got counts={class_counts.tolist()}")

    total_sum = activations.sum(dim=0)
    total_sumsq = activations.pow(2).sum(dim=0)
    mean_all = total_sum / float(sample_count)
    var_all = (total_sumsq / float(sample_count) - mean_all.pow(2)).clamp_min(0.0)
    std_all = var_all.sqrt()
    threshold = torch.quantile(activations, percentile / 100.0, dim=0)

    mean_c = torch.empty(num_classes, activations.shape[1], dtype=torch.float32)
    mean_rest = torch.empty_like(mean_c)
    var_c = torch.empty_like(mean_c)
    var_rest = torch.empty_like(mean_c)
    support_c = torch.empty_like(mean_c)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        acts_c = activations[mask]
        count_c = float(acts_c.shape[0])
        rest_count = float(sample_count - acts_c.shape[0])
        sum_c = acts_c.sum(dim=0)
        sumsq_c = acts_c.pow(2).sum(dim=0)
        mean_c[class_idx] = sum_c / count_c
        var_c[class_idx] = (sumsq_c / count_c - mean_c[class_idx].pow(2)).clamp_min(0.0)
        rest_sum = total_sum - sum_c
        rest_sumsq = total_sumsq - sumsq_c
        mean_rest[class_idx] = rest_sum / max(rest_count, 1.0)
        var_rest[class_idx] = (
            rest_sumsq / max(rest_count, 1.0) - mean_rest[class_idx].pow(2)
        ).clamp_min(0.0)
        support_c[class_idx] = (acts_c > threshold.view(1, -1)).float().mean(dim=0)

    commonness = support_c.mean(dim=0)
    diff = mean_c - mean_rest
    effect = (diff / (std_all.view(1, -1) + eps)).clamp_min(0.0)
    fisher = diff.pow(2) / (var_c + var_rest + eps)
    effect = _normalise_per_class(effect)
    fisher = _normalise_per_class(fisher)

    node_score_raw = (
        alpha * effect
        + beta * fisher
        + gamma * support_c
        - delta * commonness.view(1, -1)
    ).clamp_min(0.0)
    smoothed = _smooth_prior(node_score_raw, height=height, width=width, kernel_size=smooth_kernel)
    node_prior = _normalise_per_class(smoothed).clamp(clamp_min, clamp_max)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": sample_count,
        "class_counts": [int(v) for v in class_counts.tolist()],
        "height": int(height),
        "width": int(width),
        "support_percentile": percentile,
        "activation_threshold_min": float(threshold.min().item()),
        "activation_threshold_max": float(threshold.max().item()),
        "raw_score_min": float(node_score_raw.min().item()),
        "raw_score_max": float(node_score_raw.max().item()),
        "node_prior_min": float(node_prior.min().item()),
        "node_prior_max": float(node_prior.max().item()),
        "node_prior_mean_per_class": [float(v) for v in node_prior.mean(dim=1).tolist()],
    }
    return node_prior, node_score_raw, meta


def run_build(config: Dict[str, Any], graph_repo_path: str | None = None, output_dir: str | None = None) -> Path:
    paths = config.get("paths", {})
    graph_cfg = config.get("graph", {})
    prior_cfg = dict(config.get("prior", {}))
    repo = resolve_path(graph_repo_path or paths.get("graph_repo_path", "artifacts/graph_repo"))
    out_dir = resolve_path(output_dir or prior_cfg.get("output_dir", "artifacts/d5b_motif_prior"))
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_weights = dict(prior_cfg.get("activation_weights", {}))
    activations, labels, feature_names = collect_train_activations(
        repo_root=repo,
        split="train",
        activation_weights=activation_weights,
    )
    node_prior, node_score_raw, meta = compute_node_prior(
        activations=activations,
        labels=labels,
        config=config,
        height=int(graph_cfg.get("height", 48)),
        width=int(graph_cfg.get("width", 48)),
        num_classes=int(config.get("model", {}).get("num_classes", 7)),
    )

    payload = {
        "node_prior": node_prior,
        "node_score_raw": node_score_raw,
        "class_names": list(EMOTION_NAMES),
        "feature_names": feature_names,
        "config": config,
    }
    prior_path = out_dir / "node_prior.pt"
    torch.save(payload, prior_path)

    meta_payload = {
        **meta,
        "graph_repo_path": str(repo),
        "output_dir": str(out_dir),
        "node_prior_path": str(prior_path),
        "class_names": list(EMOTION_NAMES),
        "feature_names": feature_names,
        "prior_config": prior_cfg,
    }
    with (out_dir / "node_prior_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    save_prior_figures(node_prior, out_dir, class_names=EMOTION_NAMES)
    print(f"Saved D5B motif prior: {prior_path}")
    print(f"Saved prior metadata: {out_dir / 'node_prior_meta.json'}")
    print(f"Saved prior figures: {out_dir / 'figures'}")
    return prior_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d5b_1_fixed_motif_classifier.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_build(config, graph_repo_path=args.graph_repo_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
