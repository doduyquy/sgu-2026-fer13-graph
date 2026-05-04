"""Audit learned Stage 2C semantic projection embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_frozen_motif_separability import _evaluate, _write_pca_plot, _write_summary_csv  # noqa: E402
from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path, save_config  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.motif_graph_builder import build_motif_graph  # noqa: E402
from utils.motif_stage1_loader import load_frozen_motif_model  # noqa: E402


def _update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    output = dict(cfg.get("output", {}) or {})
    if getattr(args, "output_dir", None):
        paths["resolved_output_root"] = str(args.output_dir)
        output["dir"] = str(args.output_dir)
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
    if sys.platform.startswith("win") and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    cfg["output"] = output
    return cfg


def _extract_split(
    *,
    stage1_model: torch.nn.Module,
    projector: torch.nn.Module,
    loader,
    stage2_cfg: Dict[str, Any],
    device: torch.device,
    max_batches: int | None,
    split: str,
) -> Dict[str, torch.Tensor]:
    z_values: list[torch.Tensor] = []
    pooled_repr: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    logits_values: list[torch.Tensor] = []
    graph_ids: list[torch.Tensor] = []
    selected_indices: list[torch.Tensor] = []
    stage1_model.eval()
    projector.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(batch).items()}
            motif_graph = build_motif_graph(outputs, stage2_cfg)
            projected = projector(motif_graph["node_features"], selected_weights=motif_graph["selected_weights"])
            z_values.append(projected["z"].detach().float().cpu())
            pooled_repr.append(projected["pooled_repr"].detach().float().cpu())
            logits_values.append(projected["logits"].detach().float().cpu())
            labels.append(batch["y"].detach().long().cpu())
            selected_indices.append(motif_graph["selected_indices"].detach().long().cpu())
            if "graph_id" in batch and torch.is_tensor(batch["graph_id"]):
                graph_ids.append(batch["graph_id"].detach().cpu())
    if not labels:
        raise RuntimeError(f"No samples extracted for split={split!r}")
    out = {
        "pooled_repr": torch.cat(z_values, dim=0),
        "z": torch.cat(z_values, dim=0),
        "raw_pooled_repr": torch.cat(pooled_repr, dim=0),
        "logits": torch.cat(logits_values, dim=0),
        "labels": torch.cat(labels, dim=0),
        "selected_indices": torch.cat(selected_indices, dim=0),
    }
    if graph_ids:
        out["graph_ids"] = torch.cat(graph_ids, dim=0)
    print(f"[Extract {split}] samples={out['labels'].numel()} z={tuple(out['z'].shape)}")
    return out


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _write_npz(path: Path, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor]) -> None:
    payload = {}
    for split, data in (("train", train), ("val", val)):
        for key, value in data.items():
            payload[f"{split}_{key}"] = _as_numpy(value)
    np.savez_compressed(path, **payload)


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(resolve_existing_path(checkpoint_path), map_location=device)


def _load_projector(config: Dict[str, Any], checkpoint_path: str | Path, checkpoint: Dict[str, Any], stage1_model, loader, device, stage2_cfg):
    first_batch = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        stage1_outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(first_batch).items()}
        first_graph = build_motif_graph(stage1_outputs, stage2_cfg)
    model_cfg = dict(config.get("model", {}) or {})
    model_cfg["input_dim"] = int(first_graph["node_features"].shape[-1])
    projector = build_model(model_cfg).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint missing model_state_dict/state_dict: {checkpoint_path}")
    projector.load_state_dict(state)
    projector.eval()
    print(f"[Projector] checkpoint_loaded={checkpoint_path}")
    print(f"[Projector] node_feature_dim={model_cfg['input_dim']} z_dim={int(model_cfg.get('projection_dim', 128))}")
    return projector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--train_batches", type=int, default=60)
    parser.add_argument("--val_batches", type=int, default=60)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--no_pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(args.device, config=config)
    stage1_cfg = dict(config.get("stage1", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    default_root = PROJECT_ROOT / "outputs" / f"{Path(args.config).stem}_projection_audit"
    output_root = resolve_path(config.get("audit", {}).get("dir") or args.output_dir) or default_root
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)

    stage1_model = load_frozen_motif_model(stage1_cfg["config"], stage1_cfg["checkpoint"], device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    if "stage1_state_dict" in checkpoint:
        missing, unexpected = stage1_model.load_state_dict(checkpoint["stage1_state_dict"], strict=False)
        if missing:
            print(f"[Stage1Audit] missing_keys={list(missing)}")
        if unexpected:
            print(f"[Stage1Audit] unexpected_keys={list(unexpected)}")
        print("[Stage1Audit] loaded fine-tuned stage1_state_dict from checkpoint")
    train_loader = build_dataloader(config, split="train", shuffle=False)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    projector = _load_projector(config, args.checkpoint, checkpoint, stage1_model, train_loader, device, stage2_cfg)
    train = _extract_split(stage1_model=stage1_model, projector=projector, loader=train_loader, stage2_cfg=stage2_cfg, device=device, max_batches=args.train_batches, split="train")
    val = _extract_split(stage1_model=stage1_model, projector=projector, loader=val_loader, stage2_cfg=stage2_cfg, device=device, max_batches=args.val_batches, split="val")
    metrics = _evaluate(train, val, num_classes=int(args.num_classes))

    torch.save({"train": train, "val": val, "metrics": metrics, "config": config}, output_root / "projected_features.pt")
    _write_npz(output_root / "projected_features.npz", train, val)
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_summary_csv(output_root / "summary.csv", metrics)
    if not args.no_pca:
        _write_pca_plot(output_root / "figures" / "pca_projected_z.png", train, val)

    print(f"[Output] projected_features_pt={output_root / 'projected_features.pt'}")
    print(f"[Output] projected_features_npz={output_root / 'projected_features.npz'}")
    print(f"[Output] metrics={output_root / 'metrics.json'}")
    print(
        "[Summary] "
        f"centroid_acc={metrics['nearest_centroid_accuracy']:.4f} "
        f"knn1={metrics['knn1_accuracy']:.4f} "
        f"knn3={metrics['knn3_accuracy']:.4f} "
        f"knn5={metrics['knn5_accuracy']:.4f} "
        f"train_intra={metrics['train_cosine']['intra_class_cosine_mean']:.4f} "
        f"train_inter={metrics['train_cosine']['inter_class_cosine_mean']:.4f} "
        f"val_intra={metrics['val_cosine']['intra_class_cosine_mean']:.4f} "
        f"val_inter={metrics['val_cosine']['inter_class_cosine_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
