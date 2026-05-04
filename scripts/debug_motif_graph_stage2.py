"""Debug Stage 2A frozen-motif graph classifier wiring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402
from utils.motif_graph_builder import build_motif_graph  # noqa: E402
from utils.motif_stage1_loader import load_frozen_motif_model  # noqa: E402


def _update_config(config, args):
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    if getattr(args, "output_dir", None):
        paths["resolved_output_root"] = str(args.output_dir)
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    if sys.platform.startswith("win") and getattr(args, "num_batches", None) is not None and getattr(args, "num_workers", None) is None:
        data["num_workers"] = 0
        data["persistent_workers"] = False
        data["prefetch_factor"] = None
    cfg["paths"] = paths
    cfg["data"] = data
    return cfg


def _count_grads(model: torch.nn.Module) -> int:
    return sum(1 for param in model.parameters() if param.grad is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(args.device, config=config)
    stage1_cfg = dict(config.get("stage1", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    stage1_model = load_frozen_motif_model(stage1_cfg["config"], stage1_cfg["checkpoint"], device)
    loader = build_dataloader(config, split="train", shuffle=False)
    batch = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        stage1_outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in stage1_model(batch).items()}
        motif_graph = build_motif_graph(stage1_outputs, stage2_cfg)

    model_cfg = dict(config.get("model", {}) or {})
    model_cfg["input_dim"] = int(motif_graph["node_features"].shape[-1])
    classifier = build_model(model_cfg).to(device)
    logits = classifier(
        motif_graph["node_features"],
        motif_edge_features=motif_graph["edge_features"],
        selected_weights=motif_graph["selected_weights"],
    )
    labels = batch["y"].long()
    loss = F.cross_entropy(logits, labels)
    classifier.zero_grad(set_to_none=True)
    loss.backward()
    stage1_grad_count = _count_grads(stage1_model)
    stage2_grad_count = _count_grads(classifier)
    print(f"motif_embeddings={tuple(stage1_outputs['motif_embeddings'].shape)}")
    print(f"selection_weights={tuple(stage1_outputs.get('selection_weights', stage1_outputs['motif_scores']).shape)}")
    print(f"selected_embeddings={tuple(motif_graph['selected_embeddings'].shape)}")
    print(f"node_features={tuple(motif_graph['node_features'].shape)}")
    print(f"edge_features={tuple(motif_graph['edge_features'].shape)}")
    print(f"logits={tuple(logits.shape)}")
    print(f"loss={float(loss.detach().cpu()):.6f}")
    print(f"stage1_trainable_params={sum(p.numel() for p in stage1_model.parameters() if p.requires_grad)}")
    print(f"stage1_grad_count_after_backward={stage1_grad_count}")
    print(f"stage2_grad_count_after_backward={stage2_grad_count}")
    if stage1_grad_count != 0:
        raise RuntimeError("Stage 1 received gradients during Stage 2 debug")
    if stage2_grad_count <= 0:
        raise RuntimeError("Stage 2 classifier did not receive gradients")


if __name__ == "__main__":
    main()
