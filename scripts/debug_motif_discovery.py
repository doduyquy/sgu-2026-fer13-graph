"""Debug MotifDiscoveryModule on small FER graph batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_existing_path  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.motif_losses import MotifDiscoveryStage1Loss  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402


def _make_grid_edges(height: int = 48, width: int = 48) -> torch.Tensor:
    edges = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x + 1 < width:
                right = y * width + x + 1
                edges.append((idx, right))
                edges.append((right, idx))
            if y + 1 < height:
                down = (y + 1) * width + x
                edges.append((idx, down))
                edges.append((down, idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _synthetic_loader(batch_size: int, num_batches: int, device: torch.device):
    height = width = 48
    node_dim = 7
    edge_dim = 5
    edge_index = _make_grid_edges(height, width).to(device)
    for batch_idx in range(int(num_batches)):
        yield {
            "graph_id": torch.arange(batch_size, device=device) + batch_idx * batch_size,
            "x": torch.randn(batch_size, height * width, node_dim, device=device),
            "node_features": torch.randn(batch_size, height * width, node_dim, device=device),
            "edge_index": edge_index,
            "edge_attr": torch.randn(batch_size, edge_index.shape[1], edge_dim, device=device),
            "node_mask": torch.ones(batch_size, height * width, dtype=torch.bool, device=device),
            "y": torch.zeros(batch_size, dtype=torch.long, device=device),
        }


def _shape(value: torch.Tensor) -> str:
    return str(list(value.shape))


def _scalar_dict(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for key, value in values.items():
        if torch.is_tensor(value) and value.ndim == 0:
            out[key] = float(value.detach().cpu())
    return out


def _assert_outputs(out: Dict[str, torch.Tensor], batch_size: int, num_motifs: int, hidden_dim: int) -> None:
    embeddings = out["motif_embeddings"]
    maps = out["motif_assignment_maps"]
    scores = out["motif_scores"]
    if tuple(embeddings.shape) != (batch_size, num_motifs, hidden_dim):
        raise AssertionError(f"motif_embeddings shape {tuple(embeddings.shape)}")
    if tuple(maps.shape) != (batch_size, num_motifs, 48, 48):
        raise AssertionError(f"motif_assignment_maps shape {tuple(maps.shape)}")
    if tuple(scores.shape) != (batch_size, num_motifs):
        raise AssertionError(f"motif_scores shape {tuple(scores.shape)}")
    for key in ("motif_embeddings", "motif_assignment_maps", "motif_scores", "motif_centers"):
        if not torch.isfinite(out[key]).all():
            raise AssertionError(f"{key} contains NaN/Inf")
    assignment_sum = maps.sum(dim=(2, 3))
    if not torch.allclose(assignment_sum, torch.ones_like(assignment_sum), atol=1e-4, rtol=1e-4):
        raise AssertionError(f"assignment maps are not spatial-normalized; max_err={float((assignment_sum - 1).abs().max())}")
    audit = out.get("motif_audit", {})
    for key in (
        "mean_pairwise_map_sim",
        "max_pairwise_map_sim",
        "mean_pairwise_emb_sim",
        "max_pairwise_emb_sim",
        "mean_center_dist",
        "min_center_dist",
        "redundant_pair_count",
        "border_mass_mean",
        "center_mass_mean",
        "effective_motif_count",
    ):
        if key not in audit or not torch.is_tensor(audit[key]):
            raise AssertionError(f"Missing audit metric: {key}")


@torch.no_grad()
def run_debug(
    config,
    split: str,
    num_batches: int,
    synthetic: bool = False,
    checkpoint: str | Path | None = None,
) -> None:
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config=config)
    model_cfg = dict(config["model"])
    model = build_model(model_cfg).to(device)
    if checkpoint is not None:
        ckpt_path = resolve_existing_path(checkpoint)
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded motif discovery checkpoint: {ckpt_path}")
    model.eval()
    criterion = MotifDiscoveryStage1Loss(config.get("motif_loss", config.get("motif", {}).get("loss", {}))).to(device)
    criterion.eval()

    if synthetic:
        batch_size = int(config.get("data", {}).get("batch_size", 4))
        loader = _synthetic_loader(batch_size=batch_size, num_batches=num_batches, device=device)
    else:
        loader = build_dataloader(config, split=split, shuffle=False)

    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_motifs = int(model_cfg.get("num_motifs", 16))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(num_batches):
            break
        if not synthetic:
            batch = move_to_device(batch, device)
        out = model(batch)
        loss_dict = criterion(out, batch)
        bsz = int(batch["x"].shape[0])
        _assert_outputs(out, batch_size=bsz, num_motifs=num_motifs, hidden_dim=hidden_dim)
        assignment_sum = out["motif_assignment_maps"].sum(dim=(2, 3))
        audit = out["motif_audit"]

        print(f"\n[batch {batch_idx}]")
        print(f"h_pixel: {_shape(out['h_pixel'])}")
        print(f"motif_embeddings: {_shape(out['motif_embeddings'])}")
        print(f"motif_assignment_maps: {_shape(out['motif_assignment_maps'])}")
        print(f"motif_scores: {_shape(out['motif_scores'])}")
        print(f"assignment_sum_mean: {float(assignment_sum.mean().cpu()):.6f}")
        print(f"assignment_sum_max_abs_err: {float((assignment_sum - 1.0).abs().max().cpu()):.8f}")
        for key in (
            "mean_pairwise_map_sim",
            "max_pairwise_map_sim",
            "mean_pairwise_emb_sim",
            "max_pairwise_emb_sim",
            "mean_center_dist",
            "min_center_dist",
            "redundant_pair_count",
            "redundant_pair_ratio",
            "border_mass_mean",
            "center_mass_mean",
            "effective_motif_count",
        ):
            print(f"{key}: {float(audit[key].detach().cpu()):.6f}")
        for key, value in _scalar_dict(loss_dict).items():
            print(f"{key}: {value:.6f}")
    print("\nMotif discovery debug checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8m_motif_discovery_debug.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--split", default="val")
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run without graph repo for local smoke checks.")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_debug(
        config,
        split=args.split,
        num_batches=args.num_batches,
        synthetic=bool(args.synthetic),
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
