"""Debug one D5A batch: shapes, forward, loss, backward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, prepare_training_objects
from training.trainer import move_to_device


def _stats(name: str, value: torch.Tensor) -> None:
    v = value.detach().float()
    print(
        f"{name:<24} shape={tuple(value.shape)} device={value.device} "
        f"min={v.min().item():.6f} max={v.max().item():.6f} "
        f"mean={v.mean().item():.6f} std={v.std(unbiased=False).item():.6f}"
    )


def run_debug(config) -> None:
    loader = build_dataloader(config, split="train", shuffle=False)
    batch = next(iter(loader))
    model, criterion, optimizer, _, device = prepare_training_objects(config)
    model.to(device)
    criterion.to(device)
    batch = move_to_device(batch, device)
    first_param = next(model.parameters())
    print(f"selected device: {device}")
    print(f"model first parameter device: {first_param.device}")
    print("Batch")
    for key in ("x", "node_features", "edge_index", "edge_attr", "node_mask", "y", "graph_id"):
        value = batch[key]
        if torch.is_tensor(value):
            _stats(key, value)
    out = model(batch)
    print("Model output")
    for key in ("logits", "node_attn", "edge_attn", "class_node_gate", "class_edge_gate"):
        _stats(key, out[key])

    assert tuple(out["logits"].shape) == (batch["x"].shape[0], 7)
    assert tuple(out["node_attn"].shape) == (batch["x"].shape[0], 7, 2304)
    assert tuple(out["edge_attn"].shape) == (batch["x"].shape[0], 7, 17860)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["node_attn"]).all()
    assert torch.isfinite(out["edge_attn"]).all()

    loss_dict = criterion(out, batch["y"], batch)
    for key, value in loss_dict.items():
        print(f"{key:<24} {float(value.detach().cpu().item()):.6f} device={value.device}")
    loss = loss_dict["loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=999.0)
    assert torch.isfinite(grad_norm)
    print(f"backward OK grad_norm={float(grad_norm.detach().cpu().item()):.6f}")
    print("Debug OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
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
    run_debug(config)


if __name__ == "__main__":
    main()
