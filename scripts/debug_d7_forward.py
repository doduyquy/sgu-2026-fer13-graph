"""Fake-tensor forward checks for D7 Graph-Swin motif modes."""

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

from common import load_config  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.losses import build_loss  # noqa: E402


CONFIGS = {
    "d7a": "configs/experiments/d7a_graph_swin_standalone.yaml",
    "d7b": "configs/experiments/d7b_dual_branch_logits_fusion.yaml",
    "d7c": "configs/experiments/d7c_dual_branch_gated_fusion.yaml",
}


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


def _expect_shape(out, key: str, shape: tuple[int, ...]) -> None:
    value = out.get(key)
    if not torch.is_tensor(value):
        raise AssertionError(f"Missing tensor output: {key}")
    if tuple(value.shape) != shape:
        raise AssertionError(f"{key} shape {tuple(value.shape)} != {shape}")


def _expected_region_count(model, model_cfg) -> int:
    swin_branch = getattr(model, "swin_branch", None)
    if swin_branch is not None:
        return int(getattr(swin_branch, "num_windows")) if not getattr(swin_branch, "region_merge") else (
            int(getattr(swin_branch, "num_win_h")) // 2
        ) * (int(getattr(swin_branch, "num_win_w")) // 2)
    graph_swin = dict(model_cfg.get("graph_swin", {}) or {})
    height = int(model_cfg.get("height", 48))
    width = int(model_cfg.get("width", 48))
    window_size = int(graph_swin.get("window_size", 6))
    win_h = height // window_size
    win_w = width // window_size
    if bool(graph_swin.get("region_merge", True)):
        return (win_h // 2) * (win_w // 2)
    return win_h * win_w


def check_config(config_path: str, device: torch.device) -> None:
    config = load_config(config_path)
    model_cfg = dict(config["model"])
    model_cfg.setdefault("height", int(config.get("graph", {}).get("height", 48)))
    model_cfg.setdefault("width", int(config.get("graph", {}).get("width", 48)))
    model = build_model(model_cfg).to(device)
    criterion = build_loss(config["loss"]).to(device)
    model.train()

    batch_size = 2
    height = int(model_cfg.get("height", 48))
    width = int(model_cfg.get("width", 48))
    node_dim = int(model_cfg.get("node_dim", 7))
    edge_dim = int(model_cfg.get("edge_dim", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    expected_nodes = height * width
    expected_regions = _expected_region_count(model, model_cfg)
    edge_index = _make_grid_edges(height=height, width=width).to(device)
    edge_attr = torch.randn(batch_size, edge_index.shape[1], edge_dim, device=device)
    batch = {
        "x": torch.randn(batch_size, expected_nodes, node_dim, device=device),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.tensor([0, 6], dtype=torch.long, device=device),
        "graph_id": torch.arange(batch_size, device=device),
    }
    out = model(batch)
    loss_dict = criterion(out, batch["y"], batch)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite loss for {config_path}: {float(loss.detach().cpu())}")
    loss.backward()

    mode = str(model_cfg.get("mode"))
    _expect_shape(out, "logits", (2, 7))
    _expect_shape(out, "logits_swin", (2, 7))
    _expect_shape(out, "region_tokens", (2, expected_regions, hidden_dim))
    _expect_shape(out, "class_region_attn", (2, 7, expected_regions))
    if mode in ("logits_sum", "gated_class_repr"):
        _expect_shape(out, "logits_d6", (2, 7))
    if mode == "gated_class_repr":
        _expect_shape(out, "class_repr_d6", (2, 7, hidden_dim))
        _expect_shape(out, "class_repr_swin", (2, 7, hidden_dim))
        gate = out.get("fusion_gate")
        if not torch.is_tensor(gate) or tuple(gate.shape) not in ((2, 7, 1), (2, 7, hidden_dim)):
            raise AssertionError(f"fusion_gate shape {None if gate is None else tuple(gate.shape)}")
    print(
        f"OK {Path(config_path).name}: mode={mode} "
        f"loss={float(loss.detach().cpu()):.4f} logits={tuple(out['logits'].shape)} "
        f"region_tokens={tuple(out['region_tokens'].shape)} "
        f"class_region_attn={tuple(out['class_region_attn'].shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.config:
        check_config(args.config, device)
    else:
        for config_path in CONFIGS.values():
            check_config(config_path, device)


if __name__ == "__main__":
    main()
