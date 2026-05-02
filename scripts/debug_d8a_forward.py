"""Forward/backward smoke check for D8A Graph-Swin pre-part D6B."""

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
    if not torch.isfinite(value.float()).all():
        raise AssertionError(f"{key} contains NaN/Inf")


def check_config(config_path: str, device: torch.device) -> None:
    config = load_config(config_path)
    model_cfg = dict(config["model"])
    graph_cfg = dict(config.get("graph", {}) or {})
    model_cfg.setdefault("height", int(graph_cfg.get("height", 48)))
    model_cfg.setdefault("width", int(graph_cfg.get("width", 48)))
    model_cfg.setdefault("connectivity", int(graph_cfg.get("connectivity", 8)))

    model = build_model(model_cfg).to(device)
    criterion = build_loss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    batch_size = 2
    height = int(model_cfg.get("height", 48))
    width = int(model_cfg.get("width", 48))
    node_dim = int(model_cfg.get("node_dim", 7))
    edge_dim = int(model_cfg.get("edge_dim", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_nodes = height * width
    num_parts = int(model_cfg.get("num_part_slots", model_cfg.get("part_branch", {}).get("num_parts", 16)))
    edge_index = _make_grid_edges(height=height, width=width).to(device)
    batch = {
        "x": torch.randn(batch_size, num_nodes, node_dim, device=device),
        "node_features": None,
        "edge_index": edge_index,
        "edge_attr": torch.randn(batch_size, edge_index.shape[1], edge_dim, device=device),
        "node_mask": torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device),
        "y": torch.tensor([0, 6], dtype=torch.long, device=device),
        "graph_id": torch.arange(batch_size, device=device),
    }
    batch["node_features"] = batch["x"]

    out = model(batch)
    loss_dict = criterion(out, batch["y"], batch)
    loss = loss_dict["loss"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite loss for {config_path}: {float(loss.detach().cpu())}")
    loss.backward()

    model_name = str(model_cfg.get("name", ""))
    _expect_shape(out, "logits", (batch_size, 7))
    if model_name in {"graph_swin_prepart_d6b_d8a", "d8a_graph_swin_prepart_d6b"}:
        _expect_shape(out, "h_pixel", (batch_size, num_nodes, hidden_dim))
        _expect_shape(out, "h_context", (batch_size, num_nodes, hidden_dim))
        _expect_shape(out, "enhanced_h_pixel", (batch_size, num_nodes, hidden_dim))
        _expect_shape(out, "part_masks", (batch_size, num_parts, num_nodes))
        _expect_shape(out, "part_features", (batch_size, num_parts, hidden_dim))
        _expect_shape(out, "part_context", (batch_size, num_parts, hidden_dim))
        _expect_shape(out, "class_part_attn", (batch_size, 7, num_parts))
        _expect_shape(out, "class_pixel_motif", (batch_size, 7, num_nodes))
        mask_sum_err = (out["part_masks"].sum(dim=1) - 1.0).abs().max()
        if float(mask_sum_err.detach().cpu()) > 1e-4:
            raise AssertionError(f"part_masks.sum(dim=1) max error too high: {float(mask_sum_err.detach().cpu())}")
        alpha_param = getattr(model, "context_alpha", None)
        alpha_grad = getattr(alpha_param, "grad", None)
        if not torch.is_tensor(alpha_grad) or not torch.isfinite(alpha_grad).all():
            raise AssertionError("context alpha grad missing or non-finite")
        alpha_value = float(out["context_alpha"].detach().cpu())
        alpha_grad_value = float(alpha_grad.detach().cpu())
        diag = out.get("diagnostics", {})
        ratio = diag.get("context_to_pixel_ratio")
        ratio_value = float(ratio.detach().cpu()) if torch.is_tensor(ratio) else float("nan")
        print(
            f"OK D8A {Path(config_path).name}: loss={float(loss.detach().cpu()):.4f} "
            f"logits={tuple(out['logits'].shape)} h_context={tuple(out['h_context'].shape)} "
            f"part_masks={tuple(out['part_masks'].shape)} mask_sum_err={float(mask_sum_err.detach().cpu()):.2e} "
            f"alpha={alpha_value:.4f} alpha_grad={alpha_grad_value:.4e} "
            f"context_to_pixel_ratio={ratio_value:.4f}"
        )
    else:
        print(
            f"OK compat {Path(config_path).name}: model={model_name} "
            f"loss={float(loss.detach().cpu()):.4f} logits={tuple(out['logits'].shape)}"
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8a_graph_swin_prepart_d6b.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check_config(args.config, torch.device(args.device))


if __name__ == "__main__":
    main()
