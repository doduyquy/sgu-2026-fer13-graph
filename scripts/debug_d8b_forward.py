"""Forward/backward smoke check for D8B face-aware Graph-Swin."""

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


def _expected_counts(model, model_cfg) -> tuple[int, int]:
    branch = getattr(model, "swin_branch", None)
    if branch is not None:
        num_windows = int(getattr(branch, "num_windows"))
        if bool(getattr(branch, "region_merge")):
            num_regions = (int(getattr(branch, "num_win_h")) // 2) * (int(getattr(branch, "num_win_w")) // 2)
        else:
            num_regions = num_windows
        return num_windows, num_regions
    graph_swin = dict(model_cfg.get("graph_swin", {}) or {})
    height = int(model_cfg.get("height", 48))
    width = int(model_cfg.get("width", 48))
    window_size = int(graph_swin.get("window_size", 6))
    win_h = height // window_size
    win_w = width // window_size
    num_windows = win_h * win_w
    num_regions = (win_h // 2) * (win_w // 2) if bool(graph_swin.get("region_merge", True)) else num_windows
    return num_windows, num_regions


def _assert_param_grads(model: torch.nn.Module, label: str, fragments: tuple[str, ...]) -> None:
    checked = 0
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in fragments):
            checked += 1
            if param.requires_grad:
                if param.grad is None:
                    raise AssertionError(f"{label} parameter has no grad: {name}")
                if not torch.isfinite(param.grad.float()).all():
                    raise AssertionError(f"{label} parameter has NaN/Inf grad: {name}")
    if checked == 0:
        raise AssertionError(f"No parameters matched {label}: {fragments}")


def check_config(config_path: str, device: torch.device) -> None:
    config = load_config(config_path)
    model_cfg = dict(config["model"])
    graph_cfg = dict(config.get("graph", {}) or {})
    model_cfg.setdefault("height", int(graph_cfg.get("height", 48)))
    model_cfg.setdefault("width", int(graph_cfg.get("width", 48)))
    model_cfg.setdefault("connectivity", int(graph_cfg.get("connectivity", 8)))

    model = build_model(model_cfg).to(device)
    loss_cfg = dict(config["loss"])
    loss_cfg.setdefault("height", int(model_cfg.get("height", 48)))
    loss_cfg.setdefault("width", int(model_cfg.get("width", 48)))
    criterion = build_loss(loss_cfg).to(device)
    model.train()

    batch_size = 2
    height = int(model_cfg.get("height", 48))
    width = int(model_cfg.get("width", 48))
    node_dim = int(model_cfg.get("node_dim", 7))
    edge_dim = int(model_cfg.get("edge_dim", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_nodes = height * width
    num_windows, num_regions = _expected_counts(model, model_cfg)
    edge_index = _make_grid_edges(height=height, width=width).to(device)
    batch = {
        "x": torch.randn(batch_size, num_nodes, node_dim, device=device),
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

    _expect_shape(out, "logits", (batch_size, 7))
    _expect_shape(out, "h_pixel", (batch_size, num_nodes, hidden_dim))
    _expect_shape(out, "pixel_gate", (batch_size, num_nodes, 1))
    _expect_shape(out, "window_tokens", (batch_size, num_windows, hidden_dim))
    _expect_shape(out, "window_gate", (batch_size, num_windows, 1))
    _expect_shape(out, "region_tokens", (batch_size, num_regions, hidden_dim))
    _expect_shape(out, "region_gate", (batch_size, num_regions, 1))
    _expect_shape(out, "class_region_attn", (batch_size, 7, num_regions))

    _assert_param_grads(model, "pixel_gate", ("pixel_gate.net",))
    _assert_param_grads(model, "window_gate", ("swin_branch.window_gate.net",))
    _assert_param_grads(model, "region_gate", ("swin_branch.region_gate.net",))

    diagnostics = out.get("diagnostics", {})
    diag_text = []
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
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            diag_text.append(f"{key}={float(value.detach().cpu()):.4f}")
    print(
        f"OK D8B {Path(config_path).name}: loss={float(loss.detach().cpu()):.4f} "
        f"logits={tuple(out['logits'].shape)} h_pixel={tuple(out['h_pixel'].shape)} "
        f"window_tokens={tuple(out['window_tokens'].shape)} region_tokens={tuple(out['region_tokens'].shape)} "
        f"class_region_attn={tuple(out['class_region_attn'].shape)} "
        + " ".join(diag_text)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d8b_face_aware_graph_swin.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check_config(args.config, torch.device(args.device))


if __name__ == "__main__":
    main()
