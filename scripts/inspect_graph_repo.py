"""Inspect and validate a D5 graph repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import load_config, resolve_path

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver


def _finite(name: str, tensor: torch.Tensor) -> str:
    ok = bool(torch.isfinite(tensor).all().item())
    return f"{name}: finite={ok} min={tensor.float().min().item():.6f} max={tensor.float().max().item():.6f}"


def inspect_graph_repository(repo_root: str | Path) -> None:
    reader = GraphRepositoryReader(repo_root)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    print(f"repo_root: {Path(repo_root)}")
    print(f"shared exists: True")
    print(f"height={shared.height} width={shared.width} connectivity={shared.connectivity}")
    print(f"edge_index: {tuple(shared.edge_index.shape)}")
    print(f"edge_attr_static: {tuple(shared.edge_attr_static.shape)}")
    print(f"static feature names: {shared.static_feature_names}")
    print(_finite("edge_attr_static", shared.edge_attr_static))

    expected_nodes = int(shared.height * shared.width)
    expected_edges = 17860 if (shared.height, shared.width, shared.connectivity) == (48, 48, 8) else shared.edge_index.shape[1]
    if shared.edge_index.shape != (2, expected_edges):
        raise AssertionError(f"Unexpected edge_index shape: {shared.edge_index.shape}")

    for split in ("train", "val", "test"):
        paths = reader.chunk_paths(split)
        size = reader.split_size(split) if paths else 0
        print(f"split={split} chunks={len(paths)} samples={size}")
        if not paths:
            continue
        sample = reader.load_chunk(split, 0)[0]
        resolved = resolver.resolve(sample)
        print(f"  sample graph_id={sample.graph_id} label={sample.label}")
        print(f"  node_features: {tuple(sample.node_features.shape)} names={sample.node_feature_names}")
        print(f"  edge_attr_dynamic: {tuple(sample.edge_attr_dynamic.shape)} names={sample.dynamic_feature_names}")
        print(f"  resolved edge_attr: {tuple(resolved.edge_attr.shape)} names={resolved.edge_feature_names}")
        print(f"  {_finite('node_features', sample.node_features)}")
        print(f"  {_finite('edge_attr_dynamic', sample.edge_attr_dynamic)}")
        print(f"  {_finite('edge_attr', resolved.edge_attr)}")
        if sample.node_features.shape != (expected_nodes, 7):
            raise AssertionError("node_features shape must be [2304, 7]")
        if resolved.edge_attr.shape != (expected_edges, 5):
            raise AssertionError("edge_attr shape must be [17860, 5]")
    print("Inspection OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--repo_root", "--graph_repo_path", dest="repo_root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, environment=args.environment) if args.config else {}
    repo = args.repo_root or cfg.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo")
    inspect_graph_repository(resolve_path(repo))


if __name__ == "__main__":
    main()
