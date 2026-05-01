"""Build the D6D input-v2 graph repository with 12 node features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import find_csv_root, load_config, resolve_path
from build_graph_repo import build_graph_repository
from data.graph_config import GraphConfig, NODE_FEATURE_NAMES_V2


def d6d_graph_config(cfg: Dict, chunk_size: int | None = None) -> GraphConfig:
    graph_cfg = GraphConfig.from_dict(cfg.get("graph", {}))
    graph_cfg.node_feature_names = list(NODE_FEATURE_NAMES_V2)
    graph_cfg.version = "d6d_input_v2_node12"
    if chunk_size is not None:
        graph_cfg.chunk_size = int(chunk_size)
    return graph_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D6D input-v2 FER graph repo.")
    parser.add_argument("--config", default="configs/experiments/d6d_input_v2_build.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_dir", "--repo_root", "--graph_repo_path", dest="output_dir", default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    cfg: Dict = load_config(args.config, environment=args.environment) if args.config else {}
    graph_cfg = d6d_graph_config(cfg, chunk_size=args.chunk_size)
    output_dir = args.output_dir or cfg.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo_d6d_input_v2")
    repo_path = resolve_path(output_dir)
    csv_root = args.csv_root if args.csv_root is not None else cfg.get("paths", {}).get("csv_root", "auto")
    if str(csv_root).lower() == "auto":
        print(f"Auto CSV root: {find_csv_root(csv_root)}")
    print(f"D6D input-v2 node_dim={len(graph_cfg.node_feature_names)} edge_dim=5")
    print(f"node_feature_names={graph_cfg.node_feature_names}")
    build_graph_repository(
        csv_root=csv_root,
        repo_root=repo_path,
        graph_config=graph_cfg,
        max_samples_per_split=args.max_samples_per_split,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
