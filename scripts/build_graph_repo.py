"""Build D5 graph repository from FER-2013 split CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from common import find_csv_root, load_config, resolve_path, split_csv_paths

from fer_d5.data.graph_builder import PixelGraphBuilder, SharedGraphBuilder
from fer_d5.data.graph_config import GraphConfig
from fer_d5.data.graph_repository import GraphRepositoryWriter
from fer_d5.data.raw_dataset import RawFERDataset


def build_graph_repository(
    csv_root: str | Path | None,
    repo_root: str | Path,
    graph_config: GraphConfig,
    max_samples_per_split: Optional[int] = None,
    overwrite: bool = True,
) -> Path:
    csv_paths = split_csv_paths(csv_root)
    repo_root = Path(repo_root)
    shared = SharedGraphBuilder(graph_config).build()
    writer = GraphRepositoryWriter(repo_root=repo_root, config=graph_config, overwrite=overwrite)
    writer.write_shared(shared)
    builder = PixelGraphBuilder(graph_config, shared)

    for split, csv_path in csv_paths.items():
        raw_ds = RawFERDataset(
            csv_path=csv_path,
            split=split,
            image_size=graph_config.height,
            max_samples=max_samples_per_split,
        )
        with writer.open_split(split) as split_writer:
            for raw_sample in tqdm(raw_ds, desc=f"build {split}"):
                graph = builder.build(raw_sample)
                split_writer.add(graph)
    writer.save_manifest()
    print(f"Graph repo written to: {repo_root}")
    print(f"Expected nodes={graph_config.num_nodes} edges={shared.edge_index.shape[1]}")
    return repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--csv_root", default="auto")
    parser.add_argument("--repo_root", "--graph_repo_path", dest="repo_root", default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--connectivity", type=int, default=None)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    cfg: Dict = load_config(args.config) if args.config else {}
    graph_cfg = GraphConfig.from_dict(cfg.get("graph", {}))
    if args.chunk_size is not None:
        graph_cfg.chunk_size = int(args.chunk_size)
    if args.connectivity is not None:
        graph_cfg.connectivity = int(args.connectivity)
    repo_root = args.repo_root
    if repo_root is None:
        repo_root = cfg.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo")
    repo_path = resolve_path(repo_root)
    csv_root = args.csv_root if args.csv_root is not None else cfg.get("paths", {}).get("csv_root", "auto")
    if str(csv_root).lower() == "auto":
        print(f"Auto CSV root: {find_csv_root(csv_root)}")
    build_graph_repository(
        csv_root=csv_root,
        repo_root=repo_path,
        graph_config=graph_cfg,
        max_samples_per_split=args.max_samples_per_split,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
