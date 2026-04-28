"""Prepare a full FER-2013 graph_repo under /kaggle/working.

The fast path on Kaggle is to build or copy the graph repository into the
writable working disk, then benchmark/train from there instead of repeatedly
reading chunks from /kaggle/input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


EXPECTED = {
    "train_samples": 28709,
    "val_samples": 3589,
    "test_samples": 3589,
    "num_nodes": 2304,
    "num_edges": 17860,
    "node_dim": 7,
    "edge_dim": 5,
}


def _find_actual_repo_root(base: Path) -> Path | None:
    if not base.exists():
        return None
    if not base.is_dir():
        return None
    if (base / "manifest.pt").exists() or list((base / "train").glob("chunk_*.pt")):
        return base
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "manifest.pt").exists() or list((child / "train").glob("chunk_*.pt")):
            return child
    return None


def _finite(name: str, tensor) -> None:
    import torch

    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"{name} finite check failed")


def inspect_working_graph_repo(repo_root: str | Path, require_full: bool = True) -> Dict[str, Any]:
    """Inspect graph_repo and return the metrics required by the IO benchmark."""

    from data.graph_repository import GraphRepositoryReader
    from data.graph_resolver import GraphResolver

    requested = Path(repo_root)
    actual = _find_actual_repo_root(requested)
    if actual is None:
        raise FileNotFoundError(f"No graph repo found under: {requested}")

    reader = GraphRepositoryReader(actual)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)

    result: Dict[str, Any] = {
        "requested_path": str(requested),
        "path": str(actual),
        "train_samples": reader.split_size("train"),
        "val_samples": reader.split_size("val"),
        "test_samples": reader.split_size("test"),
        "num_nodes": int(shared.height * shared.width),
        "num_edges": int(shared.edge_index.shape[1]),
        "node_dim": None,
        "edge_dim": None,
        "finite_check": "PASS",
        "inspect_status": "PASS",
    }

    _finite("edge_attr_static", shared.edge_attr_static)

    for split in ("train", "val", "test"):
        chunk = reader.load_chunk(split, 0)
        if not chunk:
            raise AssertionError(f"{split} chunk_000.pt is empty")
        sample = chunk[0]
        resolved = resolver.resolve(sample)
        _finite(f"{split}.node_features", sample.node_features)
        _finite(f"{split}.edge_attr_dynamic", sample.edge_attr_dynamic)
        _finite(f"{split}.resolved.edge_attr", resolved.edge_attr)
        if split == "train":
            result["node_dim"] = int(sample.node_features.shape[1])
            result["edge_dim"] = int(resolved.edge_attr.shape[1])

    if require_full:
        for key, expected in EXPECTED.items():
            actual_value = result.get(key)
            if actual_value != expected:
                raise AssertionError(f"{key}: expected {expected}, got {actual_value}")

    return result


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _csv_root_if_available(csv_root: str | None) -> Path | None:
    from common import find_csv_root

    try:
        return find_csv_root(csv_root or "auto")
    except Exception:
        return None


def build_to_working(
    csv_root: str | None,
    working_graph_repo: str | Path,
    config_path: str = "configs/d5a.yaml",
    environment: str | None = None,
    force: bool = False,
) -> Path:
    from build_graph_repo import build_graph_repository
    from common import load_config
    from data.graph_config import GraphConfig

    dst = Path(working_graph_repo)
    if force:
        _remove_tree(dst)
    cfg = load_config(config_path, environment=environment)
    graph_cfg = GraphConfig.from_dict(cfg.get("graph", {}))
    build_graph_repository(
        csv_root=csv_root or "auto",
        repo_root=dst,
        graph_config=graph_cfg,
        max_samples_per_split=None,
        overwrite=True,
    )
    return dst


def copy_to_working(
    input_graph_repo: str | Path,
    working_graph_repo: str | Path,
    force: bool = False,
) -> Path:
    src_requested = Path(input_graph_repo)
    src = _find_actual_repo_root(src_requested)
    if src is None:
        raise FileNotFoundError(f"No input graph repo found under: {src_requested}")

    dst = Path(working_graph_repo)
    if dst.exists():
        if not force:
            inspect_working_graph_repo(dst, require_full=True)
            print(f"Working graph repo already exists and passed inspect: {dst}")
            return dst
        _remove_tree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run([rsync, "-a", f"{src}/", f"{dst}/"], check=True)
    else:
        shutil.copytree(src, dst)
    return dst


def prepare_graph_repo(
    method: str,
    csv_root: str | None,
    input_graph_repo: str | None,
    working_graph_repo: str | Path,
    config_path: str = "configs/d5a.yaml",
    environment: str | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    start = time.perf_counter()
    method = str(method).lower()
    working = Path(working_graph_repo)
    used = method

    if method == "auto":
        if working.exists():
            try:
                inspect = inspect_working_graph_repo(working, require_full=True)
                return {
                    **inspect,
                    "method": "existing",
                    "prepare_time_sec": time.perf_counter() - start,
                }
            except Exception as exc:
                print(f"Existing working graph repo did not pass inspect: {exc}")

        csv_path = _csv_root_if_available(csv_root)
        if csv_path is not None:
            used = "build"
            build_to_working(str(csv_path), working, config_path, environment, force=force)
        elif input_graph_repo and Path(input_graph_repo).exists():
            used = "copy"
            copy_to_working(input_graph_repo, working, force=force)
        else:
            raise FileNotFoundError(
                "Could not prepare graph_repo: no valid working repo, csv_root, or input_graph_repo."
            )
    elif method == "build":
        build_to_working(csv_root, working, config_path, environment, force=force)
    elif method == "copy":
        if not input_graph_repo:
            raise ValueError("--input_graph_repo is required when --method copy")
        copy_to_working(input_graph_repo, working, force=force)
    else:
        raise ValueError(f"Unknown method: {method}")

    inspect = inspect_working_graph_repo(working, require_full=True)
    inspect["method"] = used
    inspect["prepare_time_sec"] = time.perf_counter() - start
    return inspect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--input_graph_repo", default=None)
    parser.add_argument("--working_graph_repo", required=True)
    parser.add_argument("--method", choices=["build", "copy", "auto"], default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = prepare_graph_repo(
        method=args.method,
        csv_root=args.csv_root,
        input_graph_repo=args.input_graph_repo,
        working_graph_repo=args.working_graph_repo,
        config_path=args.config,
        environment=args.environment,
        force=args.force,
    )

    print("\nGraph repo prepare result:")
    print(f"  method: {result['method']}")
    print(f"  path: {result['path']}")
    print(
        "  samples: "
        f"train={result['train_samples']} val={result['val_samples']} test={result['test_samples']}"
    )
    print(
        "  graph: "
        f"nodes={result['num_nodes']} edges={result['num_edges']} "
        f"node_dim={result['node_dim']} edge_dim={result['edge_dim']}"
    )
    print(f"  finite_check: {result['finite_check']}")
    print(f"  inspect: {result['inspect_status']}")
    print(f"  prepare_time_sec: {result['prepare_time_sec']:.2f}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
