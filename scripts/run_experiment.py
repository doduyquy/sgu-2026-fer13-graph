"""Kaggle/local template entrypoint for D5A experiments."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, load_config, resolve_path
from build_graph_repo import build_graph_repository
from debug_d5a_batch import run_debug
from evaluate_d5a import run_evaluate
from inspect_graph_repo import inspect_graph_repository
from train_d5a import run_train
from visualize_d5 import run_visualize

from data.graph_config import GraphConfig


def _repo_ready(repo_path: Path) -> bool:
    return (
        repo_path.exists()
        and (repo_path / "manifest.pt").exists()
        and (repo_path / "shared" / "shared_graph.pt").exists()
        and (repo_path / "train").exists()
    )


def _build(config, max_samples_per_split=None) -> Path:
    graph_cfg = GraphConfig.from_dict(config.get("graph", {}))
    repo_path = resolve_path(config.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo"))
    csv_root = config.get("paths", {}).get("csv_root", "auto")
    return build_graph_repository(
        csv_root=csv_root,
        repo_root=repo_path,
        graph_config=graph_cfg,
        max_samples_per_split=max_samples_per_split,
        overwrite=True,
    )


def _smoke_config(config):
    cfg = dict(config)
    training = dict(cfg.get("training", {}))
    data = dict(cfg.get("data", {}))
    training["epochs"] = int(training.get("epochs") or 2)
    training["epochs"] = min(training["epochs"], 2)
    training["max_train_batches"] = int(training.get("max_train_batches") or 3)
    training["max_val_batches"] = int(training.get("max_val_batches") or 2)
    training["max_test_batches"] = int(training.get("max_test_batches") or 2)
    data["batch_size"] = int(data.get("batch_size", 16))
    cfg["training"] = training
    cfg["data"] = data
    return cfg


def _smoke_sample_count(config) -> int:
    training = config.get("training", {})
    data = config.get("data", {})
    batch_size = int(data.get("batch_size", 16))
    max_batches = max(
        int(training.get("max_train_batches") or 3),
        int(training.get("max_val_batches") or 2),
        int(training.get("max_test_batches") or 2),
    )
    return max(batch_size * max_batches, batch_size)


def _zip_outputs(config) -> Path:
    paths = config.get("paths", {})
    output_root = resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    zip_path = output_root.parent / f"{output_root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if output_root.exists():
            for path in output_root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(output_root.parent))
    print(f"Zipped outputs: {zip_path}")
    return zip_path


def run_mode(config, mode: str, checkpoint=None, zip_outputs: bool = False):
    mode = str(mode)
    repo_path = resolve_path(config.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo"))

    if mode == "build_graph":
        _build(config)
    elif mode == "inspect_graph":
        inspect_graph_repository(repo_path)
    elif mode == "debug":
        run_debug(config)
    elif mode == "smoke":
        config = _smoke_config(config)
        if not _repo_ready(repo_path):
            _build(config, max_samples_per_split=_smoke_sample_count(config))
        run_debug(config)
        run_train(config)
        run_evaluate(config)
    elif mode == "train":
        run_train(config)
    elif mode == "evaluate":
        run_evaluate(config, checkpoint=checkpoint)
    elif mode == "visualize":
        run_visualize(config, checkpoint=checkpoint)
    elif mode == "build_and_train":
        _build(config)
        inspect_graph_repository(repo_path)
        run_debug(config)
        run_train(config)
        run_evaluate(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "build_and_train":
        paths = config.get("paths", {})
        ckpt = checkpoint or resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs")) / "checkpoints" / "best.pth"
        if ckpt.exists():
            run_visualize(config, checkpoint=ckpt)
    if zip_outputs:
        _zip_outputs(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument(
        "--mode",
        default=None,
        choices=[
            "build_graph",
            "inspect_graph",
            "debug",
            "smoke",
            "train",
            "evaluate",
            "visualize",
            "build_and_train",
        ],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--zip_outputs", action="store_true", default=None)
    # --- Performance profiling & optimisation ---
    parser.add_argument("--profile_batches", type=int, default=None,
                        help="Profile the first N training batches with detailed timings.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader num_workers (0=main process).")
    parser.add_argument("--pin_memory", default=None,
                        help="DataLoader pin_memory (true/false).")
    parser.add_argument("--persistent_workers", default=None,
                        help="DataLoader persistent_workers (true/false, requires num_workers>0).")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="DataLoader prefetch_factor (requires num_workers>0).")
    parser.add_argument("--graph_cache_chunks", type=int, default=None,
                        help="Deprecated alias for --chunk_cache_size.")
    parser.add_argument("--chunk_cache_size", type=int, default=None,
                        help="Maximum graph repo chunks to keep in RAM (0 disables cache).")
    parser.add_argument("--chunk_aware_shuffle", action="store_true", default=False,
                        help="Use chunk-aware train batches so graph repo chunk cache can hit.")
    parser.add_argument("--no_chunk_aware_shuffle", action="store_true", default=False,
                        help="Disable chunk-aware train batches.")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable Automatic Mixed Precision (AMP) training.")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP even if set in config.")
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_cfg = config.get("run", {})
    mode = args.mode or run_cfg.get("mode", "smoke")
    zip_outputs = bool(args.zip_outputs if args.zip_outputs is not None else run_cfg.get("zip_outputs", False))
    config["run"] = {**run_cfg, "mode": mode, "zip_outputs": zip_outputs}
    print(f"resolved run.mode: {mode}")
    print(f"resolved run.zip_outputs: {zip_outputs}")
    run_mode(config, mode=mode, checkpoint=args.checkpoint, zip_outputs=zip_outputs)


if __name__ == "__main__":
    main()
