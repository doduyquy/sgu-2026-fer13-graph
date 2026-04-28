"""Run a single speed scenario and log results.

Changes v4:
- inspect_graph_repo() runs before training and emits [SPEED_BENCH] tags for
  graph_repo_path, train_samples, val_samples, test_samples.
- If train_samples < 1000 → status = FAIL_TINY_REPO (no result kept for best).
- batches_per_epoch is now len(train_loader) (actual), not ceil(28709/bs).
- est_epoch_min = sec_per_batch * len(train_loader) / 60  (actual, not hardcoded).
- bs_mismatch uses expected_first_batch = min(configured_bs, train_samples):
    if x.shape[0] != expected_first_batch → FAIL_BS_MISMATCH.
- --require_full_repo: abort if train_samples != 28709.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, create_trainer, load_config
import torch

FULL_TRAIN_SAMPLES = 28709   # FER-2013 reference – only used for require_full_repo check
TINY_REPO_THRESHOLD = 1000   # if train_samples < this → FAIL_TINY_REPO


class OutputTee:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def deep_merge(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def _find_actual_repo_root(base: Path) -> Path | None:
    """Search for the actual graph repo root under `base`.

    Kaggle datasets are sometimes mounted with an extra nesting level, e.g.:
      /kaggle/input/graph-repo/graph-repo/train/chunk_000.pt

    We look for:
      1. manifest.pt directly in `base`
      2. manifest.pt one level below
      3. train/chunk_*.pt directly in `base`
      4. train/chunk_*.pt one level below
    Returns the best candidate or None.
    """
    # Level 0: base itself
    if (base / "manifest.pt").exists():
        return base
    chunks_at_base = list((base / "train").glob("chunk_*.pt")) if (base / "train").exists() else []
    if chunks_at_base:
        return base

    # Level 1: immediate subdirectories
    best = None
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "manifest.pt").exists():
            return child   # manifest is the strongest signal
        if (child / "train").exists() and list((child / "train").glob("chunk_*.pt")):
            best = child   # chunks without manifest – keep looking for manifest
    return best


def _print_repo_tree(base: Path, max_depth: int = 2) -> None:
    """Print a compact directory tree for diagnostics."""
    def _walk(path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(path.iterdir())
        except PermissionError:
            return
        for i, child in enumerate(children[:20]):   # cap at 20 entries per dir
            connector = "└── " if i == len(children) - 1 else "├── "
            extra = f"  ({child.stat().st_size} B)" if child.is_file() else "/"
            print(f"  {prefix}{connector}{child.name}{extra}")
            if child.is_dir():
                extension = "    " if i == len(children) - 1 else "│   "
                _walk(child, depth + 1, prefix + extension)
        if len(children) > 20:
            print(f"  {prefix}  ... ({len(children) - 20} more)")

    print(f"[REPO_TREE] {base}")
    _walk(base, 0, "")


def inspect_graph_repo(config: dict) -> dict:
    """Inspect the graph repository and print diagnostic [SPEED_BENCH] tags.

    Auto-detects the actual repo root if the given path has extra nesting
    (common on Kaggle where datasets are mounted with an extra subdirectory).

    Returns a dict with keys:
        graph_repo_path         – resolved (possibly corrected) path
        graph_repo_path_orig    – path from config (before correction)
        train_samples, val_samples, test_samples
        is_tiny_repo (bool)
    """
    from common import resolve_path
    paths = config.get("paths", {})
    repo_raw = paths.get("graph_repo_path", "artifacts/graph_repo")
    repo = resolve_path(repo_raw)

    info: dict = {
        "graph_repo_path": str(repo) if repo else str(repo_raw),
        "graph_repo_path_orig": str(repo_raw),
        "train_samples": None,
        "val_samples": None,
        "test_samples": None,
        "is_tiny_repo": False,
    }

    print(f"[SPEED_BENCH] graph_repo_path_orig={repo_raw}")

    if repo is None or not repo.exists():
        print(f"[SPEED_BENCH] graph_repo_missing=True  path={info['graph_repo_path']}")
        info["is_tiny_repo"] = True
        return info

    # ---- Auto-detect actual repo root ----
    actual_root = _find_actual_repo_root(repo)
    if actual_root is None:
        print(
            f"[SPEED_BENCH] graph_repo_no_chunks=True  "
            f"(no manifest.pt or train/chunk_*.pt found under {repo})"
        )
        print("[REPO] Directory structure:")
        _print_repo_tree(repo)
        info["is_tiny_repo"] = True
        return info

    if actual_root != repo:
        print(
            f"[SPEED_BENCH] graph_repo_corrected=True  "
            f"orig={repo}  actual={actual_root}"
        )
        # Update config so the trainer also uses the corrected path
        config.setdefault("paths", {})["graph_repo_path"] = str(actual_root)
    info["graph_repo_path"] = str(actual_root)
    print(f"[SPEED_BENCH] graph_repo_path={actual_root}")

    # ---- Count samples via GraphRepositoryReader (no chunk loading) ----
    try:
        from data.graph_repository import GraphRepositoryReader
        reader = GraphRepositoryReader(actual_root)

        for split in ("train", "val", "test"):
            try:
                count = reader.split_size(split)
                info[f"{split}_samples"] = count
                print(f"[SPEED_BENCH] {split}_samples={count}")
            except Exception as exc:
                print(f"[SPEED_BENCH] {split}_samples=ERROR  ({exc})")
                # Try raw chunk glob as last resort
                chunk_pts = list((actual_root / split).glob("chunk_*.pt")) if (actual_root / split).exists() else []
                if chunk_pts:
                    print(f"[SPEED_BENCH]   {split}: found {len(chunk_pts)} chunk files (need to load to count)")

    except Exception as exc:
        print(f"[SPEED_BENCH] repo_inspect_error={exc}")

    train_n = info.get("train_samples") or 0
    info["is_tiny_repo"] = train_n < TINY_REPO_THRESHOLD
    if info["is_tiny_repo"]:
        print(
            f"[SPEED_BENCH] tiny_repo=True  "
            f"train_samples={train_n} < threshold={TINY_REPO_THRESHOLD}"
        )
        # Print tree to help diagnose
        print("[REPO] Directory structure:")
        _print_repo_tree(actual_root)
    else:
        print(f"[SPEED_BENCH] tiny_repo=False  train_samples={train_n}")

    return info


def parse_profile_output(text: str) -> dict:
    """Parse captured stdout into a profile dict.

    Priority order for sec_per_batch:
    1. [PROFILE average first N batches] block (most accurate).
    2. epoch line  train_sec/batch=Xs  (single-epoch average).

    est_epoch_min = sec_per_batch * len_train_loader / 60
    (len_train_loader = actual DataLoader length on the full repo)
    """
    profile: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Per-batch PROFILE blocks – take the last for peak VRAM.          #
    # ------------------------------------------------------------------ #
    per_batch_blocks = list(re.finditer(
        r"\[PROFILE batch=(\d+)\]"
        r".*?cuda_allocated_gb\s*=\s*([0-9.]+)"
        r".*?cuda_reserved_gb\s*=\s*([0-9.]+)"
        r".*?cuda_max_allocated_gb\s*=\s*([0-9.]+)",
        text, re.DOTALL,
    ))
    profile["profile_batches_recorded"] = len(per_batch_blocks)
    if per_batch_blocks:
        last = per_batch_blocks[-1]
        profile["cuda_allocated_gb"] = float(last.group(2))
        profile["cuda_reserved_gb"] = float(last.group(3))
        profile["cuda_max_allocated_gb"] = float(last.group(4))

    # ------------------------------------------------------------------ #
    # 2. [PROFILE average first N batches] block                          #
    # ------------------------------------------------------------------ #
    avg_match = re.search(
        r"\[PROFILE average first (\d+) batches \(recorded=(\d+)\)\](.*?)"
        r"estimated_full_epoch_minutes=([0-9.]+|unknown)",
        text, re.DOTALL,
    )
    # Fallback for old format without recorded= annotation
    if avg_match is None:
        avg_match = re.search(
            r"\[PROFILE average first (\d+) batches\](.*?)"
            r"estimated_full_epoch_minutes=([0-9.]+|unknown)",
            text, re.DOTALL,
        )
        if avg_match:
            profile["profile_batches_requested"] = int(avg_match.group(1))
            profile["profile_batches_recorded"] = int(avg_match.group(1))
            avg_block = avg_match.group(2)
            est_min_str = avg_match.group(3).strip()
        else:
            avg_block = None
            est_min_str = None
    else:
        profile["profile_batches_requested"] = int(avg_match.group(1))
        profile["profile_batches_recorded"] = int(avg_match.group(2))
        avg_block = avg_match.group(3)
        est_min_str = avg_match.group(4).strip()

    if avg_block is not None:
        metrics_map = {
            "avg_data_time":      r"avg_data_time\s*=\s*([0-9.]+)s",
            "avg_to_device_time": r"avg_to_device_time\s*=\s*([0-9.]+)s",
            "avg_forward_time":   r"avg_forward_time\s*=\s*([0-9.]+)s",
            "avg_loss_time":      r"avg_loss_time\s*=\s*([0-9.]+)s",
            "avg_backward_time":  r"avg_backward_time\s*=\s*([0-9.]+)s",
            "avg_optimizer_time": r"avg_optimizer_time\s*=\s*([0-9.]+)s",
            "train_sec_per_batch": r"avg_batch_time\s*=\s*([0-9.]+)s",
        }
        for k, pattern in metrics_map.items():
            m = re.search(pattern, avg_block)
            if m:
                profile[k] = float(m.group(1))

        # The trainer's estimated_full_epoch_minutes is computed from len(loader),
        # so it is the authoritative value when the repo is full.
        if est_min_str and est_min_str != "unknown":
            profile["estimated_train_min_per_epoch"] = float(est_min_str)

    # ------------------------------------------------------------------ #
    # 3. Fallback: epoch line  train_sec/batch=Xs                         #
    # ------------------------------------------------------------------ #
    if "train_sec_per_batch" not in profile:
        epoch_match = re.search(r"train_sec/batch=([0-9.]+)s", text)
        if epoch_match:
            profile["train_sec_per_batch"] = float(epoch_match.group(1))

    # ------------------------------------------------------------------ #
    # 4. first_batch shapes                                               #
    # ------------------------------------------------------------------ #
    x_shape_m = re.search(r"\[SPEED_BENCH\] first_batch_x_shape=(\[[^\]]+\])", text)
    if x_shape_m:
        try:
            profile["first_batch_x_shape"] = json.loads(x_shape_m.group(1))
        except Exception:
            profile["first_batch_x_shape"] = x_shape_m.group(1)

    ea_shape_m = re.search(r"\[SPEED_BENCH\] first_batch_edge_attr_shape=(\[[^\]]+\])", text)
    if ea_shape_m:
        try:
            profile["first_batch_edge_attr_shape"] = json.loads(ea_shape_m.group(1))
        except Exception:
            profile["first_batch_edge_attr_shape"] = ea_shape_m.group(1)

    # ------------------------------------------------------------------ #
    # 5. Scenario-level integer fields from [SPEED_BENCH] tags            #
    # ------------------------------------------------------------------ #
    for key in (
        "actual_batch_size",
        "len_train_loader",
        "len_train_dataset",
        "batches_per_epoch",
        "train_samples",
        "val_samples",
        "test_samples",
        "max_train_batches",
        "number_of_train_batches_run",
    ):
        m = re.search(rf"\[SPEED_BENCH\] {key}=([0-9]+)", text)
        if m:
            profile[key] = int(m.group(1))

    # ------------------------------------------------------------------ #
    # 6. Boolean flags                                                     #
    # ------------------------------------------------------------------ #
    if re.search(r"\[SPEED_BENCH\] bs_mismatch=True", text):
        profile["bs_mismatch"] = True
    if re.search(r"\[SPEED_BENCH\] bs_mismatch=False", text):
        profile["bs_mismatch"] = False

    if re.search(r"\[SPEED_BENCH\] tiny_repo=True", text):
        profile["tiny_repo"] = True
    if re.search(r"\[SPEED_BENCH\] tiny_repo=False", text):
        profile["tiny_repo"] = False

    # ------------------------------------------------------------------ #
    # 7. est_epoch_min: recompute from sec/batch * len_train_loader       #
    #    Only if the profile value is missing or computed from tiny repo.  #
    # ------------------------------------------------------------------ #
    spb = profile.get("train_sec_per_batch")
    ltl = profile.get("len_train_loader") or profile.get("batches_per_epoch")
    if spb and spb > 0 and ltl and ltl > 0:
        if not profile.get("estimated_train_min_per_epoch"):
            profile["estimated_train_min_per_epoch"] = spb * ltl / 60.0

    return profile


def run_scenario(config: dict, scenario_name: str, out_dir: Path,
                 require_full_repo: bool = False) -> tuple:
    """Execute one scenario in-process and return (result, tiny_repo_flag)."""
    import math

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    configured_bs = int(data_cfg.get("batch_size", 16))
    max_train_batches = training_cfg.get("max_train_batches")
    profile_batches_requested = int(training_cfg.get("profile_batches", 0))

    # ------------------------------------------------------------------ #
    # 1. Inspect graph repo BEFORE building the loader                    #
    # ------------------------------------------------------------------ #
    repo_info = inspect_graph_repo(config)
    train_samples = repo_info.get("train_samples") or 0
    is_tiny = repo_info["is_tiny_repo"]
    actual_repo_path = repo_info["graph_repo_path"]

    # Distinct failure: repo exists but NO chunk files found anywhere under it
    no_chunks = (repo_info.get("train_samples") is None and is_tiny)

    if require_full_repo:
        if no_chunks:
            raise RuntimeError(
                f"--require_full_repo: No chunk files found at {actual_repo_path}.\n"
                f"The directory exists but contains no train/chunk_*.pt files.\n"
                f"Possible causes:\n"
                f"  1. Wrong path – check the actual directory layout printed above.\n"
                f"  2. Repo not built yet – run:\n"
                f"     python scripts/run_experiment.py --config configs/d5a.yaml "
                f"--environment kaggle --mode build_graph --device cuda:0\n"
                f"  3. Kaggle dataset slug mismatch – verify the dataset is attached correctly."
            )
        if train_samples != FULL_TRAIN_SAMPLES:
            raise RuntimeError(
                f"--require_full_repo: expected {FULL_TRAIN_SAMPLES} train samples, "
                f"got {train_samples} from {actual_repo_path}.\n"
                f"Rebuild the full graph repo and re-run."
            )

    if is_tiny:
        print(
            f"[SPEED_BENCH] FAIL_TINY_REPO: train_samples={train_samples} < "
            f"{TINY_REPO_THRESHOLD}. Results are not representative."
        )
        # Still run training so we capture sec/batch, but mark as tiny.

    # ------------------------------------------------------------------ #
    # 2. Build data loaders                                               #
    # ------------------------------------------------------------------ #
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)

    actual_bs = getattr(train_loader, "batch_size", configured_bs)
    len_loader = len(train_loader)
    len_dataset = len(train_loader.dataset) if hasattr(train_loader, "dataset") else train_samples
    max_tb = int(max_train_batches) if max_train_batches is not None else len_loader

    # Emit actual loader stats (not hardcoded 28709)
    print(f"[SPEED_BENCH] actual_batch_size={actual_bs}")
    print(f"[SPEED_BENCH] len_train_loader={len_loader}")
    print(f"[SPEED_BENCH] len_train_dataset={len_dataset}")
    print(f"[SPEED_BENCH] batches_per_epoch={len_loader}")   # actual, not ceil(28709/bs)
    print(f"[SPEED_BENCH] train_samples={train_samples}")
    print(f"[SPEED_BENCH] max_train_batches={max_tb}")

    # Warn if profile window will be truncated by max_train_batches
    if profile_batches_requested > 0 and max_train_batches is not None:
        effective_profile = min(profile_batches_requested, int(max_train_batches))
        if effective_profile < profile_batches_requested:
            print(
                f"[SPEED_BENCH] profile_batches_warning: "
                f"requested={profile_batches_requested} but max_train_batches={max_tb}, "
                f"will only record={effective_profile} profile batches"
            )

    # ------------------------------------------------------------------ #
    # 3. Train (first_train_batch_x_shape captured inside trainer loop)  #
    # ------------------------------------------------------------------ #
    trainer = create_trainer(config)
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_cfg.get("epochs", 1)),
        monitor=training_cfg.get("monitor", "val_macro_f1"),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 20)),
        max_train_batches=max_train_batches,
        max_val_batches=training_cfg.get("max_val_batches"),
    )

    history = result.get("history", [])
    number_of_train_batches_run = int(history[-1].get("train_batches", 0)) if history else 0
    print(f"[SPEED_BENCH] number_of_train_batches_run={number_of_train_batches_run}")

    return result, is_tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_dir", default="outputs/speed_benchmark")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--require_full_repo", action="store_true",
        help="Abort if train_samples != 28709",
    )
    args = parser.parse_args()

    # Load base config
    config = load_config(args.config, environment=args.environment)

    # Load scenario config
    with open(args.scenario, "r") as f:
        scenario_cfg = yaml.safe_load(f) or {}

    # Merge configs
    config = deep_merge(config, scenario_cfg)

    # Apply CLI overrides
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.graph_repo_path:
        config.setdefault("paths", {})["graph_repo_path"] = args.graph_repo_path
    if args.csv_root:
        config.setdefault("paths", {})["csv_root"] = args.csv_root
    if args.no_wandb:
        config.setdefault("run", {})["use_wandb"] = False

    # Default to 1 epoch for speed benchmarking
    if "epochs" not in scenario_cfg.get("training", {}):
        config.setdefault("training", {})["epochs"] = 1

    config.setdefault("paths", {})["output_root"] = args.output_dir

    scenario_name = Path(args.scenario).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / f"{scenario_name}_metrics.json"
    profile_file = out_dir / f"{scenario_name}_profile.json"
    summary_file = out_dir / f"{scenario_name}_summary.txt"

    # Tee stdout so we can parse profiling output
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = OutputTee(original_stdout, stdout_capture)

    try:
        result, is_tiny = run_scenario(
            config, scenario_name, out_dir,
            require_full_repo=args.require_full_repo,
        )
        metrics = result.get("history", [{}])[-1] if result.get("history") else {}

        output_text = stdout_capture.getvalue()
        profile_data = parse_profile_output(output_text)

        # bs_mismatch is parsed from [SPEED_BENCH] tags emitted by the trainer
        bs_mismatch = bool(profile_data.get("bs_mismatch", False))

        # Configured batch size from scenario yaml (ground truth for table)
        profile_data["configured_batch_size"] = int(
            scenario_cfg.get("data", {}).get("batch_size",
                config.get("data", {}).get("batch_size", 16))
        )
        profile_data["is_tiny_repo"] = is_tiny

        # Re-confirm est_epoch_min using len_train_loader (actual) × sec/batch.
        # Overwrite any value from the trainer which used a potentially tiny loader.
        spb = profile_data.get("train_sec_per_batch")
        ltl = profile_data.get("len_train_loader") or profile_data.get("batches_per_epoch")
        if spb and spb > 0 and ltl and ltl > 0:
            profile_data["estimated_train_min_per_epoch"] = spb * ltl / 60.0
        elif is_tiny:
            profile_data["estimated_train_min_per_epoch"] = None   # not meaningful

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(profile_file, "w") as f:
            json.dump(profile_data, f, indent=2)

        with open(summary_file, "w") as f:
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
            f.write(f"Profile: {json.dumps(profile_data, indent=2)}\n")

        if is_tiny:
            status = "FAIL_TINY_REPO"
        elif bs_mismatch:
            status = "FAIL_BS_MISMATCH"
        else:
            status = "SUCCESS"
        print(f"\n[Scenario {scenario_name} Complete] status={status} Saved to {out_dir}")

    except Exception as e:
        sys.stdout = original_stdout
        print(f"Scenario {scenario_name} failed: {e}")
        traceback.print_exc()
        if "CUDA out of memory" in str(e):
            print("OOM DETECTED!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        sys.exit(1)
    finally:
        sys.stdout = original_stdout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
