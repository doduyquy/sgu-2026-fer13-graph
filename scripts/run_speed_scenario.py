"""Run a single speed scenario and log results.

Changes v2:
- Parser now extracts sec_per_batch from epoch log (train_sec/batch=...) as fallback.
- Logs actual_batch_size, first_batch shapes, len(train_loader), batches_per_epoch,
  max_train_batches, number_of_train_batches_run into profile JSON.
- bs128 validation: asserts x.shape[0] == 128 and edge_attr.shape[0] == 128.
- Profile average block now reports profile_batches_requested / profile_batches_recorded.
- Parser extracts those counts and all avg_ metrics even without the avg block
  by falling back to the epoch train_sec_per_batch.
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

TRAIN_SAMPLES = 28709  # FER-2013 training set size


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


def parse_profile_output(text: str) -> dict:
    """Parse captured stdout into a profile dict.

    Priority order for sec_per_batch / estimated_train_min_per_epoch:
    1. [PROFILE average first N batches] block (most accurate, uses real avg).
    2. epoch line ``train_sec/batch=Xs`` (single-epoch average).

    Also extracts:
    - cuda_max_allocated_gb from last [PROFILE batch=N] block.
    - profile_batches_requested / profile_batches_recorded.
    - first_batch_x_shape, first_batch_edge_attr_shape.
    - actual_batch_size, len_train_loader, batches_per_epoch, max_train_batches,
      number_of_train_batches_run (injected by run_speed_scenario main()).
    """
    profile: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Per-batch PROFILE blocks – take the last one for peak VRAM.      #
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
            # groups: 1=n, 2=block, 3=est_min
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
    # 4. If estimated_epoch still missing, compute from sec_per_batch.    #
    # ------------------------------------------------------------------ #
    if "estimated_train_min_per_epoch" not in profile and "train_sec_per_batch" in profile:
        # Use the full epoch batch count if available, else use TRAIN_SAMPLES
        bpe = profile.get("batches_per_epoch")
        if bpe is not None:
            profile["estimated_train_min_per_epoch"] = (
                profile["train_sec_per_batch"] * bpe / 60.0
            )

    # ------------------------------------------------------------------ #
    # 5. first_batch shapes                                               #
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
    # 6. Scenario-level diagnostic fields                                 #
    # ------------------------------------------------------------------ #
    for key in (
        "actual_batch_size",
        "len_train_loader",
        "batches_per_epoch",
        "max_train_batches",
        "number_of_train_batches_run",
    ):
        m = re.search(rf"\[SPEED_BENCH\] {key}=([0-9]+)", text)
        if m:
            profile[key] = int(m.group(1))

    # bs_mismatch flag
    if re.search(r"\[SPEED_BENCH\] bs_mismatch=True", text):
        profile["bs_mismatch"] = True
    if re.search(r"\[SPEED_BENCH\] bs_mismatch=False", text):
        profile["bs_mismatch"] = False

    return profile


def run_scenario(config: dict, scenario_name: str, out_dir: Path) -> dict:
    """Execute one scenario in-process and return the profile dict."""
    import math

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    configured_bs = int(data_cfg.get("batch_size", 16))
    max_train_batches = training_cfg.get("max_train_batches")

    # Build data loaders
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)

    # ---------- diagnostic prints (captured by OutputTee) ---------- #
    actual_bs = getattr(train_loader, "batch_size", configured_bs)
    len_loader = len(train_loader)
    batches_per_epoch_full = math.ceil(TRAIN_SAMPLES / actual_bs)
    max_tb = int(max_train_batches) if max_train_batches is not None else len_loader

    print(f"[SPEED_BENCH] actual_batch_size={actual_bs}")
    print(f"[SPEED_BENCH] len_train_loader={len_loader}")
    print(f"[SPEED_BENCH] batches_per_epoch={batches_per_epoch_full}")
    print(f"[SPEED_BENCH] max_train_batches={max_tb}")

    # ---------- first-batch inspection ---------- #
    first_batch = next(iter(train_loader))
    x_shape = list(first_batch["x"].shape)
    ea_shape = list(first_batch["edge_attr"].shape)
    n_samples_in_batch = x_shape[0]

    print(f"[SPEED_BENCH] first_batch_x_shape={x_shape}")
    print(f"[SPEED_BENCH] first_batch_edge_attr_shape={ea_shape}")

    # ---------- bs128 validation ---------- #
    bs_mismatch = False
    if configured_bs == 128:
        if n_samples_in_batch != 128 or ea_shape[0] != 128:
            print(
                f"[SPEED_BENCH] bs_mismatch=True  "
                f"(expected 128, got x.shape[0]={n_samples_in_batch}, "
                f"edge_attr.shape[0]={ea_shape[0]})"
            )
            bs_mismatch = True
        else:
            print(f"[SPEED_BENCH] bs_mismatch=False  (x.shape[0]={n_samples_in_batch} ✓)")

    # ---------- train ---------- #
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

    # Count actual batches run (from history)
    history = result.get("history", [])
    if history:
        number_of_train_batches_run = int(history[-1].get("train_batches", 0))
    else:
        number_of_train_batches_run = 0
    print(f"[SPEED_BENCH] number_of_train_batches_run={number_of_train_batches_run}")

    return result, bs_mismatch


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
        result, bs_mismatch = run_scenario(config, scenario_name, out_dir)
        metrics = result.get("history", [{}])[-1] if result.get("history") else {}

        output_text = stdout_capture.getvalue()
        profile_data = parse_profile_output(output_text)

        # Inject bs_mismatch flag
        profile_data["bs_mismatch"] = bs_mismatch

        # Configured batch size from scenario yaml (ground truth for table)
        profile_data["configured_batch_size"] = int(
            scenario_cfg.get("data", {}).get("batch_size", config.get("data", {}).get("batch_size", 16))
        )

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(profile_file, "w") as f:
            json.dump(profile_data, f, indent=2)

        with open(summary_file, "w") as f:
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
            f.write(f"Profile: {json.dumps(profile_data, indent=2)}\n")

        status = "FAIL_BS_MISMATCH" if bs_mismatch else "SUCCESS"
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
