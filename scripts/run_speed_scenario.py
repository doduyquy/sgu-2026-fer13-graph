"""Run a single speed scenario and log results.

Changes v3:
- fix: parse [SPEED_BENCH] fields BEFORE computing estimated_train_min_per_epoch
  so batches_per_epoch is available for the fallback formula.
- first_train_batch_x_shape is now logged INSIDE the trainer (from actual first
  training batch), not from a separate next(iter(...)) call before fit().
- bs_mismatch validation extended to ALL batch sizes (not only bs128):
  expected shape[0] == configured_bs; if last batch is smaller it is noted but
  not flagged as mismatch unless it is the FIRST batch.
- profile_batches_recorded = number of [PROFILE batch=N] blocks found by parser.
- est_epoch_min guaranteed non-zero when sec/batch is available.
"""

from __future__ import annotations

import argparse
import io
import json
import math
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

from common import build_dataloader, create_trainer, load_config
import torch

TRAIN_SAMPLES = 28709  # FER-2013 training set size


# ─────────────────────────────────────────────────────────────────────────────
# OutputTee
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_profile_output(text: str) -> dict:
    """Parse captured stdout into a profile dict.

    IMPORTANT: parse [SPEED_BENCH] scalar fields FIRST so that
    batches_per_epoch is available when computing estimated_train_min_per_epoch.
    """
    profile: dict = {}

    # ------------------------------------------------------------------ #
    # 0. [SPEED_BENCH] scalar fields  ← must be FIRST                     #
    # ------------------------------------------------------------------ #
    for key in (
        "actual_batch_size",
        "configured_batch_size",
        "len_train_loader",
        "batches_per_epoch",
        "max_train_batches",
        "number_of_train_batches_run",
    ):
        m = re.search(rf"\[SPEED_BENCH\] {key}=([0-9]+)", text)
        if m:
            profile[key] = int(m.group(1))

    # first_train_batch_x_shape (logged from inside train loop)
    x_shape_m = re.search(
        r"\[SPEED_BENCH\] first_train_batch_x_shape=(\[[^\]]+\])", text
    )
    if x_shape_m:
        try:
            profile["first_train_batch_x_shape"] = json.loads(x_shape_m.group(1))
        except Exception:
            profile["first_train_batch_x_shape"] = x_shape_m.group(1)

    ea_shape_m = re.search(
        r"\[SPEED_BENCH\] first_train_batch_edge_attr_shape=(\[[^\]]+\])", text
    )
    if ea_shape_m:
        try:
            profile["first_train_batch_edge_attr_shape"] = json.loads(ea_shape_m.group(1))
        except Exception:
            profile["first_train_batch_edge_attr_shape"] = ea_shape_m.group(1)

    # bs_mismatch
    if re.search(r"\[SPEED_BENCH\] bs_mismatch=True", text):
        profile["bs_mismatch"] = True
    elif re.search(r"\[SPEED_BENCH\] bs_mismatch=False", text):
        profile["bs_mismatch"] = False

    # ------------------------------------------------------------------ #
    # 1. Per-batch PROFILE blocks – count for profile_batches_recorded,   #
    #    take the last for peak VRAM.                                      #
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
    # 2. [PROFILE average first N batches (recorded=M)] block             #
    # ------------------------------------------------------------------ #
    avg_match = re.search(
        r"\[PROFILE average first (\d+) batches \(recorded=(\d+)\)\](.*?)"
        r"estimated_full_epoch_minutes=([0-9.]+|unknown)",
        text, re.DOTALL,
    )
    # Fallback: old format without recorded=
    if avg_match is None:
        avg_match = re.search(
            r"\[PROFILE average first (\d+) batches\](.*?)"
            r"estimated_full_epoch_minutes=([0-9.]+|unknown)",
            text, re.DOTALL,
        )
        if avg_match:
            profile["profile_batches_requested"] = int(avg_match.group(1))
            # recorded not in old format – use count from per-batch blocks
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
            "avg_data_time":       r"avg_data_time\s*=\s*([0-9.]+)s",
            "avg_to_device_time":  r"avg_to_device_time\s*=\s*([0-9.]+)s",
            "avg_forward_time":    r"avg_forward_time\s*=\s*([0-9.]+)s",
            "avg_loss_time":       r"avg_loss_time\s*=\s*([0-9.]+)s",
            "avg_backward_time":   r"avg_backward_time\s*=\s*([0-9.]+)s",
            "avg_optimizer_time":  r"avg_optimizer_time\s*=\s*([0-9.]+)s",
            "train_sec_per_batch": r"avg_batch_time\s*=\s*([0-9.]+)s",
        }
        for k, pattern in metrics_map.items():
            m = re.search(pattern, avg_block)
            if m:
                profile[k] = float(m.group(1))

        if est_min_str and est_min_str != "unknown":
            profile["estimated_train_min_per_epoch"] = float(est_min_str)

    # ------------------------------------------------------------------ #
    # 3. Fallback sec/batch: epoch line  train_sec/batch=Xs               #
    # ------------------------------------------------------------------ #
    if "train_sec_per_batch" not in profile:
        epoch_match = re.search(r"train_sec/batch=([0-9.]+)s", text)
        if epoch_match:
            profile["train_sec_per_batch"] = float(epoch_match.group(1))

    # ------------------------------------------------------------------ #
    # 4. Compute est_epoch_min if still missing (guaranteed non-zero)     #
    #    Formula: sec_per_batch * batches_per_epoch / 60                  #
    #    batches_per_epoch already parsed in step 0.                      #
    # ------------------------------------------------------------------ #
    spb = profile.get("train_sec_per_batch")
    if spb is not None and spb > 0:
        if not profile.get("estimated_train_min_per_epoch"):
            bpe = profile.get("batches_per_epoch")
            if bpe and bpe > 0:
                profile["estimated_train_min_per_epoch"] = spb * bpe / 60.0

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(config: dict) -> tuple:
    """Build loaders, run training, return (result, bs_mismatch)."""
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    configured_bs = int(data_cfg.get("batch_size", 16))
    max_train_batches = training_cfg.get("max_train_batches")

    # Build data loaders
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader   = build_dataloader(config, split="val",   shuffle=False)

    # ---- Diagnostic prints (captured by OutputTee) ----
    actual_bs          = getattr(train_loader, "batch_size", configured_bs)
    len_loader         = len(train_loader)
    batches_per_epoch  = math.ceil(TRAIN_SAMPLES / actual_bs)
    max_tb             = int(max_train_batches) if max_train_batches is not None else len_loader

    print(f"[SPEED_BENCH] configured_batch_size={configured_bs}")
    print(f"[SPEED_BENCH] actual_batch_size={actual_bs}")
    print(f"[SPEED_BENCH] len_train_loader={len_loader}")
    print(f"[SPEED_BENCH] batches_per_epoch={batches_per_epoch}")
    print(f"[SPEED_BENCH] max_train_batches={max_tb}")

    # ---- Train ----
    # first_train_batch shape is logged INSIDE trainer.train_one_epoch
    # so it comes from the real first batch of the training loop.
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

    # ---- bs_mismatch: read from stdout (logged by trainer) ----
    # bs_mismatch flag is parsed from [SPEED_BENCH] bs_mismatch= line
    # that the trainer prints inside train_one_epoch.
    bs_mismatch = False  # will be overridden by parser after capture
    return result, bs_mismatch


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--scenario",    required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root",    default=None)
    parser.add_argument("--output_dir",  default="outputs/speed_benchmark")
    parser.add_argument("--no_wandb",    action="store_true")
    args = parser.parse_args()

    # Load & merge configs
    config = load_config(args.config, environment=args.environment)
    with open(args.scenario, "r") as f:
        scenario_cfg = yaml.safe_load(f) or {}
    config = deep_merge(config, scenario_cfg)

    # CLI overrides
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.graph_repo_path:
        config.setdefault("paths", {})["graph_repo_path"] = args.graph_repo_path
    if args.csv_root:
        config.setdefault("paths", {})["csv_root"] = args.csv_root
    if args.no_wandb:
        config.setdefault("run", {})["use_wandb"] = False
    if "epochs" not in scenario_cfg.get("training", {}):
        config.setdefault("training", {})["epochs"] = 1

    config.setdefault("paths", {})["output_root"] = args.output_dir

    scenario_name = Path(args.scenario).stem
    out_dir       = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / f"{scenario_name}_metrics.json"
    profile_file = out_dir / f"{scenario_name}_profile.json"
    summary_file = out_dir / f"{scenario_name}_summary.txt"

    # Tee stdout
    stdout_capture  = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = OutputTee(original_stdout, stdout_capture)

    try:
        result, _ = run_scenario(config)
        metrics    = result.get("history", [{}])[-1] if result.get("history") else {}

        output_text  = stdout_capture.getvalue()
        profile_data = parse_profile_output(output_text)

        # bs_mismatch comes from the parsed flag (set by trainer)
        bs_mismatch = bool(profile_data.get("bs_mismatch", False))

        profile_data["configured_batch_size"] = int(
            scenario_cfg.get("data", {}).get(
                "batch_size", config.get("data", {}).get("batch_size", 16)
            )
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
        print(f"\n[Scenario {scenario_name} Complete] status={status}  saved={out_dir}")

    except Exception as e:
        sys.stdout = original_stdout
        print(f"Scenario {scenario_name} FAILED: {e}")
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
