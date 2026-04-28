"""Run a single speed scenario and log results."""

from __future__ import annotations

import argparse
import io
import json
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

from common import apply_cli_overrides, load_config
from train_d5a import run_train
import torch

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

def parse_profile_output(text: str):
    profile = {}
    
    # Parse last batch profile for memory stats
    mem_matches = list(re.finditer(r"cuda_allocated_gb\s*=\s*([0-9.]+).*?cuda_reserved_gb\s*=\s*([0-9.]+).*?cuda_max_allocated_gb\s*=\s*([0-9.]+)", text, re.DOTALL))
    if mem_matches:
        last_mem = mem_matches[-1]
        profile["cuda_allocated_gb"] = float(last_mem.group(1))
        profile["cuda_reserved_gb"] = float(last_mem.group(2))
        profile["cuda_max_allocated_gb"] = float(last_mem.group(3))

    # Parse average profile
    avg_match = re.search(r"\[PROFILE average first \d+ batches\](.*?)estimated_full_epoch_minutes=(.*?)\n", text, re.DOTALL)
    if avg_match:
        avg_block = avg_match.group(1)
        est_min = avg_match.group(2).strip()
        
        metrics_map = {
            "data_time": r"avg_data_time\s*=\s*([0-9.]+)s",
            "to_device_time": r"avg_to_device_time\s*=\s*([0-9.]+)s",
            "forward_time": r"avg_forward_time\s*=\s*([0-9.]+)s",
            "loss_time": r"avg_loss_time\s*=\s*([0-9.]+)s",
            "backward_time": r"avg_backward_time\s*=\s*([0-9.]+)s",
            "optimizer_time": r"avg_optimizer_time\s*=\s*([0-9.]+)s",
            "sec_per_batch": r"avg_batch_time\s*=\s*([0-9.]+)s",
        }
        
        for k, pattern in metrics_map.items():
            m = re.search(pattern, avg_block)
            if m:
                profile[k] = float(m.group(1))
                
        if est_min != "unknown":
            profile["estimated_train_min_per_epoch"] = float(est_min)
            
    return profile

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
        
    # Set default epochs to 1 if not specified in scenario
    if "epochs" not in scenario_cfg.get("training", {}):
        config.setdefault("training", {})["epochs"] = 1
        
    config.setdefault("paths", {})["output_root"] = args.output_dir

    scenario_name = Path(args.scenario).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = out_dir / f"{scenario_name}_metrics.json"
    profile_file = out_dir / f"{scenario_name}_profile.json"
    summary_file = out_dir / f"{scenario_name}_summary.txt"

    # Set up stdout tee to capture profiling
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = OutputTee(original_stdout, stdout_capture)
    
    try:
        # Run training
        result = run_train(config)
        metrics = result.get("history", [{}])[-1] if result.get("history") else {}
        
        # Parse captured output
        output_text = stdout_capture.getvalue()
        profile_data = parse_profile_output(output_text)
        
        # Save results
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        with open(profile_file, "w") as f:
            json.dump(profile_data, f, indent=2)
            
        with open(summary_file, "w") as f:
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
            f.write(f"Profile: {json.dumps(profile_data, indent=2)}\n")
            
        print(f"\n[Scenario {scenario_name} Complete] Saved to {out_dir}")
        
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
