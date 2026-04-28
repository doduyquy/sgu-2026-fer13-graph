"""Run multiple speed scenarios and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default="outputs/speed_benchmark")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    scenarios = [
        "configs/scenarios/d5a_speed_bs16.yaml",
        "configs/scenarios/d5a_speed_bs32.yaml",
        "configs/scenarios/d5a_speed_bs64.yaml",
        "configs/scenarios/d5a_speed_bs128.yaml",
        "configs/scenarios/d5a_speed_bs64_loader2.yaml",
        "configs/scenarios/d5a_speed_bs64_loader4.yaml",
        "configs/scenarios/d5a_speed_bs64_amp.yaml",
        "configs/scenarios/d5a_speed_bs64_loader2_amp.yaml"
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_dir / "speed_benchmark_results.csv"
    md_file = out_dir / "speed_benchmark_results.md"
    
    results = []
    
    for scenario_path in scenarios:
        if not Path(scenario_path).exists():
            print(f"Skipping missing scenario: {scenario_path}")
            continue
            
        scenario_name = Path(scenario_path).stem
        print(f"\n{'='*60}\nRunning Scenario: {scenario_name}\n{'='*60}")
        
        cmd = [
            sys.executable, "scripts/run_speed_scenario.py",
            "--config", args.config,
            "--scenario", scenario_path,
            "--device", args.device,
            "--output_dir", args.output_dir
        ]
        if args.environment:
            cmd.extend(["--environment", args.environment])
        if args.graph_repo_path:
            cmd.extend(["--graph_repo_path", args.graph_repo_path])
        if args.no_wandb:
            cmd.append("--no_wandb")
            
        oom = False
        status = "SUCCESS"
        
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(process.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stdout)
            print(e.stderr)
            if "CUDA out of memory" in e.stdout or "CUDA out of memory" in e.stderr:
                oom = True
                status = "OOM"
                print(f"Scenario {scenario_name} failed with OOM.")
            else:
                status = "FAIL"
                print(f"Scenario {scenario_name} failed with error.")
                
        # Read config details to populate CSV
        with open(scenario_path, "r") as f:
            scfg = yaml.safe_load(f) or {}
            
        data_cfg = scfg.get("data", {})
        train_cfg = scfg.get("training", {})
        
        batch_size = data_cfg.get("batch_size", 16)
        
        row = {
            "scenario_name": scenario_name,
            "batch_size": batch_size,
            "num_workers": data_cfg.get("num_workers", 0),
            "pin_memory": data_cfg.get("pin_memory", False),
            "persistent_workers": data_cfg.get("persistent_workers", False),
            "prefetch_factor": data_cfg.get("prefetch_factor", "null"),
            "graph_cache_chunks": data_cfg.get("graph_cache_chunks", 1),
            "amp": train_cfg.get("amp", False),
            "status": status,
            "oom": oom,
            "batches_per_epoch": 28709 // batch_size + (1 if 28709 % batch_size != 0 else 0)
        }
        
        # Load profile
        profile_file = out_dir / f"{scenario_name}_profile.json"
        if status == "SUCCESS" and profile_file.exists():
            with open(profile_file, "r") as f:
                prof = json.load(f)
            row.update({
                "sec_per_batch": prof.get("sec_per_batch"),
                "estimated_train_min_per_epoch": prof.get("estimated_train_min_per_epoch"),
                "data_time": prof.get("data_time"),
                "to_device_time": prof.get("to_device_time"),
                "forward_time": prof.get("forward_time"),
                "loss_time": prof.get("loss_time"),
                "backward_time": prof.get("backward_time"),
                "optimizer_time": prof.get("optimizer_time"),
                "cuda_allocated_gb": prof.get("cuda_allocated_gb"),
                "cuda_reserved_gb": prof.get("cuda_reserved_gb"),
                "cuda_max_allocated_gb": prof.get("cuda_max_allocated_gb"),
            })
            
            # Add note
            note = []
            if prof.get("sec_per_batch"):
                total_t = prof["sec_per_batch"]
                if prof.get("data_time", 0) / total_t > 0.3:
                    note.append("DataLoader bottleneck likely.")
                if (prof.get("forward_time", 0) + prof.get("loss_time", 0) + prof.get("backward_time", 0)) / total_t > 0.6:
                    note.append("Model/loss bottleneck likely.")
                if prof.get("to_device_time", 0) / total_t > 0.2:
                    note.append("CPU-GPU transfer bottleneck likely.")
            row["note"] = " ".join(note)
        else:
            row.update({
                "sec_per_batch": None,
                "estimated_train_min_per_epoch": None,
                "data_time": None,
                "to_device_time": None,
                "forward_time": None,
                "loss_time": None,
                "backward_time": None,
                "optimizer_time": None,
                "cuda_allocated_gb": None,
                "cuda_reserved_gb": None,
                "cuda_max_allocated_gb": None,
                "note": ""
            })
            
        results.append(row)
        
    # Write CSV
    headers = [
        "scenario_name", "batch_size", "num_workers", "pin_memory", "persistent_workers",
        "prefetch_factor", "graph_cache_chunks", "amp", "sec_per_batch", "batches_per_epoch",
        "estimated_train_min_per_epoch", "data_time", "to_device_time", "forward_time", "loss_time",
        "backward_time", "optimizer_time", "cuda_allocated_gb", "cuda_reserved_gb",
        "cuda_max_allocated_gb", "oom", "status", "note"
    ]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Write MD
    md_content = f"# Speed Benchmark Results\n\n| Scenario | Status | BS | Workers | AMP | Sec/Batch | Est. Epoch (min) | Max VRAM (GB) | Note |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|\n"
    for r in results:
        est_min = f"{r['estimated_train_min_per_epoch']:.1f}" if r['estimated_train_min_per_epoch'] else "-"
        sec_b = f"{r['sec_per_batch']:.3f}" if r['sec_per_batch'] else "-"
        vram = f"{r['cuda_max_allocated_gb']:.2f}" if r['cuda_max_allocated_gb'] else "-"
        md_content += f"| {r['scenario_name']} | {r['status']} | {r['batch_size']} | {r['num_workers']} | {r['amp']} | {sec_b} | {est_min} | {vram} | {r['note']} |\n"
        
    with open(md_file, "w") as f:
        f.write(md_content)
        
    # Find best scenario
    valid_results = [r for r in results if r["status"] == "SUCCESS" and r["estimated_train_min_per_epoch"] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x["estimated_train_min_per_epoch"])
        
        best_summary = {
            "scenario_name": best["scenario_name"],
            "batch_size": best["batch_size"],
            "num_workers": best["num_workers"],
            "amp": best["amp"],
            "sec_per_batch": best["sec_per_batch"],
            "estimated_train_min_per_epoch": best["estimated_train_min_per_epoch"],
            "cuda_max_allocated_gb": best["cuda_max_allocated_gb"],
            "note": best["note"]
        }
        
        with open(out_dir / "best_speed_scenario.json", "w") as f:
            json.dump(best_summary, f, indent=2)
            
        with open(out_dir / "best_speed_scenario.md", "w") as f:
            f.write("# Best Speed Scenario\n\n")
            f.write(f"- **Scenario**: `{best['scenario_name']}`\n")
            f.write(f"- **Batch Size**: {best['batch_size']}\n")
            f.write(f"- **Num Workers**: {best['num_workers']}\n")
            f.write(f"- **AMP**: {best['amp']}\n")
            f.write(f"- **Sec/Batch**: {best['sec_per_batch']:.3f}s\n")
            f.write(f"- **Est. Epoch Time**: {best['estimated_train_min_per_epoch']:.1f} minutes\n")
            f.write(f"- **Max VRAM**: {best['cuda_max_allocated_gb']:.2f} GB\n")
            f.write(f"- **Note**: {best['note']}\n")
            
        print(f"\nBest Scenario: {best['scenario_name']} ({best['estimated_train_min_per_epoch']:.1f} mins/epoch)")

if __name__ == "__main__":
    main()
