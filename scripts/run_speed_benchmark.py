"""Run multiple speed scenarios and summarize results.

Changes v2:
- Reads train_sec_per_batch, estimated_train_min_per_epoch, cuda_max_allocated_gb,
  profile_batches_requested, profile_batches_recorded, actual_batch_size,
  first_batch shapes, len_train_loader, batches_per_epoch from profile JSON.
- Table now shows Sec/Batch and Est. Epoch correctly.
- bs_mismatch flag is surfaced in Status column.
- Only the 3 required scenarios are mandatory; others are optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

TRAIN_SAMPLES = 28709  # FER-2013 training set size

# Scenarios to run (in order)
DEFAULT_SCENARIOS = [
    "configs/scenarios/d5a_speed_bs16.yaml",
    "configs/scenarios/d5a_speed_bs32.yaml",
    "configs/scenarios/d5a_speed_bs64.yaml",
    "configs/scenarios/d5a_speed_bs128.yaml",
    "configs/scenarios/d5a_speed_bs64_loader2.yaml",
    "configs/scenarios/d5a_speed_bs64_loader4.yaml",
    "configs/scenarios/d5a_speed_bs64_amp.yaml",
    "configs/scenarios/d5a_speed_bs64_loader2_amp.yaml",
]

# Scenarios that MUST pass (will be re-run if specified via --mandatory)
MANDATORY_SCENARIOS = {
    "d5a_speed_bs64",
    "d5a_speed_bs128",
    "d5a_speed_bs64_amp",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default="outputs/speed_benchmark")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these scenario stems (e.g. d5a_speed_bs64 d5a_speed_bs128)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / "speed_benchmark_results.csv"
    md_file = out_dir / "speed_benchmark_results.md"

    results = []

    for scenario_path_str in DEFAULT_SCENARIOS:
        scenario_path = Path(scenario_path_str)
        scenario_name = scenario_path.stem

        # Filter if --only specified
        if args.only and scenario_name not in args.only:
            continue

        if not scenario_path.exists():
            print(f"Skipping missing scenario: {scenario_path}")
            continue

        print(f"\n{'='*60}\nRunning Scenario: {scenario_name}\n{'='*60}")

        cmd = [
            sys.executable, "scripts/run_speed_scenario.py",
            "--config", args.config,
            "--scenario", str(scenario_path),
            "--device", args.device,
            "--output_dir", args.output_dir,
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
            if process.stderr:
                print("[stderr]", process.stderr[-2000:])
        except subprocess.CalledProcessError as e:
            print(e.stdout)
            if e.stderr:
                print("[stderr]", e.stderr[-2000:])
            combined = (e.stdout or "") + (e.stderr or "")
            if "CUDA out of memory" in combined:
                oom = True
                status = "OOM"
                print(f"Scenario {scenario_name} failed with OOM.")
            else:
                status = "FAIL"
                print(f"Scenario {scenario_name} failed with error.")

        # Read scenario config for metadata
        with open(scenario_path, "r") as f:
            scfg = yaml.safe_load(f) or {}

        data_cfg = scfg.get("data", {})
        train_cfg = scfg.get("training", {})

        configured_bs = int(data_cfg.get("batch_size", 16))
        num_workers = int(data_cfg.get("num_workers", 0))
        amp = bool(train_cfg.get("amp", False))
        profile_batches_requested_cfg = int(train_cfg.get("profile_batches", 0))
        max_train_batches_cfg = train_cfg.get("max_train_batches")

        row = {
            "scenario_name": scenario_name,
            "configured_batch_size": configured_bs,
            "num_workers": num_workers,
            "pin_memory": data_cfg.get("pin_memory", False),
            "persistent_workers": data_cfg.get("persistent_workers", False),
            "prefetch_factor": data_cfg.get("prefetch_factor", "null"),
            "graph_cache_chunks": data_cfg.get("graph_cache_chunks", 1),
            "amp": amp,
            "status": status,
            "oom": oom,
            # Will be filled from profile JSON
            "actual_batch_size": configured_bs,
            "first_batch_x_shape": None,
            "first_batch_edge_attr_shape": None,
            "len_train_loader": None,
            "batches_per_epoch": math.ceil(TRAIN_SAMPLES / configured_bs),
            "max_train_batches": max_train_batches_cfg,
            "number_of_train_batches_run": None,
            "train_sec_per_batch": None,
            "estimated_train_min_per_epoch": None,
            "avg_data_time": None,
            "avg_to_device_time": None,
            "avg_forward_time": None,
            "avg_loss_time": None,
            "avg_backward_time": None,
            "avg_optimizer_time": None,
            "cuda_allocated_gb": None,
            "cuda_reserved_gb": None,
            "cuda_max_allocated_gb": None,
            "profile_batches_requested": profile_batches_requested_cfg,
            "profile_batches_recorded": None,
            "bs_mismatch": False,
            "note": "",
        }

        # Load profile JSON written by run_speed_scenario.py
        profile_file = out_dir / f"{scenario_name}_profile.json"
        if status in ("SUCCESS", "FAIL_BS_MISMATCH") and profile_file.exists():
            with open(profile_file, "r") as f:
                prof = json.load(f)

            # Scalar fields from profile
            for key in (
                "actual_batch_size",
                "len_train_loader",
                "batches_per_epoch",
                "max_train_batches",
                "number_of_train_batches_run",
                "train_sec_per_batch",
                "estimated_train_min_per_epoch",
                "avg_data_time",
                "avg_to_device_time",
                "avg_forward_time",
                "avg_loss_time",
                "avg_backward_time",
                "avg_optimizer_time",
                "cuda_allocated_gb",
                "cuda_reserved_gb",
                "cuda_max_allocated_gb",
                "profile_batches_requested",
                "profile_batches_recorded",
            ):
                if prof.get(key) is not None:
                    row[key] = prof[key]

            # Shape fields (stored as lists)
            row["first_batch_x_shape"] = str(prof.get("first_batch_x_shape", ""))
            row["first_batch_edge_attr_shape"] = str(prof.get("first_batch_edge_attr_shape", ""))
            row["bs_mismatch"] = bool(prof.get("bs_mismatch", False))

            if row["bs_mismatch"]:
                row["status"] = "FAIL_BS_MISMATCH"

            # Recompute estimated epoch from actual values if still missing or zero
            spb_val = row.get("train_sec_per_batch")
            if spb_val is not None and spb_val > 0:
                bpe = row.get("batches_per_epoch")
                if bpe and bpe > 0:
                    computed_est = spb_val * bpe / 60.0
                    # Override if missing OR if previously computed value is 0.0
                    if not row.get("estimated_train_min_per_epoch"):
                        row["estimated_train_min_per_epoch"] = computed_est

            # Auto-annotation
            note_parts = []
            spb = row.get("train_sec_per_batch")
            if spb:
                if row.get("avg_data_time") and row["avg_data_time"] / spb > 0.3:
                    note_parts.append("DataLoader bottleneck.")
                if (
                    (row.get("avg_forward_time", 0) or 0)
                    + (row.get("avg_loss_time", 0) or 0)
                    + (row.get("avg_backward_time", 0) or 0)
                ) / spb > 0.6:
                    note_parts.append("Model/loss bottleneck.")
                if row.get("avg_to_device_time") and row["avg_to_device_time"] / spb > 0.2:
                    note_parts.append("CPU-GPU transfer bottleneck.")
            if row["bs_mismatch"]:
                note_parts.append("\u26a0 BS MISMATCH – DataLoader not using configured bs!")
            # Profile truncation warning
            rec = row.get("profile_batches_recorded")
            req = row.get("profile_batches_requested")
            if rec is not None and req is not None and rec < req:
                note_parts.append(f"Profile truncated: recorded={rec} < requested={req} (max_train_batches too small).")
            row["note"] = " ".join(note_parts)

        results.append(row)

    # ------------------------------------------------------------------ #
    # Write CSV                                                            #
    # ------------------------------------------------------------------ #
    headers = [
        "scenario_name", "configured_batch_size", "actual_batch_size",
        "first_batch_x_shape", "first_batch_edge_attr_shape",
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "graph_cache_chunks", "amp",
        "len_train_loader", "batches_per_epoch", "max_train_batches",
        "number_of_train_batches_run",
        "train_sec_per_batch", "estimated_train_min_per_epoch",
        "avg_data_time", "avg_to_device_time", "avg_forward_time",
        "avg_loss_time", "avg_backward_time", "avg_optimizer_time",
        "cuda_allocated_gb", "cuda_reserved_gb", "cuda_max_allocated_gb",
        "profile_batches_requested", "profile_batches_recorded",
        "bs_mismatch", "oom", "status", "note",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ------------------------------------------------------------------ #
    # Write Markdown                                                       #
    # ------------------------------------------------------------------ #
    md_lines = [
        "# Speed Benchmark Results\n",
        "| scenario | actual_bs | x_shape | sec/batch | batches/epoch |"
        " est_epoch_min | max_vram_gb | profile_batches_recorded | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        x_shape = r.get("first_batch_x_shape") or "-"
        spb = r.get("train_sec_per_batch")
        est = r.get("estimated_train_min_per_epoch")
        vram = r.get("cuda_max_allocated_gb")
        recorded = r.get("profile_batches_recorded")
        requested = r.get("profile_batches_requested")

        spb_str = f"{spb:.3f}s" if spb is not None else "-"
        # Never show 0.0 – recompute from spb if needed
        if est is None or est == 0.0:
            bpe = r.get("batches_per_epoch")
            if spb is not None and spb > 0 and bpe:
                est = spb * bpe / 60.0
        est_str = f"{est:.2f}" if (est is not None and est > 0) else "-"
        vram_str = f"{vram:.2f}" if vram is not None else "-"
        if recorded is not None and requested is not None:
            rec_str = f"{recorded}/{requested}"
            if recorded < requested:
                rec_str += " ⚠"
        elif recorded is not None:
            rec_str = str(recorded)
        else:
            rec_str = "-"

        md_lines.append(
            f"| {r['scenario_name']} | {r['actual_batch_size']} | {x_shape} "
            f"| {spb_str} | {r['batches_per_epoch']} | {est_str} "
            f"| {vram_str} | {rec_str} | {r['status']} |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # Note section
    note_rows = [r for r in results if r.get("note")]
    if note_rows:
        md_content += "\n## Notes\n"
        for r in note_rows:
            md_content += f"- **{r['scenario_name']}**: {r['note']}\n"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n{'='*60}")
    print(f"Results written to:\n  CSV: {csv_file}\n  MD:  {md_file}")

    # ------------------------------------------------------------------ #
    # Pretty-print summary table                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{'Speed Benchmark Results':^60}")
    header = (
        f"{'Scenario':<35} {'aBS':>5} {'Sec/Bat':>8} {'Bat/Ep':>7}"
        f" {'EstMin':>7} {'VRAM':>6} {'Rec':>5} {'Status':<20}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        spb = r.get("train_sec_per_batch")
        est = r.get("estimated_train_min_per_epoch")
        vram = r.get("cuda_max_allocated_gb")
        recorded = r.get("profile_batches_recorded")
        requested = r.get("profile_batches_requested")
        # Recompute est if missing or zero
        if (est is None or est == 0.0) and spb and spb > 0:
            bpe = r.get("batches_per_epoch")
            if bpe:
                est = spb * bpe / 60.0
        est_str = f"{est:>6.1f}m" if (est is not None and est > 0) else f"{'?':>7}"
        spb_str = f"{spb:>7.3f}s" if spb else f"{'?':>8}"
        rec_str = f"{recorded}/{requested}" if recorded is not None and requested is not None else "-"
        if recorded is not None and requested is not None and recorded < requested:
            rec_str += "⚠"
        print(
            f"{r['scenario_name']:<35} {r['actual_batch_size']:>5}"
            f" {spb_str} {r['batches_per_epoch']:>7}"
            f" {est_str} {(vram or 0):>5.2f}G"
            f" {rec_str:>6} {r['status']:<20}"
        )

    # ------------------------------------------------------------------ #
    # Best scenario                                                        #
    # ------------------------------------------------------------------ #
    valid = [
        r for r in results
        if r["status"] == "SUCCESS" and r.get("estimated_train_min_per_epoch") is not None
    ]
    if valid:
        best = min(valid, key=lambda x: x["estimated_train_min_per_epoch"])
        best_summary = {k: best[k] for k in (
            "scenario_name", "configured_batch_size", "actual_batch_size",
            "num_workers", "amp", "train_sec_per_batch",
            "estimated_train_min_per_epoch", "cuda_max_allocated_gb",
            "batches_per_epoch", "profile_batches_recorded", "note",
        ) if k in best}

        with open(out_dir / "best_speed_scenario.json", "w") as f:
            json.dump(best_summary, f, indent=2)

        with open(out_dir / "best_speed_scenario.md", "w") as f:
            f.write("# Best Speed Scenario\n\n")
            f.write(f"- **Scenario**: `{best['scenario_name']}`\n")
            f.write(f"- **Configured Batch Size**: {best['configured_batch_size']}\n")
            f.write(f"- **Actual Batch Size**: {best['actual_batch_size']}\n")
            f.write(f"- **Num Workers**: {best['num_workers']}\n")
            f.write(f"- **AMP**: {best['amp']}\n")
            f.write(f"- **Sec/Batch**: {best.get('train_sec_per_batch', 0):.3f}s\n")
            f.write(f"- **Batches/Epoch**: {best.get('batches_per_epoch', 'N/A')}\n")
            f.write(f"- **Est. Epoch Time**: {best['estimated_train_min_per_epoch']:.1f} minutes\n")
            f.write(f"- **Max VRAM**: {best.get('cuda_max_allocated_gb', 0):.2f} GB\n")
            f.write(f"- **Profile Recorded**: {best.get('profile_batches_recorded', 'N/A')}\n")
            f.write(f"- **Note**: {best.get('note', '')}\n")

        print(f"\nBest: {best['scenario_name']} → {best['estimated_train_min_per_epoch']:.1f} mins/epoch")


if __name__ == "__main__":
    main()
