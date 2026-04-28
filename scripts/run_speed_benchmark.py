"""Run multiple speed scenarios and summarize results.

Changes v4:
- Forwards --graph_repo_path and --require_full_repo to each scenario.
- Table now shows: train_samples, len_train_loader alongside the rest.
- est_epoch_min = sec/batch * len_train_loader (actual, not hardcoded 28709).
- batches_per_epoch is taken from profile (= len_train_loader), not hardcoded.
- FAIL_TINY_REPO status: excluded from best scenario selection.
- bs_mismatch check now accounts for tiny datasets (min(bs, train_samples)).
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

FULL_TRAIN_SAMPLES = 28709   # FER-2013 reference constant

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph_repo_path", default=None,
                        help="Path to the full graph repo (e.g. /kaggle/working/artifacts/graph_repo)")
    parser.add_argument("--output_dir", default="outputs/speed_benchmark")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these scenario stems (e.g. d5a_speed_bs64 d5a_speed_bs64_amp)",
    )
    parser.add_argument(
        "--require_full_repo",
        action="store_true",
        help=(
            "Abort if train_samples != 28709. "
            "Use this to ensure benchmark runs on the full FER-2013 graph repo."
        ),
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
        if args.require_full_repo:
            cmd.append("--require_full_repo")

        oom = False
        status = "SUCCESS"
        stdout_text = ""

        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stdout_text = process.stdout
            print(stdout_text)
            if process.stderr:
                print("[stderr]", process.stderr[-2000:])
        except subprocess.CalledProcessError as e:
            stdout_text = e.stdout or ""
            print(stdout_text)
            if e.stderr:
                print("[stderr]", e.stderr[-2000:])
            combined = stdout_text + (e.stderr or "")
            if "CUDA out of memory" in combined:
                oom = True
                status = "OOM"
                print(f"Scenario {scenario_name} failed with OOM.")
            elif "--require_full_repo" in combined or "FAIL_TINY_REPO" in combined:
                status = "FAIL_TINY_REPO"
                print(f"Scenario {scenario_name}: tiny/missing repo.")
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
            # Filled from profile JSON
            "graph_repo_path": None,
            "train_samples": None,
            "val_samples": None,
            "actual_batch_size": configured_bs,
            "first_batch_x_shape": None,
            "first_batch_edge_attr_shape": None,
            "len_train_loader": None,
            "batches_per_epoch": None,    # actual from profile (= len_train_loader)
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
            "is_tiny_repo": False,
            "note": "",
        }

        # Load profile JSON written by run_speed_scenario.py
        profile_file = out_dir / f"{scenario_name}_profile.json"
        if profile_file.exists():
            with open(profile_file, "r") as f:
                prof = json.load(f)

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

            row["first_batch_x_shape"] = str(prof.get("first_batch_x_shape", ""))
            row["first_batch_edge_attr_shape"] = str(prof.get("first_batch_edge_attr_shape", ""))
            row["bs_mismatch"] = bool(prof.get("bs_mismatch", False))
            row["is_tiny_repo"] = bool(prof.get("is_tiny_repo", False))
            if prof.get("graph_repo_path"):
                row["graph_repo_path"] = prof["graph_repo_path"]

            # Determine final status
            if row["is_tiny_repo"] and status not in ("OOM", "FAIL"):
                row["status"] = "FAIL_TINY_REPO"
            elif row["bs_mismatch"] and status not in ("OOM", "FAIL", "FAIL_TINY_REPO"):
                row["status"] = "FAIL_BS_MISMATCH"

            # Recompute est_epoch_min from actual len_train_loader.
            # Ensures we never show a value computed from a tiny/smoke repo.
            spb = row.get("train_sec_per_batch")
            ltl = row.get("len_train_loader") or row.get("batches_per_epoch")
            if spb and spb > 0 and ltl and ltl > 0:
                computed = spb * ltl / 60.0
                if not row.get("estimated_train_min_per_epoch") or row["is_tiny_repo"]:
                    row["estimated_train_min_per_epoch"] = computed
            if row["is_tiny_repo"]:
                row["estimated_train_min_per_epoch"] = None   # invalid for tiny repo

            # Auto-annotation notes
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
            if row["is_tiny_repo"]:
                ts = row.get("train_samples", "?")
                note_parts.append(
                    f"\u26a0 TINY REPO: train_samples={ts}. "
                    "Benchmark is not representative. Rebuild full graph repo."
                )
            rec = row.get("profile_batches_recorded")
            req = row.get("profile_batches_requested")
            if rec is not None and req is not None and rec < req:
                note_parts.append(
                    f"Profile truncated: recorded={rec} < requested={req} "
                    "(max_train_batches too small)."
                )
            row["note"] = " ".join(note_parts)

        results.append(row)

    # ------------------------------------------------------------------ #
    # Write CSV                                                            #
    # ------------------------------------------------------------------ #
    headers = [
        "scenario_name", "configured_batch_size", "actual_batch_size",
        "graph_repo_path", "train_samples", "val_samples",
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
        "bs_mismatch", "is_tiny_repo", "oom", "status", "note",
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
        "| scenario | train_samples | len_train_loader | first_train_batch_x_shape"
        " | sec/batch | batches/epoch | est_epoch_min | max_vram_gb | profile_recorded | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        x_shape = r.get("first_batch_x_shape") or "-"
        spb = r.get("train_sec_per_batch")
        est = r.get("estimated_train_min_per_epoch")
        vram = r.get("cuda_max_allocated_gb")
        recorded = r.get("profile_batches_recorded")
        requested = r.get("profile_batches_requested")
        ts = r.get("train_samples", "-")
        ltl = r.get("len_train_loader", "-")
        bpe = r.get("batches_per_epoch") or ltl   # same value

        spb_str = f"{spb:.3f}s" if spb is not None else "-"
        est_str = f"{est:.2f}" if (est is not None and not r.get("is_tiny_repo")) else ("TINY" if r.get("is_tiny_repo") else "-")
        vram_str = f"{vram:.2f}" if vram is not None else "-"
        if recorded is not None and requested is not None:
            rec_str = f"{recorded}/{requested}"
            if recorded < requested:
                rec_str += " \u26a0"
        elif recorded is not None:
            rec_str = str(recorded)
        else:
            rec_str = "-"

        md_lines.append(
            f"| {r['scenario_name']} | {ts} | {ltl} | {x_shape} "
            f"| {spb_str} | {bpe} | {est_str} "
            f"| {vram_str} | {rec_str} | {r['status']} |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # Note section
    note_rows = [r for r in results if r.get("note")]
    if note_rows:
        md_content += "\n## Notes\n"
        for r in note_rows:
            md_content += f"- **{r['scenario_name']}**: {r['note']}\n"

    # Rebuild hint
    if any(r.get("is_tiny_repo") for r in results):
        md_content += (
            "\n## \u26a0 Action Required: Rebuild Full Graph Repo\n"
            "The benchmark detected a tiny/smoke graph repo. "
            "Run the following to build the full repo on Kaggle:\n"
            "```bash\n"
            "python scripts/run_experiment.py \\\n"
            "  --config configs/d5a.yaml \\\n"
            "  --environment kaggle \\\n"
            "  --mode build_graph \\\n"
            "  --device cuda:0\n"
            "```\n"
            "Then re-run the benchmark with `--graph_repo_path /kaggle/working/artifacts/graph_repo`.\n"
        )

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n{'='*60}")
    print(f"Results written to:\n  CSV: {csv_file}\n  MD:  {md_file}")

    # ------------------------------------------------------------------ #
    # Pretty-print summary table                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{'Speed Benchmark Results':^80}")
    hdr = (
        f"{'Scenario':<30} {'Samples':>7} {'Loader':>6} {'aBS':>4} "
        f"{'Sec/Bat':>8} {'LenLdr':>6} {'EstMin':>7} {'VRAM':>6} "
        f"{'Rec':>5} {'Status':<18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        spb = r.get("train_sec_per_batch")
        est = r.get("estimated_train_min_per_epoch")
        vram = r.get("cuda_max_allocated_gb")
        recorded = r.get("profile_batches_recorded")
        requested = r.get("profile_batches_requested")
        ts = r.get("train_samples") or 0
        ltl = r.get("len_train_loader") or 0
        spb_str = f"{spb:>7.3f}s" if spb else f"{'?':>8}"
        if r.get("is_tiny_repo"):
            est_str = f"{'TINY':>7}"
        elif est is not None and est > 0:
            est_str = f"{est:>6.1f}m"
        else:
            est_str = f"{'?':>7}"
        rec_str = f"{recorded}/{requested}" if (recorded is not None and requested is not None) else "-"
        if recorded is not None and requested is not None and recorded < requested:
            rec_str += "\u26a0"
        print(
            f"{r['scenario_name']:<30} {ts:>7} {ltl:>6} {r['actual_batch_size']:>4}"
            f" {spb_str} {ltl:>6} {est_str} {(vram or 0):>5.2f}G"
            f" {rec_str:>5} {r['status']:<18}"
        )

    # ------------------------------------------------------------------ #
    # Best scenario (only SUCCESS with full repo)                          #
    # ------------------------------------------------------------------ #
    valid = [
        r for r in results
        if r["status"] == "SUCCESS"
        and r.get("estimated_train_min_per_epoch") is not None
        and not r.get("is_tiny_repo", False)
    ]
    if valid:
        best = min(valid, key=lambda x: x["estimated_train_min_per_epoch"])
        best_summary = {k: best[k] for k in (
            "scenario_name", "configured_batch_size", "actual_batch_size",
            "train_samples", "len_train_loader",
            "num_workers", "amp", "train_sec_per_batch",
            "estimated_train_min_per_epoch", "cuda_max_allocated_gb",
            "batches_per_epoch", "profile_batches_recorded", "note",
        ) if k in best}

        with open(out_dir / "best_speed_scenario.json", "w") as f:
            json.dump(best_summary, f, indent=2)

        with open(out_dir / "best_speed_scenario.md", "w") as f:
            f.write("# Best Speed Scenario\n\n")
            f.write(f"- **Scenario**: `{best['scenario_name']}`\n")
            f.write(f"- **Train Samples**: {best.get('train_samples', 'N/A')}\n")
            f.write(f"- **len(train_loader)**: {best.get('len_train_loader', 'N/A')}\n")
            f.write(f"- **Configured Batch Size**: {best['configured_batch_size']}\n")
            f.write(f"- **Actual Batch Size**: {best['actual_batch_size']}\n")
            f.write(f"- **Num Workers**: {best['num_workers']}\n")
            f.write(f"- **AMP**: {best['amp']}\n")
            f.write(f"- **Sec/Batch**: {best.get('train_sec_per_batch', 0):.3f}s\n")
            f.write(f"- **Batches/Epoch (actual)**: {best.get('batches_per_epoch', 'N/A')}\n")
            f.write(f"- **Est. Epoch Time**: {best['estimated_train_min_per_epoch']:.1f} minutes\n")
            f.write(f"- **Max VRAM**: {best.get('cuda_max_allocated_gb', 0):.2f} GB\n")
            f.write(f"- **Profile Recorded**: {best.get('profile_batches_recorded', 'N/A')}\n")
            f.write(f"- **Note**: {best.get('note', '')}\n")

        print(f"\nBest: {best['scenario_name']} → {best['estimated_train_min_per_epoch']:.1f} mins/epoch")
    else:
        print("\nNo valid SUCCESS results with full repo to select best scenario.")
        if any(r.get("is_tiny_repo") for r in results):
            print(
                "  All runs used a tiny graph repo. "
                "Please rebuild the full graph repo and rerun with --require_full_repo."
            )


if __name__ == "__main__":
    main()
