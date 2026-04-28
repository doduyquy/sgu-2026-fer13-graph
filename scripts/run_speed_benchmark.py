"""Run multiple speed scenarios and summarize results.

Changes v4:
- Forwards --graph_repo_path and --require_full_repo to each scenario.
- Table now shows: train_samples, len_train_loader alongside the rest.
- est_epoch_min = sec/batch * len_train_loader (actual, not hardcoded 28709).
- batches_per_epoch is taken from profile (= len_train_loader), not hardcoded.
- FAIL_TINY_REPO status: excluded from best scenario selection.
- bs_mismatch check now accounts for tiny datasets (min(bs, train_samples)).

Changes v5:
- Report compute-only and wall-clock timing separately.
- Select best scenario by wall_epoch_min, not compute-only avg_batch_time.
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
FULL_VAL_SAMPLES = 3589
FULL_TEST_SAMPLES = 3589
MAX_T4_VRAM_GB = 14.0

# Scenarios to run (in order)
DEFAULT_SCENARIOS = [
    "configs/scenarios/d5a_speed_bs16.yaml",
    "configs/scenarios/d5a_speed_bs32.yaml",
    "configs/scenarios/d5a_speed_bs32_amp.yaml",
    "configs/scenarios/d5a_speed_bs64.yaml",
    "configs/scenarios/d5a_speed_bs128.yaml",
    "configs/scenarios/d5a_speed_bs64_loader2.yaml",
    "configs/scenarios/d5a_speed_bs64_loader4.yaml",
    "configs/scenarios/d5a_speed_bs64_amp.yaml",
    "configs/scenarios/d5a_speed_bs64_loader2_amp.yaml",
]


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _as_float(value):
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_finite_positive(value) -> bool:
    parsed = _as_float(value)
    return parsed is not None and parsed > 0


def _graph_repo_location(path_value) -> str:
    if not path_value:
        return "-"
    normalized = str(path_value).replace("\\", "/")
    if "/kaggle/input" in normalized:
        return "/kaggle/input"
    if "/kaggle/working" in normalized:
        return "/kaggle/working"
    return normalized


def _derive_timing_fields(row: dict) -> None:
    data_spb = _as_float(row.get("data_sec_per_batch"))
    if data_spb is None:
        data_spb = _as_float(row.get("avg_data_time"))

    compute_spb = _as_float(row.get("compute_sec_per_batch"))
    if compute_spb is None:
        compute_spb = _as_float(row.get("avg_batch_time"))
    if compute_spb is None:
        pieces = [
            _as_float(row.get("avg_to_device_time")),
            _as_float(row.get("avg_forward_time")),
            _as_float(row.get("avg_loss_time")),
            _as_float(row.get("avg_backward_time")),
            _as_float(row.get("avg_optimizer_time")),
        ]
        if all(v is not None for v in pieces):
            compute_spb = sum(pieces)

    wall_spb = _as_float(row.get("wall_sec_per_batch"))
    if wall_spb is None:
        wall_spb = _as_float(row.get("train_sec_per_batch"))
    if wall_spb is None and data_spb is not None and compute_spb is not None:
        wall_spb = data_spb + compute_spb
    elif wall_spb is None and compute_spb is not None:
        wall_spb = compute_spb

    ltl = row.get("len_train_loader") or row.get("batches_per_epoch")
    try:
        ltl = int(ltl) if ltl is not None else None
    except (TypeError, ValueError):
        ltl = None

    row["data_sec_per_batch"] = data_spb
    row["compute_sec_per_batch"] = compute_spb
    row["wall_sec_per_batch"] = wall_spb
    row["graph_repo_location"] = _graph_repo_location(row.get("graph_repo_path"))

    if compute_spb is not None and ltl and ltl > 0:
        row["compute_epoch_min"] = compute_spb * ltl / 60.0
    if wall_spb is not None and ltl and ltl > 0:
        row["wall_epoch_min"] = wall_spb * ltl / 60.0
        row["estimated_train_min_per_epoch"] = row["wall_epoch_min"]
        row["train_sec_per_batch"] = wall_spb
    if data_spb is not None and wall_spb and wall_spb > 0:
        row["data_percent"] = data_spb / wall_spb * 100.0
    if compute_spb is not None and wall_spb and wall_spb > 0:
        row["compute_percent"] = compute_spb / wall_spb * 100.0


def _format_num(value, digits: int = 3, suffix: str = "") -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}{suffix}"


def _format_percent(value) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.0f}%"


def _is_valid_best_row(row: dict) -> bool:
    vram = _as_float(row.get("cuda_max_allocated_gb"))
    return (
        row.get("status") == "SUCCESS"
        and _is_finite_positive(row.get("wall_epoch_min"))
        and vram is not None
        and vram < MAX_T4_VRAM_GB
        and row.get("train_samples") == FULL_TRAIN_SAMPLES
        and row.get("val_samples") == FULL_VAL_SAMPLES
        and row.get("test_samples") == FULL_TEST_SAMPLES
        and not row.get("bs_mismatch", False)
        and not row.get("is_tiny_repo", False)
    )


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

    from run_speed_scenario import parse_profile_output

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (out_dir / "best_speed_scenario.json", out_dir / "best_speed_scenario.md"):
        if stale.exists():
            stale.unlink()

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

        for stale in (
            out_dir / f"{scenario_name}_profile.json",
            out_dir / f"{scenario_name}_metrics.json",
            out_dir / f"{scenario_name}_summary.txt",
        ):
            if stale.exists():
                stale.unlink()

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
            "graph_repo_location": None,
            "train_samples": None,
            "val_samples": None,
            "test_samples": None,
            "actual_batch_size": configured_bs,
            "first_batch_x_shape": None,
            "first_batch_edge_attr_shape": None,
            "len_train_loader": None,
            "batches_per_epoch": None,    # actual from profile (= len_train_loader)
            "max_train_batches": max_train_batches_cfg,
            "number_of_train_batches_run": None,
            "train_sec_per_batch": None,
            "estimated_train_min_per_epoch": None,
            "avg_batch_time": None,
            "avg_data_time": None,
            "data_sec_per_batch": None,
            "compute_sec_per_batch": None,
            "wall_sec_per_batch": None,
            "compute_epoch_min": None,
            "wall_epoch_min": None,
            "data_percent": None,
            "compute_percent": None,
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

        # Load profile JSON written by run_speed_scenario.py.
        # If the child failed before writing JSON (for example a full-repo
        # guard failure), still parse the captured stdout for useful metadata.
        profile_file = out_dir / f"{scenario_name}_profile.json"
        prof = parse_profile_output(stdout_text) if stdout_text else {}
        if profile_file.exists():
            with open(profile_file, "r") as f:
                prof_from_file = json.load(f)
            prof.update(prof_from_file)

        if prof:
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
                "avg_batch_time",
                "avg_data_time",
                "data_sec_per_batch",
                "compute_sec_per_batch",
                "wall_sec_per_batch",
                "compute_epoch_min",
                "wall_epoch_min",
                "data_percent",
                "compute_percent",
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
            row["is_tiny_repo"] = bool(prof.get("is_tiny_repo", prof.get("tiny_repo", False)))
            if prof.get("graph_repo_path"):
                row["graph_repo_path"] = prof["graph_repo_path"]

            _derive_timing_fields(row)

            # Determine final status
            if row["is_tiny_repo"] and status not in ("OOM", "FAIL"):
                row["status"] = "FAIL_TINY_REPO"
            elif row["bs_mismatch"] and status not in ("OOM", "FAIL", "FAIL_TINY_REPO"):
                row["status"] = "FAIL_BS_MISMATCH"
            elif args.require_full_repo and any(
                row.get(key) is not None and row.get(key) != expected
                for key, expected in (
                    ("train_samples", FULL_TRAIN_SAMPLES),
                    ("val_samples", FULL_VAL_SAMPLES),
                    ("test_samples", FULL_TEST_SAMPLES),
                )
            ):
                row["status"] = "FAIL_FULL_REPO_MISMATCH"

            if row["is_tiny_repo"]:
                row["estimated_train_min_per_epoch"] = None   # invalid for tiny repo
                row["wall_epoch_min"] = None

            # Auto-annotation notes
            note_parts = []
            wall_spb = _as_float(row.get("wall_sec_per_batch"))
            if row.get("graph_repo_path") and "/kaggle/input" in str(row["graph_repo_path"]).replace("\\", "/"):
                note_parts.append(
                    "WARNING: graph_repo is on Kaggle input mount. I/O may dominate. "
                    "Prefer /kaggle/working/artifacts/graph_repo."
                )
            if wall_spb:
                data_pct = _as_float(row.get("data_percent"))
                if data_pct is not None and data_pct > 50.0:
                    note_parts.append("I/O bottleneck: data loading dominates wall time.")
                elif data_pct is not None and data_pct > 30.0:
                    note_parts.append("DataLoader bottleneck.")
                model_time = (
                    (_as_float(row.get("avg_forward_time")) or 0)
                    + (_as_float(row.get("avg_loss_time")) or 0)
                    + (_as_float(row.get("avg_backward_time")) or 0)
                )
                if model_time / wall_spb > 0.6:
                    note_parts.append("Model/loss bottleneck.")
                to_device = _as_float(row.get("avg_to_device_time")) or 0
                if to_device / wall_spb > 0.2:
                    note_parts.append("CPU-GPU transfer bottleneck.")
            data_spb = _as_float(row.get("data_sec_per_batch"))
            if data_spb is not None and data_spb > 2.0:
                note_parts.append("I/O still dominates. Need dataset caching/preload optimization.")
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

        _derive_timing_fields(row)
        results.append(row)

    # ------------------------------------------------------------------ #
    # Write CSV                                                            #
    # ------------------------------------------------------------------ #
    headers = [
        "scenario_name", "configured_batch_size", "actual_batch_size",
        "graph_repo_path", "graph_repo_location", "train_samples", "val_samples",
        "test_samples",
        "first_batch_x_shape", "first_batch_edge_attr_shape",
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "graph_cache_chunks", "amp",
        "len_train_loader", "batches_per_epoch", "max_train_batches",
        "number_of_train_batches_run",
        "data_sec_per_batch", "compute_sec_per_batch", "wall_sec_per_batch",
        "compute_epoch_min", "wall_epoch_min", "data_percent", "compute_percent",
        "train_sec_per_batch", "estimated_train_min_per_epoch",
        "avg_batch_time", "avg_data_time", "avg_to_device_time", "avg_forward_time",
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
        "| scenario | graph_repo_location | train_samples | batch_size | amp | workers | data_sec/batch | compute_sec/batch | wall_sec/batch | data_% | wall_epoch_min | max_vram_gb | status |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        ts = r.get("train_samples") if r.get("train_samples") is not None else "-"
        wall_epoch = r.get("wall_epoch_min")
        wall_epoch_str = (
            "TINY" if r.get("is_tiny_repo")
            else _format_num(wall_epoch, digits=2)
        )
        md_lines.append(
            f"| {r['scenario_name']} | {r.get('graph_repo_location') or '-'} | {ts} "
            f"| {r['configured_batch_size']} | {str(r['amp']).lower()} | {r['num_workers']} "
            f"| {_format_num(r.get('data_sec_per_batch'), digits=3)} "
            f"| {_format_num(r.get('compute_sec_per_batch'), digits=3)} "
            f"| {_format_num(r.get('wall_sec_per_batch'), digits=3)} "
            f"| {_format_percent(r.get('data_percent'))} "
            f"| {wall_epoch_str} "
            f"| {_format_num(r.get('cuda_max_allocated_gb'), digits=2)} "
            f"| {r['status']} |"
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
        f"{'Scenario':<26} {'Repo':>15} {'Samples':>7} {'BS':>4} "
        f"{'Data':>8} {'Compute':>8} {'Wall':>8} {'Data%':>6} "
        f"{'Epoch':>8} {'VRAM':>6} {'Status':<18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ts = r.get("train_samples") or 0
        wall_epoch = _as_float(r.get("wall_epoch_min"))
        if r.get("is_tiny_repo"):
            epoch_str = f"{'TINY':>8}"
        elif wall_epoch is not None and wall_epoch > 0:
            epoch_str = f"{wall_epoch:>7.1f}m"
        else:
            epoch_str = f"{'?':>8}"
        repo_str = str(r.get("graph_repo_location") or "-")[-15:]
        print(
            f"{r['scenario_name']:<26} {repo_str:>15} {ts:>7} {r['configured_batch_size']:>4}"
            f" {_format_num(r.get('data_sec_per_batch'), digits=3):>8}"
            f" {_format_num(r.get('compute_sec_per_batch'), digits=3):>8}"
            f" {_format_num(r.get('wall_sec_per_batch'), digits=3):>8}"
            f" {_format_percent(r.get('data_percent')):>6}"
            f" {epoch_str} {_format_num(r.get('cuda_max_allocated_gb'), digits=2):>6}"
            f" {r['status']:<18}"
        )

    # ------------------------------------------------------------------ #
    # Best scenario (SUCCESS, full repo, finite wall clock, T4-safe VRAM)  #
    # ------------------------------------------------------------------ #
    valid = [r for r in results if _is_valid_best_row(r)]
    if valid:
        best = min(valid, key=lambda x: x["wall_epoch_min"])
        best_summary = {k: best[k] for k in (
            "scenario_name", "configured_batch_size", "actual_batch_size",
            "graph_repo_path", "graph_repo_location",
            "train_samples", "val_samples", "test_samples", "len_train_loader",
            "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
            "graph_cache_chunks", "amp",
            "data_sec_per_batch", "compute_sec_per_batch", "wall_sec_per_batch",
            "compute_epoch_min", "wall_epoch_min", "data_percent", "compute_percent",
            "cuda_max_allocated_gb",
            "batches_per_epoch", "profile_batches_recorded", "note",
        ) if k in best}

        with open(out_dir / "best_speed_scenario.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        reason = (
            "Lowest wall_epoch_min among SUCCESS scenarios on the full repo, "
            f"with max_vram_gb < {MAX_T4_VRAM_GB:.0f}."
        )
        with open(out_dir / "best_speed_scenario.md", "w", encoding="utf-8") as f:
            f.write("# Best Speed Scenario\n\n")
            f.write("Best config:\n")
            f.write(f"- graph_repo_path: `{best.get('graph_repo_path', 'N/A')}`\n")
            f.write(f"- batch_size: {best['configured_batch_size']}\n")
            f.write(f"- amp: {str(best['amp']).lower()}\n")
            f.write(f"- num_workers: {best['num_workers']}\n")
            f.write(f"- pin_memory: {str(best.get('pin_memory')).lower()}\n")
            f.write(f"- persistent_workers: {str(best.get('persistent_workers')).lower()}\n")
            f.write(f"- prefetch_factor: {best.get('prefetch_factor')}\n")
            f.write(f"- graph_cache_chunks: {best.get('graph_cache_chunks')}\n")
            f.write(f"- wall_sec_per_batch: {best.get('wall_sec_per_batch', 0):.3f}\n")
            f.write(f"- wall_epoch_min: {best.get('wall_epoch_min', 0):.2f}\n")
            f.write(f"- max_vram_gb: {best.get('cuda_max_allocated_gb', 0):.2f}\n")
            f.write(f"- reason: {reason}\n")
            if best.get("note"):
                f.write(f"- note: {best.get('note')}\n")

        print(f"\nBest: {best['scenario_name']} -> {best['wall_epoch_min']:.1f} mins/epoch wall-clock")
    else:
        print("\nNo valid SUCCESS results with full repo to select best scenario.")
        if any(r.get("is_tiny_repo") for r in results):
            print(
                "  All runs used a tiny graph repo. "
                "Please rebuild the full graph repo and rerun with --require_full_repo."
            )


if __name__ == "__main__":
    main()
