"""Prepare /kaggle/working graph_repo and run wall-clock IO benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_working_graph_repo import inspect_working_graph_repo, prepare_graph_repo


SCENARIOS = [
    "d5a_speed_bs32",
    "d5a_speed_bs32_amp",
    "d5a_speed_bs64",
    "d5a_speed_bs64_amp",
]
MAX_T4_VRAM_GB = 14.0


def _as_float(value):
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _valid_best(row: Dict[str, Any]) -> bool:
    wall_spb = _as_float(row.get("wall_sec_per_batch"))
    wall_epoch = _as_float(row.get("wall_epoch_min"))
    vram = _as_float(row.get("cuda_max_allocated_gb"))
    return (
        row.get("status") == "SUCCESS"
        and wall_spb is not None
        and wall_spb > 0
        and wall_epoch is not None
        and wall_epoch > 0
        and vram is not None
        and vram < MAX_T4_VRAM_GB
        and _as_int(row.get("train_samples")) == 28709
        and _as_int(row.get("val_samples")) == 3589
        and _as_int(row.get("test_samples")) == 3589
        and str(row.get("bs_mismatch", "")).lower() not in ("true", "1")
        and str(row.get("is_tiny_repo", "")).lower() not in ("true", "1")
    )


def _select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [row for row in rows if _valid_best(row)]
    if not valid:
        return None
    return min(valid, key=lambda row: _as_float(row.get("wall_epoch_min")) or float("inf"))


def _train_command(best: Dict[str, Any], graph_repo_path: str, no_wandb: bool) -> str:
    cmd = [
        "python scripts/run_experiment.py",
        "  --config configs/d5a.yaml",
        "  --environment kaggle",
        "  --mode train",
        "  --epochs 10",
        "  --device cuda:0",
        f"  --graph_repo_path {graph_repo_path}",
        f"  --batch_size {_as_int(best.get('configured_batch_size'))}",
        f"  --num_workers {_as_int(best.get('num_workers'))}",
        f"  --pin_memory {str(_as_bool(best.get('pin_memory'))).lower()}",
        f"  --persistent_workers {str(_as_bool(best.get('persistent_workers'))).lower()}",
        f"  --graph_cache_chunks {_as_int(best.get('graph_cache_chunks'), 1)}",
    ]
    prefetch = str(best.get("prefetch_factor", "")).strip()
    if prefetch and prefetch.lower() not in ("none", "null"):
        cmd.append(f"  --prefetch_factor {_as_int(prefetch)}")
    if _as_bool(best.get("amp")):
        cmd.append("  --amp")
    if no_wandb:
        cmd.append("  --no_wandb")
    return " \\\n".join(cmd)


def _write_best_files(
    best: Dict[str, Any] | None,
    out_dir: Path,
    graph_repo_path: str,
    no_wandb: bool,
) -> None:
    json_path = out_dir / "best_io_scenario.json"
    md_path = out_dir / "best_io_scenario.md"

    if best is None:
        payload = {"status": "NO_VALID_SCENARIO"}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(
            "# Best IO Scenario\n\n"
            "No valid SUCCESS scenario was found.\n\n"
            "Next optimization:\n"
            "- preload/cache graph repo\n"
            "- reduce edge motif\n"
            "- D5A-node-only\n"
            "- DDP after I/O is stable\n",
            encoding="utf-8",
        )
        return

    reason = (
        "Lowest wall_epoch_min among SUCCESS scenarios on the full graph repo, "
        f"with max_vram_gb < {MAX_T4_VRAM_GB:.0f}."
    )
    wall_epoch = _as_float(best.get("wall_epoch_min")) or 0.0
    payload = {
        "scenario": best.get("scenario_name"),
        "graph_repo_path": graph_repo_path,
        "batch_size": _as_int(best.get("configured_batch_size")),
        "amp": _as_bool(best.get("amp")),
        "num_workers": _as_int(best.get("num_workers")),
        "pin_memory": _as_bool(best.get("pin_memory")),
        "persistent_workers": _as_bool(best.get("persistent_workers")),
        "prefetch_factor": best.get("prefetch_factor"),
        "graph_cache_chunks": _as_int(best.get("graph_cache_chunks"), 1),
        "wall_sec_per_batch": _as_float(best.get("wall_sec_per_batch")),
        "wall_epoch_min": wall_epoch,
        "max_vram_gb": _as_float(best.get("cuda_max_allocated_gb")),
        "reason": reason,
        "note": best.get("note", ""),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Best IO Scenario",
        "",
        "Best config:",
        f"- graph_repo_path: `{payload['graph_repo_path']}`",
        f"- batch_size: {payload['batch_size']}",
        f"- amp: {str(payload['amp']).lower()}",
        f"- num_workers: {payload['num_workers']}",
        f"- pin_memory: {str(payload['pin_memory']).lower()}",
        f"- persistent_workers: {str(payload['persistent_workers']).lower()}",
        f"- prefetch_factor: {payload['prefetch_factor']}",
        f"- graph_cache_chunks: {payload['graph_cache_chunks']}",
        f"- wall_sec_per_batch: {payload['wall_sec_per_batch']:.3f}",
        f"- wall_epoch_min: {payload['wall_epoch_min']:.2f}",
        f"- max_vram_gb: {payload['max_vram_gb']:.2f}",
        f"- reason: {payload['reason']}",
    ]
    if payload["note"]:
        lines.append(f"- note: {payload['note']}")

    if wall_epoch <= 15.0:
        lines.extend([
            "",
            "Recommended train command:",
            "```bash",
            _train_command(best, graph_repo_path, no_wandb),
            "```",
        ])
    else:
        lines.extend([
            "",
            "wall_epoch_min > 15, so do not start a long train yet.",
            "",
            "Next optimization:",
            "- preload/cache graph repo",
            "- reduce edge motif",
            "- D5A-node-only",
            "- DDP after I/O is stable",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default="kaggle")
    parser.add_argument("--csv_root", default=None)
    parser.add_argument(
        "--input_graph_repo",
        default="/kaggle/input/datasets/irthn1311/graph-repo/graph_repo",
    )
    parser.add_argument("--working_graph_repo", default="/kaggle/working/artifacts/graph_repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare_method", choices=["build", "copy", "auto"], default="auto")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--force_prepare", "--force", dest="force_prepare", action="store_true")
    parser.add_argument("--output_dir", default="/kaggle/working/fer_d5_outputs/io_benchmark")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    working_repo = str(Path(args.working_graph_repo))

    if args.skip_prepare:
        prepare_result = inspect_working_graph_repo(working_repo, require_full=True)
        prepare_result["method"] = "skip_prepare"
        prepare_result["prepare_time_sec"] = 0.0
    else:
        prepare_result = prepare_graph_repo(
            method=args.prepare_method,
            csv_root=args.csv_root,
            input_graph_repo=args.input_graph_repo,
            working_graph_repo=working_repo,
            config_path=args.config,
            environment=args.environment,
            force=args.force_prepare,
        )

    (out_dir / "graph_repo_prepare_result.json").write_text(
        json.dumps(prepare_result, indent=2),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_speed_benchmark.py"),
        "--config", args.config,
        "--environment", args.environment,
        "--device", args.device,
        "--graph_repo_path", working_repo,
        "--output_dir", str(out_dir),
        "--only", *SCENARIOS,
        "--require_full_repo",
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")

    print("Running IO speed benchmark:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    speed_csv = out_dir / "speed_benchmark_results.csv"
    speed_md = out_dir / "speed_benchmark_results.md"
    io_csv = out_dir / "io_benchmark_results.csv"
    io_md = out_dir / "io_benchmark_results.md"
    if speed_csv.exists():
        shutil.copyfile(speed_csv, io_csv)
    if speed_md.exists():
        shutil.copyfile(speed_md, io_md)

    rows = _read_rows(io_csv)
    best = _select_best(rows)
    _write_best_files(best, out_dir, working_repo, no_wandb=args.no_wandb)

    print(f"IO benchmark outputs written to: {out_dir}")
    if best is not None:
        print(
            "Best IO scenario: "
            f"{best.get('scenario_name')} wall_epoch_min={_as_float(best.get('wall_epoch_min')):.2f}"
        )
    else:
        print("No valid IO scenario selected.")


if __name__ == "__main__":
    main()
