"""Compare lightweight D5A run metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_pred_count(path: Path) -> List[int]:
    if not path.exists():
        return []
    preds = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                preds.append(int(row["y_pred"]))
    except Exception:
        return []
    return np.bincount(np.asarray(preds, dtype=np.int64), minlength=7).astype(int).tolist()


def _entropy(counts: List[int]) -> str:
    total = float(sum(counts))
    if total <= 0:
        return "N/A"
    probs = [c / total for c in counts if c > 0]
    value = -sum(p * math.log(p + 1e-12) for p in probs)
    return f"{value:.4f}"


def _best_history(history) -> Dict[str, Any]:
    if not isinstance(history, list) or not history:
        return {}
    return max(history, key=lambda item: float(item.get("val_macro_f1", -1.0)))


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _note(pred_count: List[int]) -> str:
    total = sum(pred_count)
    if total <= 0:
        return "no test predictions"
    share = max(pred_count) / total
    if share >= 0.70:
        return "possible collapse"
    return ""


def compare_runs(runs: List[Path]) -> Path:
    rows = []
    for run in runs:
        history = _load_json(run / "training_history.json")
        best = _best_history(history)
        metrics = _load_json(run / "evaluation" / "metrics.json") or {}
        pred_count = _read_pred_count(run / "evaluation" / "predictions.csv")
        rows.append(
            {
                "run": str(run),
                "best_epoch": best.get("epoch"),
                "best_val_macro_f1": best.get("val_macro_f1"),
                "test_macro_f1": metrics.get("macro_f1"),
                "test_acc": metrics.get("accuracy"),
                "pred_entropy": _entropy(pred_count),
                "note": _note(pred_count),
            }
        )

    lines = [
        "# D5A Run Comparison",
        "",
        "| run | best_epoch | best_val_macro_f1 | test_macro_f1 | test_acc | pred_entropy | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {_fmt(row['best_epoch'])} | {_fmt(row['best_val_macro_f1'])} | "
            f"{_fmt(row['test_macro_f1'])} | {_fmt(row['test_acc'])} | "
            f"{row['pred_entropy']} | {row['note']} |"
        )
    out_path = Path("comparison_d5a_runs.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    out_path = compare_runs([Path(p) for p in args.runs])
    print(f"Comparison report: {out_path}")


if __name__ == "__main__":
    main()
