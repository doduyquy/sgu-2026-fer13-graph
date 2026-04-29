"""Create a lightweight diagnostic report for a D5A run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTION_NAMES)


def _load_json(path: Path, warnings: List[str]):
    if not path.exists():
        warnings.append(f"WARN missing file: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warnings.append(f"WARN failed to read {path}: {exc}")
        return None


def _read_predictions(path: Path, warnings: List[str]) -> tuple[List[int], List[int]]:
    if not path.exists():
        warnings.append(f"WARN missing file: {path}")
        return [], []
    y_true: List[int] = []
    y_pred: List[int] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(int(row["y_true"]))
                y_pred.append(int(row["y_pred"]))
    except Exception as exc:
        warnings.append(f"WARN failed to read {path}: {exc}")
    return y_true, y_pred


def _counts(values: List[int]) -> List[int]:
    if not values:
        return [0] * NUM_CLASSES
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=NUM_CLASSES).astype(int).tolist()


def _best_history(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not history:
        return {}
    return max(history, key=lambda item: float(item.get("val_macro_f1", -1.0)))


def _latest_history(history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    return history[-1] if history else {}


def _collapse_note(pred_count: List[int]) -> str:
    total = sum(pred_count)
    if total <= 0:
        return "No predictions available."
    shares = sorted(((count / total, idx) for idx, count in enumerate(pred_count)), reverse=True)
    if shares[0][0] >= 0.70:
        return f"WARNING possible class collapse: {EMOTION_NAMES[shares[0][1]]} has {shares[0][0]:.1%} of predictions."
    if shares[0][0] + shares[1][0] >= 0.85:
        return (
            "WARNING possible two-class collapse: "
            f"{EMOTION_NAMES[shares[0][1]]}+{EMOTION_NAMES[shares[1][1]]} have "
            f"{shares[0][0] + shares[1][0]:.1%} of predictions."
        )
    return "No severe prediction collapse detected."


def _classification_rows(report: Optional[Dict[str, Any]]) -> List[tuple[str, float, float, float, int]]:
    rows = []
    if not isinstance(report, dict):
        return rows
    for idx, name in enumerate(EMOTION_NAMES):
        item = report.get(name) or report.get(str(idx))
        if isinstance(item, dict):
            rows.append(
                (
                    name,
                    float(item.get("precision", 0.0)),
                    float(item.get("recall", 0.0)),
                    float(item.get("f1-score", 0.0)),
                    int(item.get("support", 0)),
                )
            )
    return rows


def _confusion_from_predictions(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[t, p] += 1
    return cm


def _confusion_notes(cm: np.ndarray) -> List[str]:
    notes = []
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    flat = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm2[i, j] > 0:
                flat.append((int(cm2[i, j]), i, j))
    for count, true_idx, pred_idx in sorted(flat, reverse=True)[:5]:
        notes.append(f"- {EMOTION_NAMES[true_idx]} -> {EMOTION_NAMES[pred_idx]}: {count}")
    return notes or ["- No confusion pairs available."]


def _figure_counts(root: Path) -> Dict[str, int]:
    figures = root / "figures"
    if not figures.exists():
        return {"class_gates": 0, "attention_maps": 0, "total_png": 0}
    pngs = list(figures.rglob("*.png"))
    class_gates = [p for p in pngs if "gate" in p.name.lower() or "class_gate" in str(p).lower()]
    attention = [p for p in pngs if "attention" in str(p).lower() or "attn" in str(p).lower()]
    return {"class_gates": len(class_gates), "attention_maps": len(attention), "total_png": len(pngs)}


def _entropy(counts: List[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return float("nan")
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def build_report(output_root: Path, eval_root: Optional[Path]) -> Path:
    warnings: List[str] = []
    eval_root = eval_root or output_root / "evaluation"
    history = _load_json(output_root / "training_history.json", warnings)
    metrics = _load_json(eval_root / "metrics.json", warnings)
    standalone_report = _load_json(eval_root / "classification_report.json", warnings)
    y_true, y_pred = _read_predictions(eval_root / "predictions.csv", warnings)

    best = _best_history(history if isinstance(history, list) else None)
    latest = _latest_history(history if isinstance(history, list) else None)
    report = metrics.get("classification_report") if isinstance(metrics, dict) else standalone_report
    rows = _classification_rows(report)
    pred_count = _counts(y_pred)
    true_count = _counts(y_true)
    cm = _confusion_from_predictions(y_true, y_pred)
    fig_counts = _figure_counts(output_root)

    best_epoch = best.get("epoch", "N/A")
    best_val_macro_f1 = best.get("val_macro_f1", "N/A")
    test_acc = metrics.get("accuracy", "N/A") if isinstance(metrics, dict) else "N/A"
    test_macro = metrics.get("macro_f1", "N/A") if isinstance(metrics, dict) else "N/A"
    test_weighted = metrics.get("weighted_f1", "N/A") if isinstance(metrics, dict) else "N/A"

    best_class = max(rows, key=lambda r: r[3], default=None)
    worst_class = min(rows, key=lambda r: r[3], default=None)
    disgust = next((r for r in rows if r[0] == "Disgust"), None)
    fear = next((r for r in rows if r[0] == "Fear"), None)
    happy_pred_share = pred_count[3] / max(sum(pred_count), 1)

    lines = [
        "# D5A Diagnostic Report",
        "",
        "## Summary",
        f"- output_root: `{output_root}`",
        f"- eval_root: `{eval_root}`",
        f"- best_epoch: {best_epoch}",
        f"- best_val_macro_f1: {best_val_macro_f1}",
        f"- final_epoch: {latest.get('epoch', 'N/A')}",
        f"- test_accuracy: {test_acc}",
        f"- test_macro_f1: {test_macro}",
        f"- test_weighted_f1: {test_weighted}",
        "",
        "## Prediction Distribution",
        f"- true_count: {dict(zip(EMOTION_NAMES, true_count))}",
        f"- pred_count: {dict(zip(EMOTION_NAMES, pred_count))}",
        f"- pred_entropy: {_entropy(pred_count):.4f}",
        f"- {_collapse_note(pred_count)}",
        "",
        "## Per-Class Quality",
    ]
    if rows:
        lines += ["| class | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|"]
        for name, precision, recall, f1, support in rows:
            lines.append(f"| {name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |")
        lines += [
            "",
            f"- highest_f1: {best_class[0]} ({best_class[3]:.4f})" if best_class else "- highest_f1: N/A",
            f"- lowest_f1: {worst_class[0]} ({worst_class[3]:.4f})" if worst_class else "- lowest_f1: N/A",
            f"- Disgust status: f1={disgust[3]:.4f}, recall={disgust[2]:.4f}" if disgust else "- Disgust status: N/A",
            f"- Fear status: f1={fear[3]:.4f}, recall={fear[2]:.4f}" if fear else "- Fear status: N/A",
            f"- Happy prediction share: {happy_pred_share:.1%}",
        ]
    else:
        lines.append("- WARN no classification report available.")

    lines += [
        "",
        "## Confusion Notes",
        *_confusion_notes(cm),
        "",
        "## Attention/Gate Quick Check",
        f"- class_gate_png_count: {fig_counts['class_gates']}",
        f"- attention_png_count: {fig_counts['attention_maps']}",
        f"- total_figure_png_count: {fig_counts['total_png']}",
        f"- figures_dir: `{output_root / 'figures'}`",
        "",
        "## Conclusion",
    ]
    if isinstance(best_val_macro_f1, (int, float)) and float(best_val_macro_f1) > 0.25:
        lines.append("- D5A is learning a non-trivial signal, but quality is still weak/moderate.")
    else:
        lines.append("- D5A may be under-learning or collapsed; inspect per-class metrics first.")
    lines += [
        "- Main suspected issues to test next:",
        "  - A. class collapse if pred_count is concentrated in 1-2 classes",
        "  - B. edge motif noise if node_score_only improves macro F1",
        "  - C. aux loss too strong if ce_only improves macro F1",
        "  - D. prototype score too weak if all ablations stay flat",
    ]
    if warnings:
        lines += ["", "## Warnings", *[f"- {w}" for w in warnings]]

    out_dir = output_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d5a_diagnostic_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--eval_root", default=None)
    args = parser.parse_args()
    out_path = build_report(Path(args.output_root), Path(args.eval_root) if args.eval_root else None)
    print(f"Diagnostic report: {out_path}")


if __name__ == "__main__":
    main()
