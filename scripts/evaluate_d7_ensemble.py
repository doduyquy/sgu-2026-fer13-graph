"""Official D7 / D7+D8B Graph-Swin ensemble evaluation.

The script can evaluate an ensemble either from checkpoints on a shared
dataloader or from saved predictions.csv files containing per-class logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTION_NAMES)


def confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[dict[str, float]], np.ndarray]:
    cm = confusion_matrix_array(y_true, y_pred)
    pred_count = cm.sum(axis=0)
    support = cm.sum(axis=1)
    rows: list[dict[str, float]] = []
    for class_idx in range(NUM_CLASSES):
        tp = float(cm[class_idx, class_idx])
        precision = tp / float(pred_count[class_idx]) if pred_count[class_idx] else 0.0
        recall = tp / float(support[class_idx]) if support[class_idx] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": float(support[class_idx]),
            }
        )
    return rows, support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rows, support = _per_class_report(y_true, y_pred)
    f1 = np.asarray([row["f1-score"] for row in rows], dtype=np.float64)
    total = float(len(y_true))
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if total else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / total) if total else 0.0,
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rows, support = _per_class_report(y_true, y_pred)
    total = float(len(y_true))
    report: dict[str, Any] = {
        class_name: rows[class_idx] for class_idx, class_name in enumerate(EMOTION_NAMES)
    }
    precision = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    f1 = np.asarray([row["f1-score"] for row in rows], dtype=np.float64)
    report["accuracy"] = float(np.mean(y_true == y_pred)) if total else 0.0
    report["macro avg"] = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1-score": float(f1.mean()),
        "support": total,
    }
    report["weighted avg"] = {
        "precision": float(np.sum(precision * support) / total) if total else 0.0,
        "recall": float(np.sum(recall * support) / total) if total else 0.0,
        "f1-score": float(np.sum(f1 * support) / total) if total else 0.0,
        "support": total,
    }
    return report


OLD_D7_OUTPUT_DIR = "output/d7_ensemble_seed44_long150_window4"
NEW_D7_D8B_OUTPUT_DIR = "output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg"

OLD_D7_MEMBERS = [
    (
        "seed44",
        "configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml",
        "output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv",
    ),
    (
        "long150_resume",
        "configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml",
        "output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv",
    ),
    (
        "window4_region_transformer",
        "configs/experiments/d7a_graph_swin_region_transformer_window4.yaml",
        "output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv",
    ),
]

NEW_D7_D8B_MEMBERS = [
    (
        "d7_seed44",
        "configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml",
        "output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv",
    ),
    (
        "d7_long150_resume",
        "configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml",
        "output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv",
    ),
    (
        "d7_window4_region_transformer",
        "configs/experiments/d7a_graph_swin_region_transformer_window4.yaml",
        "output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth",
        "output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv",
    ),
    (
        "d8b_border020",
        "configs/experiments/d8b_face_aware_graph_swin_border020.yaml",
        "output/d8b_face_aware_graph_swin_border020/checkpoints/best.pth",
        "output/d8b_face_aware_graph_swin_border020/evaluation/predictions.csv",
    ),
    (
        "d8b_area045",
        "configs/experiments/d8b_face_aware_graph_swin_area045.yaml",
        "output/d8b_face_aware_graph_swin_area045/checkpoints/best.pth",
        "output/d8b_face_aware_graph_swin_area045/evaluation/predictions.csv",
    ),
]

ENSEMBLE_PRESETS = {
    "d7_seed44_long150_window4_logit": {
        "members": OLD_D7_MEMBERS,
        "method": "logit_average",
        "output_dir": OLD_D7_OUTPUT_DIR,
    },
    "d7_d8b_border020_area045_probavg": {
        "members": NEW_D7_D8B_MEMBERS,
        "method": "probability_average",
        "output_dir": NEW_D7_D8B_OUTPUT_DIR,
    },
}

CONFUSION_FOCUS = [
    ("fear_to_sad", 2, 4),
    ("fear_to_neutral", 2, 6),
    ("fear_to_surprise", 2, 5),
    ("sad_to_fear", 4, 2),
    ("sad_to_neutral", 4, 6),
    ("neutral_to_sad", 6, 4),
    ("neutral_to_fear", 6, 2),
    ("disgust_to_angry", 1, 0),
    ("disgust_to_sad", 1, 4),
    ("disgust_to_happy", 1, 3),
    ("happy_to_sad", 3, 4),
    ("happy_to_neutral", 3, 6),
]


@dataclass
class MemberSpec:
    name: str
    config: str | None = None
    checkpoint: str | None = None
    predictions: str | None = None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_member(value: str) -> MemberSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Members must use name:config:checkpoint format; got " + value
        )
    return MemberSpec(name=parts[0], config=parts[1], checkpoint=parts[2])


def parse_prediction_file(value: str) -> MemberSpec:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Prediction files must use name:path format; got " + value
        )
    return MemberSpec(name=parts[0], predictions=parts[1])


def default_members(preset: str = "d7_d8b_border020_area045_probavg") -> list[MemberSpec]:
    default_rows = ENSEMBLE_PRESETS[preset]["members"]
    return [
        MemberSpec(name=name, config=config, checkpoint=checkpoint, predictions=predictions)
        for name, config, checkpoint, predictions in default_rows
    ]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights.astype(np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    return np.sum(values * weights[:, None, None], axis=0)


def combine_scores(member_logits: np.ndarray, method: str, weights: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if member_logits.ndim != 3 or member_logits.shape[2] != NUM_CLASSES:
        raise ValueError(f"Expected member logits [M,N,{NUM_CLASSES}], got {member_logits.shape}")
    if method == "logit_average":
        ensemble_logits = member_logits.mean(axis=0)
        ensemble_prob = softmax_np(ensemble_logits)
    elif method == "probability_average":
        member_probs = np.stack([softmax_np(member_logits[i]) for i in range(member_logits.shape[0])], axis=0)
        ensemble_prob = member_probs.mean(axis=0)
        ensemble_logits = np.log(np.clip(ensemble_prob, 1e-12, None))
    elif method == "weighted_logit_average":
        if weights is None:
            raise ValueError("--weights is required for weighted_logit_average")
        if len(weights) != member_logits.shape[0]:
            raise ValueError(f"Expected {member_logits.shape[0]} weights, got {len(weights)}")
        ensemble_logits = weighted_average(member_logits, np.asarray(weights, dtype=np.float64))
        ensemble_prob = softmax_np(ensemble_logits)
    elif method == "weighted_probability_average":
        if weights is None:
            raise ValueError("--weights is required for weighted_probability_average")
        if len(weights) != member_logits.shape[0]:
            raise ValueError(f"Expected {member_logits.shape[0]} weights, got {len(weights)}")
        member_probs = np.stack([softmax_np(member_logits[i]) for i in range(member_logits.shape[0])], axis=0)
        ensemble_prob = weighted_average(member_probs, np.asarray(weights, dtype=np.float64))
        ensemble_logits = np.log(np.clip(ensemble_prob, 1e-12, None))
    else:
        raise ValueError(f"Unsupported method: {method}")
    return ensemble_logits, ensemble_prob


def id_column(fieldnames: list[str]) -> str:
    for candidate in ("sample_id", "graph_id", "index"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Cannot safely align predictions: no sample_id, graph_id, or index column found"
    )


def true_column(fieldnames: list[str]) -> str:
    for candidate in ("y_true", "true_label", "label"):
        if candidate in fieldnames:
            return candidate
    raise ValueError("Predictions file has no y_true/true_label/label column")


def pred_column(fieldnames: list[str]) -> str | None:
    for candidate in ("y_pred", "pred_label", "prediction"):
        if candidate in fieldnames:
            return candidate
    return None


def score_columns(fieldnames: list[str]) -> list[str]:
    for prefix in ("score_", "logit_", "ensemble_logits_"):
        cols = [f"{prefix}{i}" for i in range(NUM_CLASSES)]
        if all(col in fieldnames for col in cols):
            return cols
    raise ValueError("Predictions file must contain score_0..score_6 or logit_0..logit_6")


def read_prediction_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Predictions file has no header: {path}")
        fields = list(reader.fieldnames)
        id_col = id_column(fields)
        y_col = true_column(fields)
        p_col = pred_column(fields)
        s_cols = score_columns(fields)
        rows = list(reader)
    ids: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    logits: list[list[float]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row[id_col])
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        seen.add(sample_id)
        ids.append(sample_id)
        y_true.append(int(row[y_col]))
        if p_col is not None and row.get(p_col, "") != "":
            y_pred.append(int(row[p_col]))
        else:
            y_pred.append(int(np.argmax([float(row[col]) for col in s_cols])))
        logits.append([float(row[col]) for col in s_cols])
    return {
        "path": str(path),
        "id_col": id_col,
        "ids": ids,
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float64),
    }


def load_prediction_members(members: list[MemberSpec]) -> dict[str, Any]:
    if not members:
        raise ValueError("No prediction members provided")
    loaded = [(member, read_prediction_file(member.predictions or "")) for member in members]
    reference_ids = loaded[0][1]["ids"]
    reference_set = set(reference_ids)
    for member, data in loaded[1:]:
        ids = data["ids"]
        current_set = set(ids)
        if current_set != reference_set:
            missing = sorted(reference_set - current_set)[:5]
            extra = sorted(current_set - reference_set)[:5]
            raise ValueError(
                f"Prediction IDs do not match for {member.name}: missing={missing}, extra={extra}"
            )
    index_by_id = {sample_id: idx for idx, sample_id in enumerate(reference_ids)}
    member_logits = []
    member_preds = {}
    y_true_ref = loaded[0][1]["y_true"]
    for member, data in loaded:
        order = [data["ids"].index(sample_id) for sample_id in reference_ids]
        y_true = data["y_true"][order]
        if not np.array_equal(y_true_ref, y_true):
            bad = np.where(y_true_ref != y_true)[0][:5].tolist()
            raise ValueError(f"y_true mismatch for {member.name} at aligned rows {bad}")
        member_logits.append(data["logits"][order])
        member_preds[member.name] = data["y_pred"][order]
    return {
        "sample_ids": reference_ids,
        "graph_ids": reference_ids,
        "y_true": y_true_ref,
        "member_logits": np.stack(member_logits, axis=0),
        "member_preds": member_preds,
        "aligned_samples": len(reference_ids),
        "alignment_key": loaded[0][1]["id_col"],
    }


def load_checkpoint_members(
    members: list[MemberSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    from common import apply_cli_overrides, build_dataloader, load_checkpoint_model, load_config
    from training.trainer import move_to_device

    if not members:
        raise ValueError("No checkpoint members provided")
    models = []
    configs = []
    devices = []
    for member in members:
        if member.config is None or member.checkpoint is None:
            raise ValueError(f"Checkpoint member {member.name} needs config and checkpoint")
        override_args = SimpleNamespace(
            environment=args.environment,
            batch_size=args.batch_size,
            epochs=None,
            device=args.device,
            graph_repo_path=args.graph_repo_path,
            csv_root=None,
            output_root=None,
            max_train_batches=None,
            max_val_batches=None,
            max_test_batches=None,
            no_wandb=True,
            wandb=False,
            wandb_project=None,
            wandb_entity=None,
            num_workers=None,
            pin_memory=None,
            persistent_workers=None,
            prefetch_factor=None,
            chunk_cache_size=args.chunk_cache_size,
            graph_cache_chunks=None,
            chunk_aware_shuffle=False,
            no_chunk_aware_shuffle=False,
            profile_batches=None,
            amp=False,
            no_amp=False,
        )
        config = apply_cli_overrides(load_config(member.config, environment=args.environment), override_args)
        model, device, _ = load_checkpoint_model(config, member.checkpoint)
        models.append(model)
        configs.append(config)
        devices.append(device)
    if len({str(device) for device in devices}) != 1:
        raise RuntimeError(f"Ensemble members resolved different devices: {devices}")
    device = devices[0]
    loader = build_dataloader(configs[0], split=args.split, shuffle=False)
    y_true: list[int] = []
    graph_ids: list[str] = []
    member_logits: list[list[list[float]]] = [[] for _ in members]
    member_preds: dict[str, list[int]] = {member.name: [] for member in members}
    correct_examples: list[dict[str, Any]] = []
    wrong_examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = move_to_device(batch, device)
            logits_for_batch = []
            for idx, model in enumerate(models):
                logits = model(batch)["logits"].detach().float()
                logits_for_batch.append(logits)
                pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
                member_logits[idx].extend(logits.cpu().numpy().tolist())
                member_preds[members[idx].name].extend(pred.tolist())
            stacked = torch.stack(logits_for_batch, dim=0).cpu().numpy()
            ens_logits, ens_prob = combine_scores(
                stacked,
                args.method,
                parse_weights(args.weights) if args.method.startswith("weighted_") else None,
            )
            ens_pred = ens_logits.argmax(axis=1).astype(int)
            labels = batch["y"].detach().cpu().numpy().astype(int)
            gids = batch["graph_id"].detach().cpu().numpy().astype(int)
            y_true.extend(labels.tolist())
            graph_ids.extend([str(x) for x in gids.tolist()])
            if len(correct_examples) < 10 or len(wrong_examples) < 10:
                x_cpu = batch["x"].detach().cpu()
                for i in range(x_cpu.shape[0]):
                    item = {
                        "graph_id": int(gids[i]),
                        "y_true": int(labels[i]),
                        "y_pred": int(ens_pred[i]),
                        "confidence": float(ens_prob[i, ens_pred[i]]),
                        "image": x_cpu[i, :, 0].float().reshape(48, 48).numpy(),
                    }
                    if labels[i] == ens_pred[i] and len(correct_examples) < 10:
                        correct_examples.append(item)
                    elif labels[i] != ens_pred[i] and len(wrong_examples) < 10:
                        wrong_examples.append(item)
    return {
        "sample_ids": graph_ids,
        "graph_ids": graph_ids,
        "y_true": np.asarray(y_true, dtype=np.int64),
        "member_logits": np.asarray(member_logits, dtype=np.float64),
        "member_preds": {name: np.asarray(pred, dtype=np.int64) for name, pred in member_preds.items()},
        "aligned_samples": len(y_true),
        "alignment_key": "graph_id",
        "correct_examples": correct_examples,
        "wrong_examples": wrong_examples,
    }


def parse_weights(value: str | None) -> list[float] | None:
    if value is None or str(value).strip() == "":
        return None
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def summarize_member_predictions(
    members: list[MemberSpec],
    y_true: np.ndarray,
    member_preds: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for member in members:
        pred = np.asarray(member_preds[member.name], dtype=np.int64)
        metric = compute_metrics(y_true, pred)
        rows.append(
            {
                "model": member.name,
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "weighted_f1": metric["weighted_f1"],
                "pred_count": np.bincount(pred, minlength=NUM_CLASSES).astype(int).tolist(),
            }
        )
    for name, metrics_path in {
        "D7A baseline": "output/d7a_graph_swin_standalone/evaluation/metrics.json",
        "region_transformer_original": "output/d7a_graph_swin_region_transformer/evaluation/metrics.json",
    }.items():
        path = Path(metrics_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": name,
                    "accuracy": float(data.get("accuracy")),
                    "macro_f1": float(data.get("macro_f1")),
                    "weighted_f1": float(data.get("weighted_f1")),
                    "pred_count": data.get("pred_count"),
                }
            )
    return rows


def classification_report_lines(report: dict[str, Any]) -> list[str]:
    lines = []
    for label, values in report.items():
        if isinstance(values, dict):
            lines.append(
                f"{label:<14} "
                f"precision={values.get('precision', 0.0):.4f} "
                f"recall={values.get('recall', 0.0):.4f} "
                f"f1={values.get('f1-score', 0.0):.4f} "
                f"support={values.get('support', 0.0):.0f}"
            )
        else:
            lines.append(f"{label:<14} {float(values):.4f}")
    return lines


def confusion_focus(cm: np.ndarray) -> dict[str, int]:
    return {name: int(cm[src, dst]) for name, src, dst in CONFUSION_FOCUS}


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dump_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_confusion_matrix(cm: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = float(np.trace(cm) / max(np.sum(cm), 1))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - accuracy={accuracy:.4f}")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_checkpoint_example_grid(examples: list[dict[str, Any]], path: Path, title: str) -> None:
    from evaluation.evaluator import save_example_grid

    save_example_grid(examples, path, title)


def save_placeholder_examples(path: Path, title: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.38, detail, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_predictions(
    path: Path,
    members: list[MemberSpec],
    sample_ids: list[str],
    y_true: np.ndarray,
    ensemble_logits: np.ndarray,
    ensemble_prob: np.ndarray,
    ensemble_pred: np.ndarray,
    member_logits: np.ndarray,
    member_preds: dict[str, np.ndarray],
    save_logits: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["graph_id", "y_true", "y_pred", "true_label", "pred_label"]
    fields += [f"ensemble_logits_{i}" for i in range(NUM_CLASSES)]
    fields += [f"ensemble_prob_{i}" for i in range(NUM_CLASSES)]
    fields += [f"member_{member.name}_pred" for member in members]
    if save_logits:
        for member in members:
            fields += [f"member_{member.name}_logit_{i}" for i in range(NUM_CLASSES)]
            fields += [f"member_{member.name}_prob_{i}" for i in range(NUM_CLASSES)]
    fields.append("correct")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_idx, sample_id in enumerate(sample_ids):
            row: dict[str, Any] = {
                "graph_id": sample_id,
                "y_true": int(y_true[row_idx]),
                "y_pred": int(ensemble_pred[row_idx]),
                "true_label": int(y_true[row_idx]),
                "pred_label": int(ensemble_pred[row_idx]),
                "correct": bool(int(y_true[row_idx]) == int(ensemble_pred[row_idx])),
            }
            for i in range(NUM_CLASSES):
                row[f"ensemble_logits_{i}"] = float(ensemble_logits[row_idx, i])
                row[f"ensemble_prob_{i}"] = float(ensemble_prob[row_idx, i])
            for member_idx, member in enumerate(members):
                row[f"member_{member.name}_pred"] = int(member_preds[member.name][row_idx])
                if save_logits:
                    member_prob = softmax_np(member_logits[member_idx])
                    for i in range(NUM_CLASSES):
                        row[f"member_{member.name}_logit_{i}"] = float(member_logits[member_idx, row_idx, i])
                        row[f"member_{member.name}_prob_{i}"] = float(member_prob[row_idx, i])
            writer.writerow(row)


def run_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    preset = ENSEMBLE_PRESETS[args.ensemble_preset]
    if args.method is None:
        args.method = preset["method"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]

    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    explicit_prediction_mode = bool(args.prediction_files) and not args.members
    if args.members:
        members = args.members
        for member in members:
            default = next((x for x in default_members(args.ensemble_preset) if x.name == member.name), None)
            if default is not None:
                member.predictions = default.predictions
    elif args.prediction_files:
        members = args.prediction_files
        for member in members:
            default = next((x for x in default_members(args.ensemble_preset) if x.name == member.name), None)
            if default is not None:
                member.config = default.config
                member.checkpoint = default.checkpoint
    else:
        members = default_members(args.ensemble_preset)

    checkpoint_ready = all(member.config and member.checkpoint and Path(member.checkpoint).exists() for member in members)
    prediction_ready = all(member.predictions and Path(member.predictions).exists() for member in members)

    if args.mode == "checkpoint" or (args.mode == "auto" and checkpoint_ready and not explicit_prediction_mode):
        mode = "checkpoint"
        if not checkpoint_ready:
            missing = [member.checkpoint for member in members if not member.checkpoint or not Path(member.checkpoint).exists()]
            raise FileNotFoundError(f"Checkpoint mode requested but missing checkpoints: {missing}")
        loaded = load_checkpoint_members(members, args)
    elif args.mode == "predictions" or (args.mode == "auto" and prediction_ready):
        mode = "predictions"
        if not prediction_ready:
            missing = [member.predictions for member in members if not member.predictions or not Path(member.predictions).exists()]
            raise FileNotFoundError(f"Predictions mode requested but missing files: {missing}")
        loaded = load_prediction_members(members)
    else:
        raise FileNotFoundError(
            "Could not choose ensemble mode: checkpoints and predictions are not complete"
        )

    weights = parse_weights(args.weights)
    ensemble_logits, ensemble_prob = combine_scores(loaded["member_logits"], args.method, weights)
    ensemble_pred = ensemble_logits.argmax(axis=1).astype(np.int64)
    y_true = loaded["y_true"].astype(np.int64)
    metrics = compute_metrics(y_true, ensemble_pred)
    report = classification_report_dict(y_true, ensemble_pred)
    cm = confusion_matrix_array(y_true, ensemble_pred)
    pred_count = np.bincount(ensemble_pred, minlength=NUM_CLASSES).astype(int).tolist()
    focus = confusion_focus(cm)
    member_comparison = summarize_member_predictions(members, y_true, loaded["member_preds"])

    metrics_payload = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "pred_count": pred_count,
        "classification_report": report,
    }
    dump_json(metrics_payload, eval_dir / "metrics.json")
    dump_json(report, eval_dir / "classification_report.json")
    (eval_dir / "classification_report.txt").write_text(
        "\n".join(classification_report_lines(report)) + "\n",
        encoding="utf-8",
    )
    save_confusion_matrix(cm, eval_dir / "confusion_matrix.png")
    save_predictions(
        eval_dir / "predictions.csv",
        members,
        loaded["graph_ids"],
        y_true,
        ensemble_logits,
        ensemble_prob,
        ensemble_pred,
        loaded["member_logits"],
        loaded["member_preds"],
        bool(args.save_logits),
    )
    if loaded.get("correct_examples"):
        save_checkpoint_example_grid(loaded["correct_examples"], eval_dir / "correct_examples.png", "10 correct ensemble predictions")
    else:
        save_placeholder_examples(
            eval_dir / "correct_examples.png",
            "Correct examples unavailable",
            "Predictions mode does not include pixel tensors. Use checkpoint mode for image grids.",
        )
    if loaded.get("wrong_examples"):
        save_checkpoint_example_grid(loaded["wrong_examples"], eval_dir / "wrong_examples.png", "10 wrong ensemble predictions")
    else:
        save_placeholder_examples(
            eval_dir / "wrong_examples.png",
            "Wrong examples unavailable",
            "Predictions mode does not include pixel tensors. Use checkpoint mode for image grids.",
        )

    per_class_rows = []
    for class_idx, class_name in enumerate(EMOTION_NAMES):
        values = report[class_name]
        per_class_rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    write_csv(
        output_dir / "ensemble_per_class_metrics.csv",
        per_class_rows,
        ["class_idx", "class_name", "precision", "recall", "f1", "support"],
    )
    write_csv(
        output_dir / "ensemble_confusion_focus.csv",
        [{"model": "ensemble", **focus}],
        ["model"] + [name for name, _, _ in CONFUSION_FOCUS],
    )
    write_csv(
        output_dir / "ensemble_member_comparison.csv",
        [
            {
                **row,
                "pred_count": json.dumps(row["pred_count"]),
            }
            for row in member_comparison
        ],
        ["model", "accuracy", "macro_f1", "weighted_f1", "pred_count"],
    )

    member_payload = [asdict(member) for member in members]
    has_d8b_member = any(member.name.startswith("d8b") for member in members)
    summary = {
        "mode": mode,
        "ensemble_preset": args.ensemble_preset,
        "has_d8b_member": has_d8b_member,
        "method": args.method,
        "weights": weights,
        "split": args.split,
        "aligned_samples": int(loaded["aligned_samples"]),
        "alignment_key": loaded["alignment_key"],
        "metrics": metrics_payload,
        "confusion_focus": focus,
        "member_comparison": member_comparison,
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "ensemble_summary.json")
    dump_json({"members": member_payload}, output_dir / "ensemble_members.json")
    (output_dir / "resolved_ensemble_config.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": mode,
                "ensemble_preset": args.ensemble_preset,
                "has_d8b_member": has_d8b_member,
                "method": args.method,
                "weights": weights,
                "split": args.split,
                "output_dir": str(output_dir),
                "members": member_payload,
                "aligned_samples": int(loaded["aligned_samples"]),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_md = make_report(summary, report, focus)
    (output_dir / "d7_ensemble_report.md").write_text(report_md, encoding="utf-8")
    if has_d8b_member:
        (output_dir / "d7_d8b_ensemble_report.md").write_text(report_md, encoding="utf-8")

    print("\n=======================================================")
    print("D7/D8B ENSEMBLE EVALUATION")
    print("=======================================================")
    print(f"mode:        {mode}")
    print(f"preset:      {args.ensemble_preset}")
    print(f"method:      {args.method}")
    print(f"samples:     {loaded['aligned_samples']}")
    print(f"accuracy:    {metrics['accuracy']:.4f}")
    print(f"macro_f1:    {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print(f"pred_count:  {pred_count}")
    print(f"outputs:     {output_dir}")
    return summary


def make_report(summary: dict[str, Any], report: dict[str, Any], focus: dict[str, int]) -> str:
    metric = summary["metrics"]
    is_d7_d8b = bool(summary.get("has_d8b_member"))
    title = "D7 + D8B Ensemble Official Evaluation Report" if is_d7_d8b else "D7 Ensemble Official Evaluation Report"
    decision = (
        "This D7 + D8B official ensemble candidate should be compared against the old "
        "D7 seed44 + long150 + window4 logit-average ensemble. If probability averaging "
        "keeps the expected accuracy, macro F1, and weighted F1 gains, freeze this as the "
        "next official ensemble while seed-repeat checks validate the D8B border020 single model."
        if is_d7_d8b
        else (
            "This official ensemble should be compared against the single-model champions: "
            "`seed44` for macro/research and `window4_region_transformer` for accuracy/weighted F1. "
            "If the ensemble keeps the expected gains without class collapse, freeze it as the D7 performance champion before opening D8."
        )
    )
    member_lines = [
        f"- `{row['model']}`: accuracy `{row['accuracy']:.4f}`, macro F1 `{row['macro_f1']:.4f}`, weighted F1 `{row['weighted_f1']:.4f}`"
        for row in summary["member_comparison"]
    ]
    class_lines = []
    for class_name in EMOTION_NAMES:
        values = report[class_name]
        class_lines.append(
            f"|{class_name}|{values['precision']:.4f}|{values['recall']:.4f}|{values['f1-score']:.4f}|{int(values['support'])}|"
        )
    focus_lines = [f"|{name}|{value}|" for name, value in focus.items()]
    return (
        f"# {title}\n\n"
        "## Summary\n\n"
        f"- mode: `{summary['mode']}`\n"
        f"- ensemble preset: `{summary.get('ensemble_preset', 'custom')}`\n"
        f"- method: `{summary['method']}`\n"
        f"- split: `{summary['split']}`\n"
        f"- aligned samples: `{summary['aligned_samples']}` by `{summary['alignment_key']}`\n"
        f"- accuracy: `{metric['accuracy']:.4f}`\n"
        f"- macro F1: `{metric['macro_f1']:.4f}`\n"
        f"- weighted F1: `{metric['weighted_f1']:.4f}`\n"
        f"- pred_count: `{metric['pred_count']}`\n\n"
        "## Members\n\n"
        + "\n".join(member_lines)
        + "\n\n## Per-Class Metrics\n\n"
        "|class|precision|recall|f1|support|\n|---|---|---|---|---|\n"
        + "\n".join(class_lines)
        + "\n\n## Confusion Focus\n\n"
        "|error|count|\n|---|---|\n"
        + "\n".join(focus_lines)
        + "\n\n## Decision\n\n"
        f"{decision}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="*", type=parse_member, default=None)
    parser.add_argument("--prediction_files", nargs="*", type=parse_prediction_file, default=None)
    parser.add_argument(
        "--ensemble_preset",
        choices=sorted(ENSEMBLE_PRESETS),
        default="d7_d8b_border020_area045_probavg",
        help="Named official ensemble preset used when --members/--prediction_files are omitted.",
    )
    parser.add_argument("--mode", choices=["auto", "checkpoint", "predictions"], default="auto")
    parser.add_argument(
        "--method",
        choices=["logit_average", "probability_average", "weighted_logit_average", "weighted_probability_average"],
        default=None,
        help="Ensemble score-combination method. Defaults to the selected preset method.",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--save_logits", type=str_to_bool, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_ensemble(args)


if __name__ == "__main__":
    main()
