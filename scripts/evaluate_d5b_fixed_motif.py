"""Evaluate a trained D5B-1 fixed motif classifier checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    apply_cli_overrides,
    build_dataloader,
    dump_json,
    load_config,
    output_root_from_checkpoint,
    resolve_device,
    resolve_existing_path,
    resolve_path,
)
from data.labels import EMOTION_NAMES
from evaluation.evaluator import evaluate_model, save_confusion_matrix, save_predictions_csv
from models.fixed_motif_classifier import FixedMotifMLPClassifier


def _torch_load(path: str | Path, device: torch.device | str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_prior_for_eval(config, checkpoint) -> torch.Tensor:
    prior_path = config.get("prior", {}).get("node_prior_path")
    if prior_path is not None:
        try:
            payload = _torch_load(resolve_existing_path(prior_path), device="cpu")
            if "node_prior" in payload:
                return payload["node_prior"]
        except FileNotFoundError:
            pass
    state = checkpoint.get("model_state_dict", checkpoint)
    if "node_prior" not in state:
        raise FileNotFoundError(
            "Could not load node prior from prior.node_prior_path or checkpoint model_state_dict"
        )
    return state["node_prior"].detach().cpu()


def load_d5b_checkpoint_model(config, checkpoint_path: str | Path):
    device = resolve_device(config=config)
    ckpt_path = resolve_existing_path(checkpoint_path)
    checkpoint = _torch_load(ckpt_path, device=device)
    node_prior = _load_prior_for_eval(config, checkpoint)
    model_cfg = dict(config.get("model", {}))
    model = FixedMotifMLPClassifier(
        node_prior=node_prior,
        node_dim=int(model_cfg.get("node_dim", 7)),
        num_classes=int(model_cfg.get("num_classes", 7)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, device, checkpoint


def _write_report_txt(report, out_path: Path) -> None:
    report_lines = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_lines.append(
                f"{label:<14} "
                f"precision={values.get('precision', 0.0):.4f} "
                f"recall={values.get('recall', 0.0):.4f} "
                f"f1={values.get('f1-score', 0.0):.4f} "
                f"support={values.get('support', 0.0):.0f}"
            )
        else:
            report_lines.append(f"{label:<14} {values:.4f}")
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_evaluate(config, checkpoint=None):
    paths = config.get("paths", {})
    experiment_name = config.get("experiment", {}).get("name", "d5b_1_fixed_motif_classifier")
    if checkpoint is not None:
        output_root = output_root_from_checkpoint(checkpoint)
    else:
        output_root = resolve_path(paths.get("resolved_output_root"))
    if output_root is None:
        output_base = resolve_path(paths.get("output_root", "output"))
        output_root = output_base / str(experiment_name)
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"

    model, device, _ = load_d5b_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split="test", shuffle=False)
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        max_batches=config.get("training", {}).get("max_test_batches"),
    )

    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics["confusion_matrix"], eval_dir / "confusion_matrix.png")
    save_predictions_csv(metrics, eval_dir / "predictions.csv")
    dump_json(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "pred_count": metrics["pred_count"],
            "class_names": EMOTION_NAMES,
            "classification_report": metrics["classification_report"],
        },
        eval_dir / "metrics.json",
    )
    dump_json(metrics["classification_report"], eval_dir / "classification_report.json")
    _write_report_txt(metrics["classification_report"], eval_dir / "classification_report.txt")

    print(
        "\n=======================================================\n"
        "D5B-1 TEST SET EVALUATION\n"
        "======================================================="
    )
    print(f"Accuracy:    {metrics['accuracy'] * 100.0:.2f}%")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"pred_count:  {metrics['pred_count']}")
    print(f"Evaluation outputs: {eval_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d5b_1_fixed_motif_classifier.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_evaluate(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
