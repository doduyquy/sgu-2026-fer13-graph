"""Evaluate a trained D5A checkpoint."""

from __future__ import annotations

import argparse

from common import (
    apply_cli_overrides,
    build_dataloader,
    dump_json,
    load_checkpoint_model,
    load_config,
    resolve_path,
)

from fer_d5.evaluation.evaluator import evaluate_model, save_confusion_matrix, save_predictions_csv


def run_evaluate(config, checkpoint=None):
    paths = config.get("paths", {})
    output_root = resolve_path(paths.get("output_root", "outputs"))
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"
    model, device, _ = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split="test", shuffle=False)
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        max_batches=config.get("training", {}).get("max_test_batches"),
    )
    eval_dir = output_root / "evaluation"
    save_confusion_matrix(metrics["confusion_matrix"], eval_dir / "confusion_matrix.png")
    save_predictions_csv(metrics, eval_dir / "predictions.csv")
    dump_json(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "classification_report": metrics["classification_report"],
        },
        eval_dir / "metrics.json",
    )
    print(
        f"test accuracy={metrics['accuracy']:.6f} "
        f"macro_f1={metrics['macro_f1']:.6f} weighted_f1={metrics['weighted_f1']:.6f}"
    )
    print(f"Evaluation outputs: {eval_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a_experiment.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    run_evaluate(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
