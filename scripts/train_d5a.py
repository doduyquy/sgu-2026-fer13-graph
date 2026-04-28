"""Train D5A class-level pixel motif graph retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import apply_cli_overrides, build_dataloader, create_trainer, load_config


def run_train(config):
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    trainer = create_trainer(config)
    training_cfg = config.get("training", {})
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_cfg.get("epochs", 80)),
        monitor=training_cfg.get("monitor", "val_macro_f1"),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 20)),
        max_train_batches=training_cfg.get("max_train_batches"),
        max_val_batches=training_cfg.get("max_val_batches"),
    )
    print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_train(config)


if __name__ == "__main__":
    main()
