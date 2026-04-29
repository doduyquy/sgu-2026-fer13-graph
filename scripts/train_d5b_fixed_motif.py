"""Train D5B-1 fixed motif MLP classifier."""

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
    load_config,
    log_device_info,
    resolve_device,
    resolve_existing_path,
    resolve_path,
    save_config,
)
from models.fixed_motif_classifier import FixedMotifMLPClassifier
from training.losses import FixedMotifClassificationLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import D5Trainer, set_seed


def _torch_load(path: str | Path, device: torch.device | str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_d5b_model(config, device: torch.device) -> FixedMotifMLPClassifier:
    prior_cfg = config.get("prior", {})
    prior_path = resolve_existing_path(prior_cfg.get("node_prior_path", "artifacts/d5b_motif_prior/node_prior.pt"))
    payload = _torch_load(prior_path, device="cpu")
    if "node_prior" not in payload:
        raise KeyError(f"Missing node_prior in {prior_path}")
    model_cfg = dict(config.get("model", {}))
    model_cfg.pop("name", None)
    model_cfg.pop("freeze_prior", None)
    model = FixedMotifMLPClassifier(
        node_prior=payload["node_prior"],
        node_dim=int(model_cfg.get("node_dim", 7)),
        num_classes=int(model_cfg.get("num_classes", 7)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    return model.to(device)


def create_d5b_trainer(config) -> D5Trainer:
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config=config)
    log_device_info(device)
    model = build_d5b_model(config, device=device)
    criterion = FixedMotifClassificationLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))

    experiment_name = config.get("experiment", {}).get("name", "d5b_1_fixed_motif_classifier")
    paths = config.get("paths", {})
    output_root = resolve_path(paths.get("resolved_output_root"))
    if output_root is None:
        output_base = resolve_path(paths.get("output_root", "output"))
        output_root = output_base / str(experiment_name)
    config.setdefault("paths", {})["resolved_output_root"] = str(output_root)
    save_config(config, output_root)
    print(f"[Output] run_dir={output_root}")

    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    return D5Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_root=output_root,
        config=config,
        use_wandb=bool(logging_cfg.get("use_wandb", False)),
        wandb_project=logging_cfg.get("project"),
        wandb_entity=logging_cfg.get("entity"),
        wandb_run_name=logging_cfg.get("run_name"),
        grad_clip_norm=training_cfg.get("grad_clip_norm", 5.0),
        amp=bool(training_cfg.get("amp", False)),
        profile_batches=int(training_cfg.get("profile_batches", 0)),
    )


def run_train(config):
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    trainer = create_d5b_trainer(config)
    training_cfg = config.get("training", {})
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_cfg.get("epochs", 40)),
        monitor=training_cfg.get("monitor", "val_macro_f1"),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 10)),
        max_train_batches=training_cfg.get("max_train_batches"),
        max_val_batches=training_cfg.get("max_val_batches"),
    )
    print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d5b_1_fixed_motif_classifier.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--profile_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", default=None)
    parser.add_argument("--persistent_workers", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--no_chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_train(config)


if __name__ == "__main__":
    main()
