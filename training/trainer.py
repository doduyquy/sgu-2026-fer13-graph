"""Clean full-graph trainer for D5A."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from fer_d5.evaluation.metrics import compute_metrics
from fer_d5.training.optimizer import step_scheduler


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    return value


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class D5Trainer:
    """Trainer with one intentional route: full graph D5 retrieval."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device | str = "cpu",
        output_root: str | Path = "outputs",
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
        grad_clip_norm: Optional[float] = 5.0,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.output_root = Path(output_root)
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.use_wandb = bool(use_wandb)
        self.grad_clip_norm = grad_clip_norm
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.wandb = None
        if self.use_wandb:
            import wandb

            self.wandb = wandb

    def train_one_epoch(self, loader, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        progress = tqdm(loader, desc=f"train {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch)
            loss_dict = self.criterion(out, batch["y"], batch)
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}")
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
            self.optimizer.step()

            pred = out["logits"].detach().argmax(dim=1).cpu()
            pred_count += torch.bincount(pred, minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            self._add_diagnostics(totals, out.get("diagnostics", {}))
            count += 1
            progress.set_postfix(loss=f"{_to_float(loss):.4f}")

        metrics = {f"train_{k}": v / max(count, 1) for k, v in totals.items()}
        metrics["train_batches"] = float(count)
        for i, c in enumerate(pred_count.tolist()):
            metrics[f"train_pred_count_{i}"] = float(c)
        return metrics

    @torch.no_grad()
    def validate(self, loader, max_batches: Optional[int] = None, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        y_true = []
        y_pred = []
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        for batch_idx, batch in enumerate(tqdm(loader, desc=prefix, leave=False)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, self.device)
            out = self.model(batch)
            loss_dict = self.criterion(out, batch["y"], batch)
            logits = out["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits during {prefix} at batch {batch_idx}")
            pred = logits.argmax(dim=1)
            y_true.extend(batch["y"].detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            pred_count += torch.bincount(pred.detach().cpu(), minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            self._add_diagnostics(totals, out.get("diagnostics", {}))
            count += 1

        metrics = {f"{prefix}_{k}": v / max(count, 1) for k, v in totals.items()}
        cls_metrics = compute_metrics(y_true, y_pred) if y_true else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
        metrics.update({f"{prefix}_{k}": float(v) for k, v in cls_metrics.items()})
        metrics[f"{prefix}_batches"] = float(count)
        for i, c in enumerate(pred_count.tolist()):
            metrics[f"{prefix}_pred_count_{i}"] = float(c)
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        monitor: str = "val_macro_f1",
        early_stopping_patience: int = 20,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        stale_epochs = 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch, max_train_batches)
            val_metrics = self.validate(val_loader, max_val_batches, prefix="val")
            metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
            monitor_value = float(metrics.get(monitor, val_metrics.get("val_macro_f1", 0.0)))
            step_scheduler(self.scheduler, monitor_value)
            improved = monitor_value > self.best_metric
            if improved:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                stale_epochs = 0
                self.save_checkpoint("best.pth", epoch, metrics)
            else:
                stale_epochs += 1
            self.save_checkpoint("last.pth", epoch, metrics)
            history.append(metrics)
            self._log_metrics(metrics)
            print(
                f"epoch={epoch:03d} train_loss={metrics.get('train_loss', 0.0):.4f} "
                f"val_macro_f1={metrics.get('val_macro_f1', 0.0):.4f} "
                f"best={self.best_metric:.4f}"
            )
            if stale_epochs >= int(early_stopping_patience):
                print(f"Early stopping after {stale_epochs} stale epochs")
                break
        self._save_history(history)
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "history": history}

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        return path

    def _save_history(self, history) -> None:
        path = self.output_root / "training_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics)

    @staticmethod
    def _add_diagnostics(totals: Dict[str, float], diagnostics: Dict[str, Any]) -> None:
        for key, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[f"diag_{key}"] = totals.get(f"diag_{key}", 0.0) + _to_float(value)
