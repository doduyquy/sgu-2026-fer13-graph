"""Clean full-graph trainer for D5A with profiling and AMP support."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from evaluation.metrics import compute_metrics
from training.optimizer import step_scheduler


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    return value


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _sync(device: torch.device) -> None:
    """Synchronize CUDA device if applicable for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_mem_stats(device: torch.device) -> Dict[str, float]:
    """Return current CUDA memory stats in GiB (zeros on CPU)."""
    if device.type != "cuda":
        return {"cuda_allocated_gb": 0.0, "cuda_reserved_gb": 0.0, "cuda_max_allocated_gb": 0.0}
    gib = 1024 ** 3
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated(device) / gib,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / gib,
        "cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / gib,
    }


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
        amp: bool = False,
        profile_batches: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
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
        self._logged_train_device = False
        self.wandb = None
        if self.use_wandb:
            import wandb
            self.wandb = wandb

        # AMP setup
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        if bool(amp) and self.device.type != "cuda":
            print("[AMP] WARNING: AMP requested but device is not CUDA – AMP disabled.")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        print(f"[AMP] amp_enabled={self.amp_enabled}")

        # Profiling
        self.profile_batches = int(profile_batches)
        # configured_bs for bs_mismatch validation (set by fit() before training)
        self._configured_bs: Optional[int] = None

        first_param = next(self.model.parameters())
        print(f"trainer device: {self.device}")
        print(f"model first parameter device: {first_param.device}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        """Return the appropriate autocast context manager."""
        try:
            # torch >= 1.10 supports torch.amp.autocast
            return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=self.amp_enabled)

    def _print_profile(self, batch_idx: int, times: Dict[str, float], mem: Dict[str, float]) -> None:
        print(
            f"[PROFILE batch={batch_idx}]\n"
            f"  data_time      ={times.get('data', 0.0):.4f}s\n"
            f"  to_device_time ={times.get('to_device', 0.0):.4f}s\n"
            f"  forward_time   ={times.get('forward', 0.0):.4f}s\n"
            f"  loss_time      ={times.get('loss', 0.0):.4f}s\n"
            f"  backward_time  ={times.get('backward', 0.0):.4f}s\n"
            f"  optimizer_time ={times.get('optimizer', 0.0):.4f}s\n"
            f"  batch_time     ={times.get('batch', 0.0):.4f}s\n"
            f"  cuda_allocated_gb    ={mem.get('cuda_allocated_gb', 0.0):.3f}\n"
            f"  cuda_reserved_gb     ={mem.get('cuda_reserved_gb', 0.0):.3f}\n"
            f"  cuda_max_allocated_gb={mem.get('cuda_max_allocated_gb', 0.0):.3f}"
        )

    def _print_profile_avg(
        self,
        requested: int,
        recorded: int,
        acc: Dict[str, float],
        total_train_batches: Optional[int] = None,
    ) -> None:
        """Print average profile over *recorded* batches.

        requested = profile_batches config value
        recorded  = actual number of batches we accumulated (may be < requested
                    if max_train_batches cut training short).
        """
        avg = {k: v / max(recorded, 1) for k, v in acc.items()}
        est_epoch_min: Optional[float] = None
        if total_train_batches is not None and avg.get("batch", 0.0) > 0:
            est_epoch_min = avg["batch"] * total_train_batches / 60.0
        est_str = f"{est_epoch_min:.2f}" if est_epoch_min is not None else "unknown"
        print(
            f"\n[PROFILE average first {requested} batches (recorded={recorded})]\n"
            f"  avg_data_time      ={avg.get('data', 0.0):.4f}s\n"
            f"  avg_to_device_time ={avg.get('to_device', 0.0):.4f}s\n"
            f"  avg_forward_time   ={avg.get('forward', 0.0):.4f}s\n"
            f"  avg_loss_time      ={avg.get('loss', 0.0):.4f}s\n"
            f"  avg_backward_time  ={avg.get('backward', 0.0):.4f}s\n"
            f"  avg_optimizer_time ={avg.get('optimizer', 0.0):.4f}s\n"
            f"  avg_batch_time     ={avg.get('batch', 0.0):.4f}s\n"
            f"  estimated_full_epoch_minutes={est_str}"
        )

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        loader,
        epoch: int,
        max_batches: Optional[int] = None,
        total_train_batches: Optional[int] = None,
        full_epoch_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        epoch_start = time.perf_counter()
        progress = tqdm(loader, desc=f"train {epoch}", leave=False)

        # Profiling accumulators
        profile_n = self.profile_batches          # requested
        profile_recorded = 0                       # actually accumulated
        profile_acc: Dict[str, float] = {k: 0.0 for k in (
            "data", "to_device", "forward", "loss", "backward", "optimizer", "batch"
        )}
        profile_avg_printed = False

        # first-batch shape logging (only on epoch 1, batch_idx 0)
        _first_batch_logged = epoch > 1  # skip for epochs after first

        # Data timer starts right before the first batch is fetched
        t_data_start = time.perf_counter()

        for batch_idx, batch in enumerate(progress):
            is_last = (max_batches is not None and batch_idx + 1 >= int(max_batches))

            if max_batches is not None and batch_idx >= int(max_batches):
                break

            # ---- first-batch shape + bs_mismatch (from actual train loop, epoch 1 only) ----
            if not _first_batch_logged:
                x_shape = list(batch["x"].shape)
                ea_shape = list(batch["edge_attr"].shape)
                n_samples = x_shape[0]
                print(f"[SPEED_BENCH] first_batch_x_shape={x_shape}")
                print(f"[SPEED_BENCH] first_batch_edge_attr_shape={ea_shape}")
                configured_bs = self._configured_bs
                if configured_bs is not None:
                    if n_samples != configured_bs or ea_shape[0] != configured_bs:
                        print(
                            f"[SPEED_BENCH] bs_mismatch=True  "
                            f"(expected {configured_bs}, got x.shape[0]={n_samples}, "
                            f"edge_attr.shape[0]={ea_shape[0]})"
                        )
                    else:
                        print(f"[SPEED_BENCH] bs_mismatch=False  (x.shape[0]={n_samples} \u2713)")
                _first_batch_logged = True

            do_profile = profile_n > 0 and batch_idx < profile_n

            # ---- data_time ----
            if do_profile:
                _sync(self.device)
                t_data_end = time.perf_counter()
                t_batch_start = t_data_end
                data_time = t_data_end - t_data_start

            # ---- to_device_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            batch = move_to_device(batch, self.device)
            if do_profile:
                _sync(self.device)
                to_device_time = time.perf_counter() - t0

            self.optimizer.zero_grad(set_to_none=True)

            # ---- forward_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with self._autocast():
                out = self.model(batch)
            if do_profile:
                _sync(self.device)
                forward_time = time.perf_counter() - t0

            # ---- loss_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with self._autocast():
                loss_dict = self.criterion(out, batch["y"], batch)
            loss = loss_dict["loss"]
            if do_profile:
                _sync(self.device)
                loss_time = time.perf_counter() - t0

            # Device log (first batch only)
            if not self._logged_train_device:
                print(
                    "train tensor devices: "
                    f"x={batch['x'].device} edge_index={batch['edge_index'].device} "
                    f"edge_attr={batch['edge_attr'].device} y={batch['y'].device} "
                    f"logits={out['logits'].device} loss={loss.device}"
                )
                self._logged_train_device = True

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}")

            # ---- backward_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.scale(loss).backward()
            if do_profile:
                _sync(self.device)
                backward_time = time.perf_counter() - t0

            # ---- optimizer_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.grad_clip_norm)
                )
            else:
                grad_norm = self._compute_grad_norm()

            grad_norm_finite = bool(
                torch.isfinite(torch.as_tensor(grad_norm)).detach().cpu().item()
            )
            if not grad_norm_finite:
                raise FloatingPointError(f"Non-finite grad norm at batch {batch_idx}")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if do_profile:
                _sync(self.device)
                optimizer_time = time.perf_counter() - t0

            # ---- profiling record ----
            if do_profile:
                _sync(self.device)
                batch_time = time.perf_counter() - t_batch_start
                mem = _cuda_mem_stats(self.device)
                times = {
                    "data": data_time,
                    "to_device": to_device_time,
                    "forward": forward_time,
                    "loss": loss_time,
                    "backward": backward_time,
                    "optimizer": optimizer_time,
                    "batch": batch_time,
                }
                self._print_profile(batch_idx, times, mem)
                for k, v in times.items():
                    profile_acc[k] += v
                profile_recorded += 1

                # Print avg when requested window is full, OR on last batch
                # (in case max_batches cuts the run before profile_n).
                window_done = (batch_idx == profile_n - 1)
                last_profile_batch = is_last and not profile_avg_printed
                if (window_done or last_profile_batch) and not profile_avg_printed:
                    # Use full_epoch_batches for accurate epoch estimate;
                    # fall back to total_train_batches if not provided.
                    epoch_batches_for_estimate = full_epoch_batches or total_train_batches
                    self._print_profile_avg(
                        requested=profile_n,
                        recorded=profile_recorded,
                        acc=profile_acc,
                        total_train_batches=epoch_batches_for_estimate,
                    )
                    profile_avg_printed = True

            pred = out["logits"].detach().argmax(dim=1).cpu()
            pred_count += torch.bincount(pred, minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + _to_float(grad_norm)
            self._add_diagnostics(totals, out.get("diagnostics", {}))
            count += 1
            progress.set_postfix(loss=f"{_to_float(loss):.4f}")

            # Reset data timer for next batch
            t_data_start = time.perf_counter()

        # If profiling window was never printed (e.g. profile_n=0 or nothing recorded),
        # print a final avg if we did accumulate something.
        if profile_recorded > 0 and not profile_avg_printed:
            epoch_batches_for_estimate = full_epoch_batches or total_train_batches
            self._print_profile_avg(
                requested=profile_n,
                recorded=profile_recorded,
                acc=profile_acc,
                total_train_batches=epoch_batches_for_estimate,
            )

        metrics = {f"train_{k}": v / max(count, 1) for k, v in totals.items()}
        metrics["train_batches"] = float(count)
        elapsed = time.perf_counter() - epoch_start
        metrics["train_seconds"] = float(elapsed)
        metrics["train_sec_per_batch"] = float(elapsed / max(count, 1))
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
        start_time = time.perf_counter()
        for batch_idx, batch in enumerate(tqdm(loader, desc=prefix, leave=False)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, self.device)
            with self._autocast():
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
        elapsed = time.perf_counter() - start_time
        metrics[f"{prefix}_seconds"] = float(elapsed)
        metrics[f"{prefix}_sec_per_batch"] = float(elapsed / max(count, 1))
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
        import math
        dataset_size = getattr(train_loader.dataset, "__len__", lambda: None)()
        batch_size = getattr(train_loader, "batch_size", None)
        # Set configured_bs for bs_mismatch validation in train_one_epoch
        self._configured_bs = int(batch_size) if batch_size is not None else None
        # full_epoch_batches: used for estimated epoch time (NOT capped by max_train_batches).
        # total_train_batches: passed to train_one_epoch as the effective batches this run.
        if dataset_size is not None and batch_size is not None and batch_size > 0:
            full_epoch_batches = math.ceil(dataset_size / batch_size)
            total_train_batches = (
                min(full_epoch_batches, int(max_train_batches))
                if max_train_batches is not None
                else full_epoch_batches
            )
        else:
            full_epoch_batches = None
            total_train_batches = None

        stale_epochs = 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.train_one_epoch(
                train_loader,
                epoch,
                max_batches=max_train_batches,
                total_train_batches=total_train_batches,
                full_epoch_batches=full_epoch_batches,
            )
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
                f"best={self.best_metric:.4f} "
                f"train_sec/batch={metrics.get('train_sec_per_batch', 0.0):.3f}s"
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

    def _compute_grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.detach().norm(2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=self.device)
        return torch.norm(torch.stack(norms), p=2)
