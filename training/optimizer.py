"""Optimizer and scheduler builders."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    cfg = dict(config)
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 7e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[Dict[str, Any]],
):
    if not config:
        return None
    cfg = dict(config)
    name = cfg.get("name")
    if name in (None, "none", "None"):
        return None
    if str(name).lower() == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "max"),
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
    if str(name).lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("t_max", 20)),
            eta_min=float(cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def step_scheduler(scheduler, monitor_value: float | None = None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(monitor_value if monitor_value is not None else 0.0))
    else:
        scheduler.step()
