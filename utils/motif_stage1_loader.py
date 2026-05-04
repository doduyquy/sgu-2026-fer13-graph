"""Helpers for loading frozen D8M Stage 1 motif discovery checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from models.registry import build_model
from scripts.common import load_config, resolve_existing_path


def _extract_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain model_state_dict or state_dict")


def load_frozen_motif_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device | str):
    """Load a Stage 1 motif model, freeze all parameters, and switch it to eval mode."""

    config_file = resolve_existing_path(config_path)
    checkpoint_file = resolve_existing_path(checkpoint_path)
    config = load_config(config_file)
    model = build_model(dict(config.get("model", {}) or {})).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    missing, unexpected = model.load_state_dict(_extract_model_state(checkpoint), strict=False)
    if missing:
        print(f"[Stage1Loader] missing_keys={list(missing)}")
    if unexpected:
        print(f"[Stage1Loader] unexpected_keys={list(unexpected)}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable_params != 0:
        raise RuntimeError(f"Frozen Stage 1 still has trainable parameters: {trainable_params}")
    print(f"[Stage1Loader] total_stage1_params={total_params:,}")
    print(f"[Stage1Loader] trainable_stage1_params={trainable_params}")
    print(f"[Stage1Loader] checkpoint_loaded={checkpoint_file}")
    return model
