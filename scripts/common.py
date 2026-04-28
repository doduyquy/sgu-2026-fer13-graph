"""Shared helpers for D5 command-line scripts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from data.full_graph_dataset import FullGraphDataset, collate_fn_full_graph
from data.graph_config import GraphConfig
from models.registry import build_model
from training.losses import D5RetrievalLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import D5Trainer, set_seed


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, environment: str | None = None) -> Dict[str, Any]:
    path = resolve_existing_path(config_path)
    cfg = _load_config_tree(path)
    return resolve_environment_config(cfg, environment=environment)


def _load_config_tree(config_path: str | Path, seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherited = cfg.pop("inherits", [])
    if isinstance(inherited, (str, Path)):
        inherited = [inherited]

    merged: Dict[str, Any] = {}
    for parent in inherited:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_update(merged, _load_config_tree(parent_path, seen))
    seen.remove(path)
    return deep_update(merged, cfg)


def resolve_environment_config(config: Dict[str, Any], environment: str | None = None) -> Dict[str, Any]:
    cfg = dict(config)
    env = environment or cfg.get("environment") or os.environ.get("D5_ENV")
    if not env:
        return cfg
    env = str(env).lower()
    profiles = cfg.get("environments", {})
    if profiles and env not in profiles:
        raise ValueError(f"Unknown environment={env!r}. Available: {sorted(profiles)}")
    if profiles:
        cfg = deep_update(cfg, profiles.get(env, {}) or {})
    cfg["environment"] = env
    return cfg


def save_config(config: Dict[str, Any], output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Path not found: {path_like}")


def resolve_path(path_like: str | Path | None, default: Optional[str] = None) -> Optional[Path]:
    if path_like is None:
        path_like = default
    if path_like is None:
        return None
    value = str(path_like)
    if value.lower() == "auto":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_csv_root(csv_root: str | Path | None = "auto") -> Path:
    if csv_root is not None and str(csv_root).lower() != "auto":
        path = Path(csv_root)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if _has_split_csvs(path):
            return path
        raise FileNotFoundError(f"CSV root does not contain train/val/test CSVs: {path}")

    candidates = [
        Path.cwd(),
        Path.cwd() / "data",
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
    ]
    for candidate in candidates:
        if _has_split_csvs(candidate):
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_csv in kaggle_input.rglob("train.csv"):
            candidate = train_csv.parent
            if _has_split_csvs(candidate):
                return candidate.resolve()

    for train_csv in PROJECT_ROOT.parent.rglob("train.csv"):
        candidate = train_csv.parent
        if _has_split_csvs(candidate):
            return candidate.resolve()
    raise FileNotFoundError("Could not auto-find train.csv, val.csv, and test.csv")


def _has_split_csvs(path: Path) -> bool:
    return all((path / f"{split}.csv").exists() for split in ("train", "val", "test"))


def split_csv_paths(csv_root: str | Path | None = "auto") -> Dict[str, Path]:
    root = find_csv_root(csv_root)
    return {split: root / f"{split}.csv" for split in ("train", "val", "test")}


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)
    environment = getattr(args, "environment", None)
    if environment is not None:
        cfg = resolve_environment_config(cfg, environment=environment)
    paths = dict(cfg.get("paths", {}))
    data = dict(cfg.get("data", {}))
    training = dict(cfg.get("training", {}))
    logging_cfg = dict(cfg.get("logging", {}))

    for attr in ("csv_root", "graph_repo_path", "output_root"):
        value = getattr(args, attr, None)
        if value is not None:
            paths[attr] = value
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        training["epochs"] = int(args.epochs)
    if getattr(args, "device", None) is not None:
        training["device"] = str(args.device)
    for attr in ("max_train_batches", "max_val_batches", "max_test_batches"):
        value = getattr(args, attr, None)
        if value is not None:
            training[attr] = int(value)
    if getattr(args, "no_wandb", False):
        logging_cfg["use_wandb"] = False

    cfg["paths"] = paths
    cfg["data"] = data
    cfg["training"] = training
    cfg["logging"] = logging_cfg
    return cfg


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool = False,
) -> DataLoader:
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    repo = resolve_path(paths.get("graph_repo_path", "artifacts/graph_repo"))
    dataset = FullGraphDataset(
        repo_root=repo,
        split=split,
        graph_cache_chunks=int(data_cfg.get("graph_cache_chunks", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=bool(shuffle),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=collate_fn_full_graph,
    )


def resolve_device(device_arg: str | None = None, config: Optional[Dict[str, Any]] = None) -> torch.device:
    config = config or {}
    requested = (
        device_arg
        or config.get("training", {}).get("device")
        or config.get("device")
        or "auto"
    )
    requested = "auto" if requested is None else str(requested).strip()
    if requested == "" or requested.lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif requested.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda:0" if requested.lower() == "cuda" else requested)
    else:
        device = torch.device(requested)

    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def log_device_info(device: torch.device | str) -> None:
    device = torch.device(device)
    print(f"--- torch version: {torch.__version__}")
    print(f"--- cuda available: {torch.cuda.is_available()}")
    print(f"--- cuda device count: {torch.cuda.device_count()}")
    print(f"--- selected device: {device}")
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        print(f"--- gpu name: {torch.cuda.get_device_name(idx)}")
        print(f"--- current cuda device: {torch.cuda.current_device()}")


def infer_device(config: Dict[str, Any]) -> torch.device:
    return resolve_device(config=config)


def prepare_training_objects(config: Dict[str, Any]):
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = infer_device(config)
    log_device_info(device)
    graph_cfg = GraphConfig.from_dict(config.get("graph", {}))
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("height", graph_cfg.height)
    model_cfg.setdefault("width", graph_cfg.width)
    model_cfg.setdefault("connectivity", graph_cfg.connectivity)
    model = build_model(model_cfg).to(device)
    criterion = D5RetrievalLoss(config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    return model, criterion, optimizer, scheduler, device


def create_trainer(config: Dict[str, Any]) -> D5Trainer:
    model, criterion, optimizer, scheduler, device = prepare_training_objects(config)
    paths = config.get("paths", {})
    logging_cfg = config.get("logging", {})
    output_root = resolve_path(paths.get("output_root", "outputs"))
    save_config(config, output_root)
    return D5Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_root=output_root,
        config=config,
        use_wandb=bool(logging_cfg.get("use_wandb", False)),
        grad_clip_norm=config.get("training", {}).get("grad_clip_norm", 5.0),
    )


def load_checkpoint_model(config: Dict[str, Any], checkpoint_path: str | Path):
    model, _, _, _, device = prepare_training_objects(config)
    ckpt_path = resolve_existing_path(checkpoint_path)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device, checkpoint


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return str(value)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
