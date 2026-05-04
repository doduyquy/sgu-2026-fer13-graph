"""Audit teacher saliency against D8M Stage 1 selected motif maps.

This is a read-only diagnostic script: it loads frozen checkpoints, computes
input-gradient saliency when possible, and writes visual/CSV evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device, set_seed  # noqa: E402


CSV_FIELDS = [
    "sample_id",
    "graph_id",
    "label",
    "teacher_pred",
    "teacher_confidence",
    "saliency_border_mass",
    "saliency_center_mass",
    "motif_border_mass",
    "motif_center_mass",
    "teacher_motif_cosine",
    "teacher_motif_iou_top20",
    "notes",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_torch(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_model_from_config(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    config = load_config(config_path)
    model_cfg = dict(config.get("model", {}) or {})
    model = build_model(model_cfg).to(device)
    ckpt_path = resolve_existing_path(checkpoint_path)
    checkpoint = _load_torch(ckpt_path, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config, ckpt_path


def _normalize_tensor_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = values.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1)
    return ((values - min_v) / (max_v - min_v).clamp_min(eps)).clamp(0.0, 1.0)


def _normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _extract_logits(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ("logits", "logits_fused", "logits_swin", "aux_logits"):
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return value
    available = ", ".join(sorted(out.keys()))
    raise KeyError(f"Teacher output has no 2D logits tensor; available keys: {available}")


def _target_classes(logits: torch.Tensor, labels: torch.Tensor | None, target_mode: str) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    if target_mode == "predicted" or labels is None:
        return pred
    if target_mode != "ground_truth":
        raise ValueError(f"Unsupported teacher.target={target_mode!r}; use ground_truth or predicted")
    return labels.long().clamp(min=0, max=logits.shape[1] - 1)


def _attention_to_pixel_map(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    class_region_attn = out.get("class_region_attn")
    if torch.is_tensor(class_region_attn) and class_region_attn.ndim == 3:
        bsz, _, num_regions = class_region_attn.shape
        side = int(round(math.sqrt(float(num_regions))))
        if side * side != num_regions:
            raise ValueError(f"class_region_attn has non-square region count: {num_regions}")
        picked = class_region_attn[torch.arange(bsz, device=targets.device), targets]
        maps = picked.reshape(bsz, 1, side, side)
        maps = F.interpolate(maps, size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
        return _normalize_tensor_map(maps.detach())

    pixel_gate = out.get("pixel_gate")
    if torch.is_tensor(pixel_gate) and pixel_gate.ndim == 3 and pixel_gate.shape[1] == height * width:
        maps = pixel_gate.squeeze(-1).reshape(pixel_gate.shape[0], height, width)
        return _normalize_tensor_map(maps.detach())

    raise KeyError("No usable fallback attention found: expected class_region_attn or pixel_gate")


def _teacher_saliency(
    teacher: torch.nn.Module,
    batch: Dict[str, Any],
    height: int,
    width: int,
    target_mode: str,
    fallback_to_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    labels = batch.get("y")
    if torch.is_tensor(labels):
        labels = labels.long()
    notes = "input_gradient"
    try:
        teacher.zero_grad(set_to_none=True)
        x = batch["x"].detach().clone().requires_grad_(True)
        grad_batch = dict(batch)
        grad_batch["x"] = x
        grad_batch["node_features"] = x
        out = teacher(grad_batch)
        logits = _extract_logits(out)
        targets = _target_classes(logits, labels, target_mode)
        score = logits[torch.arange(logits.shape[0], device=logits.device), targets].sum()
        score.backward()
        if x.grad is None:
            raise RuntimeError("node feature gradient is None")
        saliency = x.grad.detach().abs().sum(dim=-1).reshape(x.shape[0], height, width)
        saliency = _normalize_tensor_map(saliency)
        probs = torch.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        return saliency, pred, conf, targets, notes
    except Exception as exc:
        if not fallback_to_attention:
            raise
        with torch.no_grad():
            out = teacher(batch)
            logits = _extract_logits(out)
            targets = _target_classes(logits, labels, target_mode)
            saliency = _attention_to_pixel_map(out, targets, height=height, width=width)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
        notes = f"fallback_attention: {type(exc).__name__}: {exc}"
        return saliency, pred, conf, targets, notes


@torch.no_grad()
def _selected_motif_maps(motif_model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    out = motif_model(batch)
    maps = out["motif_assignment_maps"]
    if "selection_weights" in out:
        weights = out["selection_weights"]
    else:
        weights = torch.softmax(out["motif_scores"], dim=1)
    selected = (maps * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    return _normalize_tensor_map(selected)


def _border_center_mass(values: np.ndarray, border_width: int = 4, eps: float = 1e-8) -> tuple[float, float]:
    mask = np.zeros_like(values, dtype=bool)
    bw = int(border_width)
    if bw > 0:
        mask[:bw, :] = True
        mask[-bw:, :] = True
        mask[:, :bw] = True
        mask[:, -bw:] = True
    total = float(values.sum()) + eps
    border = float(values[mask].sum()) / total
    center = float(values[~mask].sum()) / total
    return border, center


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) + eps
    return float(np.dot(av, bv) / denom)


def _top_iou(a: np.ndarray, b: np.ndarray, q: float = 0.80) -> float:
    ta = float(np.quantile(a.reshape(-1), q))
    tb = float(np.quantile(b.reshape(-1), q))
    ma = a >= ta
    mb = b >= tb
    union = np.logical_or(ma, mb).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum() / union)


def _plot_heatmap(map_values: np.ndarray, out_path: Path, title: str, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.imshow(map_values, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_comparison(
    image: np.ndarray,
    saliency: np.ndarray,
    motif: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.8))
    panels = [
        ("image", image, "gray", None),
        ("teacher", saliency, "magma", None),
        ("D8M selected", motif, "viridis", None),
        ("teacher overlay", saliency, "magma", image),
        ("motif overlay", motif, "viridis", image),
    ]
    for ax, (name, heat, cmap, base) in zip(axes, panels):
        if base is not None:
            ax.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
            ax.imshow(heat, cmap=cmap, alpha=0.65, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_combined_grid(samples: list[dict[str, Any]], out_path: Path) -> None:
    if not samples:
        return
    rows = len(samples)
    fig, axes = plt.subplots(rows, 5, figsize=(12.5, 2.55 * rows))
    axes = np.asarray(axes).reshape(rows, 5)
    for row_idx, sample in enumerate(samples):
        panels = [
            ("image", sample["image"], "gray", None),
            ("teacher", sample["saliency"], "magma", None),
            ("D8M selected", sample["motif"], "viridis", None),
            ("teacher overlay", sample["saliency"], "magma", sample["image"]),
            ("motif overlay", sample["motif"], "viridis", sample["image"]),
        ]
        for ax, (name, heat, cmap, base) in zip(axes[row_idx], panels):
            if base is not None:
                ax.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
                ax.imshow(heat, cmap=cmap, alpha=0.65, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(name if row_idx == 0 else "", fontsize=8)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(f"{sample['sample_id']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    data_cfg = dict(cfg.get("data", {}) or {})
    teacher_cfg = dict(cfg.get("teacher", {}) or {})
    motif_cfg = dict(cfg.get("motif", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    paths_cfg = dict(cfg.get("paths", {}) or {})

    if args.graph_repo_path is not None:
        data_cfg["graph_repo_path"] = args.graph_repo_path
        paths_cfg["graph_repo_path"] = args.graph_repo_path
    elif data_cfg.get("graph_repo_path") is not None:
        paths_cfg["graph_repo_path"] = data_cfg["graph_repo_path"]

    if args.teacher_config is not None:
        teacher_cfg["model_config"] = args.teacher_config
    if args.teacher_checkpoint is not None:
        teacher_cfg["checkpoint"] = args.teacher_checkpoint
    if args.motif_config is not None:
        motif_cfg["config"] = args.motif_config
    if args.motif_checkpoint is not None:
        motif_cfg["checkpoint"] = args.motif_checkpoint
    if args.split is not None:
        data_cfg["split"] = args.split
    if args.max_samples is not None:
        data_cfg["max_samples"] = int(args.max_samples)
    if args.device is not None:
        cfg.setdefault("training", {})["device"] = args.device

    output_dir = output_cfg.get("dir") or paths_cfg.get("resolved_output_root") or "outputs/d8m_stage1h_teacher_saliency_audit"
    paths_cfg["resolved_output_root"] = output_dir

    cfg["data"] = data_cfg
    cfg["teacher"] = teacher_cfg
    cfg["motif"] = motif_cfg
    cfg["output"] = output_cfg
    cfg["paths"] = paths_cfg
    return cfg


def run_audit(config: Dict[str, Any], dry_run: bool = False) -> Path:
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(config=config)
    data_cfg = dict(config.get("data", {}) or {})
    teacher_cfg = dict(config.get("teacher", {}) or {})
    motif_cfg = dict(config.get("motif", {}) or {})
    output_dir = resolve_path(config.get("output", {}).get("dir") or config.get("paths", {}).get("resolved_output_root"))
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "d8m_stage1h_teacher_saliency_audit"
    out_dir = _ensure_dir(output_dir)

    motif_config_path = motif_cfg.get("config")
    motif_checkpoint = motif_cfg.get("checkpoint")
    if not motif_config_path or not motif_checkpoint:
        raise ValueError("motif.config and motif.checkpoint are required")
    motif_model, motif_config, motif_ckpt_path = _load_model_from_config(motif_config_path, motif_checkpoint, device)
    print(f"[Motif] loaded config={motif_config_path} checkpoint={motif_ckpt_path}")

    if dry_run:
        print("[DryRun] Motif model/checkpoint loaded. Teacher saliency was not run.")
        return out_dir

    teacher_config_path = teacher_cfg.get("model_config") or teacher_cfg.get("config")
    teacher_checkpoint = teacher_cfg.get("checkpoint")
    if not teacher_config_path or not teacher_checkpoint:
        raise ValueError(
            "Teacher config/checkpoint are required for saliency audit. "
            "Pass --teacher_config and --teacher_checkpoint, or set teacher.model_config and teacher.checkpoint."
        )
    teacher, teacher_config, teacher_ckpt_path = _load_model_from_config(teacher_config_path, teacher_checkpoint, device)
    print(f"[Teacher] loaded config={teacher_config_path} checkpoint={teacher_ckpt_path}")

    loader_config = {
        "paths": {"graph_repo_path": config.get("paths", {}).get("graph_repo_path", "artifacts/graph_repo")},
        "data": {
            "batch_size": int(data_cfg.get("batch_size", 4)),
            "num_workers": int(data_cfg.get("num_workers", 0)),
            "pin_memory": bool(data_cfg.get("pin_memory", False)),
            "persistent_workers": bool(data_cfg.get("persistent_workers", False)),
            "prefetch_factor": data_cfg.get("prefetch_factor"),
            "chunk_cache_size": int(data_cfg.get("chunk_cache_size", 8) or 0),
            "chunk_aware_shuffle": bool(data_cfg.get("chunk_aware_shuffle", False)),
        },
    }
    split = str(data_cfg.get("split", "val"))
    max_batches = data_cfg.get("max_batches", None)
    max_samples = int(data_cfg.get("max_samples", 8))
    loader = build_dataloader(loader_config, split=split, shuffle=False)

    height = int(motif_config.get("model", {}).get("height", motif_config.get("model", {}).get("image_size", 48)))
    width = int(motif_config.get("model", {}).get("width", motif_config.get("model", {}).get("image_size", 48)))
    target_mode = str(teacher_cfg.get("target", "ground_truth")).lower()
    saliency_method = str(teacher_cfg.get("saliency_method", "input_gradient")).lower()
    if saliency_method != "input_gradient":
        raise ValueError(f"Unsupported teacher.saliency_method={saliency_method!r}; only input_gradient is implemented")
    fallback = bool(teacher_cfg.get("fallback_to_attention", True))

    rows: list[dict[str, Any]] = []
    grid_samples: list[dict[str, Any]] = []
    saved = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if saved >= max_samples:
            break
        batch = move_to_device(batch, device)
        saliency, pred, conf, targets, notes = _teacher_saliency(
            teacher=teacher,
            batch=batch,
            height=height,
            width=width,
            target_mode=target_mode,
            fallback_to_attention=fallback,
        )
        motif_map = _selected_motif_maps(motif_model, batch)

        bsz = int(saliency.shape[0])
        for i in range(bsz):
            if saved >= max_samples:
                break
            graph_id = int(batch.get("graph_id", torch.arange(bsz, device=device))[i].detach().cpu())
            label = int(batch["y"][i].detach().cpu()) if torch.is_tensor(batch.get("y")) else -1
            sample_id = saved
            image = _normalize_np(batch["x"][i, :, 0].detach().float().cpu().numpy().reshape(height, width))
            s_map = saliency[i].detach().float().cpu().numpy()
            m_map = motif_map[i].detach().float().cpu().numpy()

            s_border, s_center = _border_center_mass(s_map)
            m_border, m_center = _border_center_mass(m_map)
            cosine = _cosine(s_map, m_map)
            iou20 = _top_iou(s_map, m_map, q=0.80)
            row = {
                "sample_id": sample_id,
                "graph_id": graph_id,
                "label": label,
                "teacher_pred": int(pred[i].detach().cpu()),
                "teacher_confidence": float(conf[i].detach().cpu()),
                "saliency_border_mass": s_border,
                "saliency_center_mass": s_center,
                "motif_border_mass": m_border,
                "motif_center_mass": m_center,
                "teacher_motif_cosine": cosine,
                "teacher_motif_iou_top20": iou20,
                "notes": notes,
            }
            rows.append(row)

            stem = f"sample_{sample_id:03d}"
            _plot_heatmap(s_map, out_dir / f"{stem}_teacher_saliency.png", title=f"{stem} teacher")
            _plot_heatmap(m_map, out_dir / f"{stem}_motif_selected.png", title=f"{stem} D8M selected", cmap="viridis")
            _plot_comparison(
                image=image,
                saliency=s_map,
                motif=m_map,
                out_path=out_dir / f"{stem}_comparison.png",
                title=f"{stem} id={graph_id} y={label} pred={row['teacher_pred']} conf={row['teacher_confidence']:.3f}",
            )
            grid_samples.append({"sample_id": sample_id, "image": image, "saliency": s_map, "motif": m_map})
            saved += 1

    with (out_dir / "saliency_audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _plot_combined_grid(grid_samples, out_dir / "combined_grid.png")
    print(f"[Output] samples={len(rows)} summary={out_dir / 'saliency_audit_summary.csv'}")
    print(f"[Output] figures={out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/d8m_stage1h_teacher_saliency_audit.yaml")
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--teacher_config", default=None)
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--motif_config", default=None)
    parser.add_argument("--motif_checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Load config and motif checkpoint only; skip teacher audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = _merge_config(args)
        run_audit(config, dry_run=bool(args.dry_run))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
