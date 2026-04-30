"""Visualize D6A soft part slots and part-to-part attention."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    apply_cli_overrides,
    build_dataloader,
    load_checkpoint_model,
    load_config,
    output_root_from_checkpoint,
    resolve_path,
)
from data.labels import EMOTION_NAMES  # noqa: E402
from training.trainer import move_to_device  # noqa: E402

CONFUSION_PAIRS = [
    (2, 4, "Fear", "Sad"),
    (2, 6, "Fear", "Neutral"),
    (2, 5, "Fear", "Surprise"),
    (4, 6, "Sad", "Neutral"),
    (0, 1, "Angry", "Disgust"),
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _plot_part_mask_grid(
    image: np.ndarray,
    masks: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    k = masks.shape[0]
    cols = int(math.ceil(math.sqrt(k + 1)))
    rows = int(math.ceil((k + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.1))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=8)
    axes[0].axis("off")
    for idx in range(k):
        ax = axes[idx + 1]
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(masks[idx], cmap="magma", alpha=0.65)
        ax.set_title(f"slot {idx:02d}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("slot")
    ax.set_ylabel("slot")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_class_slot_heatmap(matrix: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("slot")
    ax.set_ylabel("emotion")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(EMOTION_NAMES[: matrix.shape[0]])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_class_part_attention_csvs(avg_attn: np.ndarray, out_dir: Path, top_k: int = 5) -> None:
    attn = np.asarray(avg_attn, dtype=np.float64)
    eps = 1e-8
    entropy = -(attn * np.log(np.clip(attn, eps, None))).sum(axis=1)
    max_prob = attn.max(axis=1)

    with (out_dir / "class_part_attention_entropy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "entropy", "max_prob"])
        for idx, name in enumerate(EMOTION_NAMES[: attn.shape[0]]):
            writer.writerow([idx, name, float(entropy[idx]), float(max_prob[idx])])

    denom = np.linalg.norm(attn, axis=1, keepdims=True).clip(min=eps)
    normed = attn / denom
    sim = normed @ normed.T
    with (out_dir / "class_part_attention_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_a", "class_b", "cosine_similarity"])
        for i, name_i in enumerate(EMOTION_NAMES[: sim.shape[0]]):
            for j, name_j in enumerate(EMOTION_NAMES[: sim.shape[1]]):
                writer.writerow([name_i, name_j, float(sim[i, j])])

    with (out_dir / "top_slots_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "rank", "slot", "attention"])
        top_k = min(int(top_k), attn.shape[1])
        for idx, name in enumerate(EMOTION_NAMES[: attn.shape[0]]):
            top_slots = np.argsort(attn[idx])[-top_k:][::-1]
            for rank, slot in enumerate(top_slots, start=1):
                writer.writerow([idx, name, rank, int(slot), float(attn[idx, slot])])

    with (out_dir / "confusion_pair_attention_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_a_idx", "class_a", "class_b_idx", "class_b", "cosine_similarity"])
        for left, right, left_name, right_name in CONFUSION_PAIRS:
            if sim.shape[0] > max(left, right):
                writer.writerow([left, left_name, right, right_name, float(sim[left, right])])


def _make_border_mask_np(height: int, width: int, border_width: int = 3) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float64)
    bw = int(border_width)
    if bw > 0:
        mask[:bw, :] = 1.0
        mask[-bw:, :] = 1.0
        mask[:, :bw] = 1.0
        mask[:, -bw:] = 1.0
    return mask.reshape(-1)


def _save_class_motif_border_mass_csv(
    true_maps: np.ndarray,
    pred_maps: np.ndarray,
    out_dir: Path,
    height: int,
    width: int,
    border_width: int = 3,
) -> None:
    border_mask = _make_border_mask_np(height, width, border_width=border_width)
    rows = []
    for source, maps in (("true_class_avg", true_maps), ("pred_class_avg", pred_maps)):
        flat_maps = np.asarray(maps, dtype=np.float64).reshape(maps.shape[0], -1)
        for idx, name in enumerate(EMOTION_NAMES[: flat_maps.shape[0]]):
            motif = flat_maps[idx]
            total_mass = float(np.sum(motif))
            border_ratio = float(np.sum(motif * border_mask) / max(total_mass, 1e-8))
            rows.append([source, idx, name, border_ratio, total_mass])

    with (out_dir / "class_motif_border_mass_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "class_idx", "class_name", "border_mass_ratio", "motif_mass"])
        writer.writerows(rows)

    with (out_dir / "true_class_border_mass_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "border_mass_ratio", "motif_mass"])
        for source, idx, name, border_ratio, total_mass in rows:
            if source == "true_class_avg":
                writer.writerow([idx, name, border_ratio, total_mass])


def _plot_class_motif_grid(maps: np.ndarray, out_path: Path, title: str) -> None:
    cols = 4
    rows = int(math.ceil(maps.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    vmax = float(np.nanmax(maps)) if maps.size else 1.0
    vmax = max(vmax, 1e-8)
    for idx in range(maps.shape[0]):
        axes[idx].imshow(maps[idx], cmap="magma", vmin=0.0, vmax=vmax)
        axes[idx].set_title(EMOTION_NAMES[idx], fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_sample_class_motif(
    image: np.ndarray,
    true_map: np.ndarray,
    pred_map: np.ndarray,
    class_attn: np.ndarray,
    out_path: Path,
    title: str,
    true_label: int,
    pred_label: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(true_map, cmap="magma", alpha=0.68)
    axes[1].set_title(f"true {EMOTION_NAMES[true_label]}", fontsize=9)
    axes[2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(pred_map, cmap="magma", alpha=0.68)
    axes[2].set_title(f"pred {EMOTION_NAMES[pred_label]}", fontsize=9)
    top = np.argsort(class_attn[pred_label])[-6:][::-1]
    axes[3].bar(np.arange(len(top)), class_attn[pred_label, top])
    axes[3].set_xticks(np.arange(len(top)))
    axes[3].set_xticklabels([str(int(t)) for t in top])
    axes[3].set_title("pred top slots", fontsize=9)
    axes[3].set_xlabel("slot")
    for ax in axes[:3]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_avg_slots(avg_masks: np.ndarray, out_dir: Path) -> None:
    k = avg_masks.shape[0]
    for idx in range(k):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(avg_masks[idx], cmap="magma")
        ax.set_title(f"avg slot {idx:02d}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_slot_{idx:02d}.png", dpi=160)
        plt.close(fig)

    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx in range(k):
        axes[idx].imshow(avg_masks[idx], cmap="magma")
        axes[idx].set_title(f"slot {idx:02d}", fontsize=8)
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_slot_grid.png", dpi=160)
    plt.close(fig)


def _cosine_similarity(flat_masks: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(flat_masks, axis=1, keepdims=True).clip(min=1e-8)
    normed = flat_masks / denom
    return normed @ normed.T


def _save_slot_area(area_values: list[np.ndarray], out_dir: Path) -> None:
    if not area_values:
        return
    areas = np.concatenate(area_values, axis=0)
    mean = areas.mean(axis=0)
    std = areas.std(axis=0)
    with (out_dir / "slot_area.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot", "mean", "std"])
        for idx, (m, s) in enumerate(zip(mean, std)):
            writer.writerow([idx, float(m), float(s)])
    with (out_dir / "slot_area.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(mean)), mean, yerr=std, capsize=2)
    ax.set_xlabel("slot")
    ax.set_ylabel("area mass")
    ax.set_title("Slot area distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "slot_area.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_visualize(
    config,
    checkpoint=None,
    split: str = "test",
    max_samples: int = 16,
    max_batches: int | None = None,
) -> None:
    paths = config.get("paths", {})
    if checkpoint is not None:
        output_root = output_root_from_checkpoint(checkpoint) or resolve_path(
            paths.get("resolved_output_root") or paths.get("output_root", "outputs")
        )
    else:
        output_root = resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"

    model, device, _ = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split=split, shuffle=False)
    graph_cfg = config.get("graph", {})
    height = int(graph_cfg.get("height", 48))
    width = int(graph_cfg.get("width", 48))

    fig_root = output_root / "figures"
    masks_dir = _ensure_dir(fig_root / "d6_part_masks")
    attn_dir = _ensure_dir(fig_root / "d6_part_attention")
    summary_dir = _ensure_dir(fig_root / "d6_slot_summary")
    class_attn_dir = _ensure_dir(fig_root / "d6_class_part_attention")
    class_motif_dir = _ensure_dir(fig_root / "d6_class_motif_maps")

    mask_sum = None
    sample_count = 0
    saved_samples = 0
    saved_correct = 0
    saved_wrong = 0
    saved_attn = 0
    saved_class_motif = 0
    area_values: list[np.ndarray] = []
    class_attn_sum = None
    class_attn_count = 0
    class_attn_true_sum = None
    class_attn_pred_sum = None
    class_pixel_true_sum = None
    class_pixel_pred_sum = None
    class_true_count = np.zeros(len(EMOTION_NAMES), dtype=np.float64)
    class_pred_count = np.zeros(len(EMOTION_NAMES), dtype=np.float64)

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        probs = torch.softmax(logits.float(), dim=1)
        pred = logits.argmax(dim=1)
        masks = out["part_masks"].detach().float()
        area_values.append(_to_numpy(out["slot_area"]))
        class_part_attn = out.get("class_part_attn")
        class_pixel_attn = None
        if class_part_attn is not None:
            class_part_attn = class_part_attn.detach().float()
            class_pixel_attn = torch.einsum("bck,bkn->bcn", class_part_attn, masks)
            cpa_np = _to_numpy(class_part_attn)
            if class_attn_sum is None:
                class_attn_sum = cpa_np.sum(axis=0)
                class_attn_true_sum = np.zeros_like(class_attn_sum)
                class_attn_pred_sum = np.zeros_like(class_attn_sum)
                class_pixel_true_sum = np.zeros((cpa_np.shape[1], height * width), dtype=np.float64)
                class_pixel_pred_sum = np.zeros((cpa_np.shape[1], height * width), dtype=np.float64)
            else:
                class_attn_sum += cpa_np.sum(axis=0)
            class_attn_count += int(cpa_np.shape[0])

        batch_mask_sum = masks.sum(dim=0)
        mask_sum = batch_mask_sum if mask_sum is None else mask_sum + batch_mask_sum
        sample_count += int(masks.shape[0])

        for i in range(masks.shape[0]):
            image = _to_numpy(batch["x"][i, :, 0]).reshape(height, width)
            item_masks = _to_numpy(masks[i]).reshape(masks.shape[1], height, width)
            y_true = int(batch["y"][i].item())
            y_pred = int(pred[i].item())
            confidence = float(probs[i, y_pred].item())
            graph_id = int(batch["graph_id"][i].item())
            title = (
                f"id={graph_id} true={EMOTION_NAMES[y_true]} "
                f"pred={EMOTION_NAMES[y_pred]} conf={confidence:.2f}"
            )
            if saved_samples < int(max_samples):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"sample_{saved_samples:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_samples += 1
            if y_true == y_pred and saved_correct < int(max_samples // 2):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"correct_{saved_correct:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_correct += 1
            elif y_true != y_pred and saved_wrong < int(max_samples // 2):
                _plot_part_mask_grid(
                    image,
                    item_masks,
                    masks_dir / f"wrong_{saved_wrong:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title,
                )
                saved_wrong += 1

            part_attn = out.get("part_attn")
            if part_attn is not None and saved_attn < int(max_samples):
                attn = part_attn[i].detach().float()
                if attn.ndim == 3:
                    attn = attn.mean(dim=0)
                _plot_heatmap(
                    _to_numpy(attn),
                    attn_dir / f"part_attn_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                    title="part-to-part attention",
                    cmap="Blues",
                )
                saved_attn += 1

            if class_part_attn is not None and class_pixel_attn is not None:
                item_class_attn = _to_numpy(class_part_attn[i])
                item_class_pixel = _to_numpy(class_pixel_attn[i])
                if class_attn_true_sum is not None and class_attn_pred_sum is not None:
                    class_attn_true_sum[y_true] += item_class_attn[y_true]
                    class_attn_pred_sum[y_pred] += item_class_attn[y_pred]
                    class_pixel_true_sum[y_true] += item_class_pixel[y_true]
                    class_pixel_pred_sum[y_pred] += item_class_pixel[y_pred]
                    class_true_count[y_true] += 1.0
                    class_pred_count[y_pred] += 1.0
                if saved_class_motif < int(max_samples):
                    _plot_sample_class_motif(
                        image=image,
                        true_map=item_class_pixel[y_true].reshape(height, width),
                        pred_map=item_class_pixel[y_pred].reshape(height, width),
                        class_attn=item_class_attn,
                        out_path=class_motif_dir / f"sample_class_motif_{saved_class_motif:03d}_id_{graph_id}_true_{y_true}_pred_{y_pred}.png",
                        title=title,
                        true_label=y_true,
                        pred_label=y_pred,
                    )
                    _plot_class_slot_heatmap(
                        item_class_attn,
                        class_attn_dir / f"class_part_attn_per_sample_{saved_class_motif:03d}_id_{graph_id}.png",
                        title=title,
                    )
                    saved_class_motif += 1

    if mask_sum is None or sample_count == 0:
        print("No samples were visualized.")
        return

    avg_masks = _to_numpy(mask_sum / float(sample_count)).reshape(-1, height, width)
    _plot_avg_slots(avg_masks, summary_dir)
    sim = _cosine_similarity(avg_masks.reshape(avg_masks.shape[0], -1))
    _plot_heatmap(sim, summary_dir / "slot_similarity.png", "Average slot mask cosine similarity", cmap="magma")
    _save_slot_area(area_values, summary_dir)

    if class_attn_sum is not None and class_attn_count > 0:
        avg_class_attn = class_attn_sum / float(class_attn_count)
        _plot_class_slot_heatmap(
            avg_class_attn,
            class_attn_dir / "class_part_attn_grid.png",
            "Average class-to-part attention",
        )
        _save_class_part_attention_csvs(avg_class_attn, class_attn_dir)
        true_den = np.maximum(class_true_count[:, None], 1.0)
        pred_den = np.maximum(class_pred_count[:, None], 1.0)
        _plot_class_slot_heatmap(
            class_attn_true_sum / true_den,
            class_attn_dir / "class_part_attn_avg_by_true_class.png",
            "Class-to-part attention by true class",
        )
        _plot_class_slot_heatmap(
            class_attn_pred_sum / pred_den,
            class_attn_dir / "class_part_attn_avg_by_pred_class.png",
            "Class-to-part attention by predicted class",
        )
        _plot_class_motif_grid(
            (class_pixel_true_sum / np.maximum(class_true_count[:, None], 1.0)).reshape(-1, height, width),
            class_motif_dir / "class_pixel_motif_trueclass_avg.png",
            "Class pixel motif by true class",
        )
        _plot_class_motif_grid(
            (class_pixel_pred_sum / np.maximum(class_pred_count[:, None], 1.0)).reshape(-1, height, width),
            class_motif_dir / "class_pixel_motif_predclass_avg.png",
            "Class pixel motif by predicted class",
        )
        true_class_maps = (class_pixel_true_sum / np.maximum(class_true_count[:, None], 1.0)).reshape(-1, height, width)
        pred_class_maps = (class_pixel_pred_sum / np.maximum(class_pred_count[:, None], 1.0)).reshape(-1, height, width)
        border_width = int(config.get("loss", {}).get("border_width", graph_cfg.get("border_width", 3)))
        _save_class_motif_border_mass_csv(
            true_class_maps,
            pred_class_maps,
            class_motif_dir,
            height=height,
            width=width,
            border_width=border_width,
        )

    print(f"D6 part masks: {masks_dir}")
    print(f"D6 part attention: {attn_dir}")
    print(f"D6 slot summary: {summary_dir}")
    if class_attn_sum is not None:
        print(f"D6 class-part attention: {class_attn_dir}")
        print(f"D6 class motif maps: {class_motif_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d6a_slot_pixel_part_graph_motif.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_visualize(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
