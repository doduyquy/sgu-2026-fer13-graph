"""Visualize D5B fixed motif node priors."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import resolve_path
from data.labels import EMOTION_NAMES


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_prior_figures(
    node_prior: torch.Tensor,
    output_dir: str | Path,
    class_names: Sequence[str] | None = None,
    height: int = 48,
    width: int = 48,
) -> None:
    prior = torch.as_tensor(node_prior, dtype=torch.float32).cpu()
    if prior.ndim != 2:
        raise ValueError(f"node_prior must be [C, N], got {tuple(prior.shape)}")
    if prior.shape[1] != height * width:
        raise ValueError(f"Expected {height * width} nodes, got {prior.shape[1]}")
    class_names = list(class_names or EMOTION_NAMES)
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for class_idx in range(prior.shape[0]):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        heat = prior[class_idx].reshape(height, width).numpy()
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        im = ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"{class_idx} {class_name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"class_node_prior_{class_idx}_{_slug(class_name)}.png", dpi=160)
        plt.close(fig)

    cols = min(4, prior.shape[0])
    rows = (prior.shape[0] + cols - 1) // cols
    fig, _ = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = fig.axes
    for ax in axes:
        ax.axis("off")
    for class_idx in range(prior.shape[0]):
        ax = axes[class_idx]
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        heat = prior[class_idx].reshape(height, width).numpy()
        ax.imshow(heat, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"{class_idx} {class_name}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_node_prior_grid.png", dpi=180)
    plt.close(fig)


def run_visualize(prior_path: str | Path, output_dir: str | Path | None = None) -> None:
    prior_path = resolve_path(prior_path)
    payload = _torch_load(prior_path)
    if "node_prior" not in payload:
        raise KeyError(f"Missing node_prior in {prior_path}")
    out_dir = resolve_path(output_dir) if output_dir is not None else prior_path.parent
    save_prior_figures(
        payload["node_prior"],
        output_dir=out_dir,
        class_names=payload.get("class_names", EMOTION_NAMES),
    )
    print(f"Saved prior figures to {Path(out_dir) / 'figures'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_path", default="artifacts/d5b_motif_prior/node_prior.pt")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    run_visualize(args.prior_path, args.output_dir)


if __name__ == "__main__":
    main()
