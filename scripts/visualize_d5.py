"""Generate D5 class gate and attention heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    apply_cli_overrides,
    build_dataloader,
    load_checkpoint_model,
    load_config,
    resolve_path,
)

from visualization.visualize_d5_motifs import (
    save_class_gate_heatmaps,
    save_sample_attention_maps,
)


def run_visualize(config, checkpoint=None, split: str = "test", max_samples: int = 16):
    paths = config.get("paths", {})
    output_root = resolve_path(paths.get("output_root", "outputs"))
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"
    model, device, _ = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split=split, shuffle=False)
    gate_dir = output_root / "figures" / "d5a_class_gates"
    attn_dir = output_root / "figures" / "d5a_attention"
    graph_cfg = config.get("graph", {})
    height = int(graph_cfg.get("height", 48))
    width = int(graph_cfg.get("width", 48))
    save_class_gate_heatmaps(model, gate_dir, height=height, width=width)
    save_sample_attention_maps(
        model=model,
        loader=loader,
        device=device,
        out_dir=attn_dir,
        max_samples=max_samples,
        height=height,
        width=width,
        include_edges=True,
    )
    print(f"Class gates: {gate_dir}")
    print(f"Attention maps: {attn_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_visualize(config, checkpoint=args.checkpoint, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
