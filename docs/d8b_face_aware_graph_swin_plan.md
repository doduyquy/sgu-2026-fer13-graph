# D8B Face-Aware Graph-Swin Plan

## Why D8B after D8A

D8A-PrePart showed that the context branch learns real signal, but the D6B-style pixel-to-part slot head remains the bottleneck. Its validation diagnostics had strong context usage, while final test accuracy and macro F1 stayed far below the D7 region-transformer line. The next step should therefore stay on the D7 Graph-Swin region-transformer path and avoid forcing the representation through part slots.

## Difference from D8A

D8A injects Graph-Swin context before a D6B part-slot motif head. D8B removes the part branch entirely. It keeps pixel-level graph encoding and the D7 window/region hierarchy, then adds soft face/expression gates directly inside that hierarchy.

## Difference from D7

D7 uses:

pixel graph -> pixel encoder -> window tokens -> region tokens -> region transformer -> class-region attention -> logits

D8B keeps that backbone, but adds:

- pixel_gate after the pixel encoder
- gate-weighted window tokenization
- window_gate before region merging
- region_gate before the region transformer
- light pixel-gate regularization
- gate and attention diagnostics for visualization

No hard face detector, landmarks, CNN backbone, D6B slots, D8A prepart branch, D7C fusion, D8C motifs, or node_dim changes are introduced.

## Architecture

Model name aliases:

- `face_aware_graph_swin_d8b`
- `d8b_face_aware_graph_swin`

Main config:

- `configs/experiments/d8b_face_aware_graph_swin.yaml`

Backbone settings:

- `window_size: 6`
- `shift_size: 3`
- `region_merge: true`
- `use_region_transformer: true`
- `region_layers: 1`
- `region_heads: 4`
- `class_head_type: attn_only`

Window4 follow-up config:

- `configs/experiments/d8b_face_aware_graph_swin_window4.yaml`

## Gate Design

Pixel gate:

`pixel_gate = sigmoid(MLP([h_pixel, xy, node_features]))`

Shape: `[B, 2304, 1]`

Window tokenization:

`window_token = sum(pixel_gate_i * h_pixel_i) / (sum(pixel_gate_i) + eps)`

The same weighted pooling is applied to regular and shifted windows.

Window gate:

`window_tokens = window_tokens * (1 + beta_window * window_gate)`

Region gate:

`region_tokens = region_tokens * (1 + beta_region * region_gate)`

Both betas default to `0.5` and are non-learnable in the first D8B config.

## Loss

Loss name:

- `d8b_face_aware_loss`

Total:

`CE + lambda_area * L_area + lambda_border * L_border + lambda_smooth * L_smooth`

Defaults:

- `pixel_gate_area_weight: 0.01`
- `pixel_gate_area_target: 0.55`
- `pixel_gate_border_weight: 0.01`
- `pixel_gate_border_width: 3`
- `pixel_gate_smooth_weight: 0.001`

## Configs

- `d8b_face_aware_graph_swin.yaml`: first run
- `d8b_face_aware_graph_swin_border020.yaml`: stronger border penalty
- `d8b_face_aware_graph_swin_area045.yaml`: lower target gate area
- `d8b_face_aware_graph_swin_window4.yaml`: D7 window4-style follow-up

## Smoke Commands

```bash
python -m py_compile models/face_aware_graph_swin_d8b.py scripts/debug_d8b_forward.py scripts/visualize_d8b_face_aware.py
python scripts/debug_d8b_forward.py --config configs/experiments/d8b_face_aware_graph_swin.yaml
python scripts/train_d5a.py --config configs/experiments/d8b_face_aware_graph_swin.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
python scripts/evaluate_d5a.py --config configs/experiments/d8b_face_aware_graph_swin.yaml --checkpoint output/d8b_face_aware_graph_swin/checkpoints/best.pth --max_test_batches 1
python scripts/visualize_d8b_face_aware.py --config configs/experiments/d8b_face_aware_graph_swin.yaml --checkpoint output/d8b_face_aware_graph_swin/checkpoints/best.pth --max_samples 4
```

## Full Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d8b_face_aware_graph_swin.yaml
python scripts/evaluate_d5a.py --config configs/experiments/d8b_face_aware_graph_swin.yaml --checkpoint output/d8b_face_aware_graph_swin/checkpoints/best.pth
python scripts/visualize_d8b_face_aware.py --config configs/experiments/d8b_face_aware_graph_swin.yaml --checkpoint output/d8b_face_aware_graph_swin/checkpoints/best.pth --max_samples 32
```

## Success Criteria

Very good:

- single-model macro F1 > 0.5883, or accuracy > 0.6105

Good:

- macro F1 >= 0.58
- accuracy >= 0.60
- gate maps reduce border/background visibly

Research signal:

- does not beat D7 seed44, but clearly beats D8A
- gate maps are meaningful
- Fear/Sad/Neutral or Disgust improves

Failure:

- performance close to D8A
- gate collapse near all-zero or all-one
- gate prefers border/hair/background
- region gate or class-region attention has no usable signal

## Next Directions

If D8B works, try window4, stronger/lower area targets, or a D8C region motif layer on top of gated region tokens. If D8B does not work, inspect whether the gate is too weak, too border-biased, or fighting the D7 region transformer; then adjust regularization before adding new branches.
