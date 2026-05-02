# D7A Region Transformer Next Experiments Plan

## Current Main

`d7a_graph_swin_region_transformer` is the current D7A main because it is the only Stage 2 variant that beats the D7A baseline on all three headline metrics:

- accuracy: `0.6002`
- macro F1: `0.5794`
- weighted F1: `0.5969`
- best validation macro F1: `0.576968` at epoch `119`

The model keeps the full pixel graph input (`node_dim=7`, `edge_dim=5`) and improves the Graph-Swin branch by adding region-level self-attention before class-to-region attention.

## Why Seed Repeat

The champion result must be checked for robustness. Two seed repeats isolate whether the region transformer improvement is stable or mostly seed variance.

New configs:

- `configs/experiments/d7a_graph_swin_region_transformer_seed43.yaml`
- `configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml`

Decision rule for seed repeat:

- mean macro F1 should be at least `0.57`
- mean accuracy should be at least `0.59`
- no severe collapse in Fear, Sad, or Neutral
- at least one seed near or above the current champion macro F1 is a strong positive signal

Baseline references:

- D7A seed43: accuracy `0.5759`, macro F1 `0.5498`
- D7A seed44: accuracy `0.5896`, macro F1 `0.5694`

## Why Layers2

`d7a_graph_swin_region_transformer_layers2` tests whether a second region self-attention layer improves cross-region relations. This is most relevant for Fear, Sad, and Neutral, where evidence can depend on eye, eyebrow, and mouth relationships rather than one local patch.

New config:

- `configs/experiments/d7a_graph_swin_region_transformer_layers2.yaml`

Risk:

- The champion already has high region token norm (`7.3285`), so an extra layer may overfit or amplify token magnitudes.

Decision rule:

- macro F1 should exceed `0.5794`
- accuracy should be at least `0.6002`, or macro F1 must improve clearly with only minor accuracy loss
- Fear at least `0.4136`
- Sad at least `0.445`
- Neutral at least `0.58`
- no large train/val gap increase
- no unusual AMP skip behavior

## Why Region Transformer + Window4

`d7a_graph_swin_region_transformer_window4` combines smaller Graph-Swin windows with region relation modeling. `window_size=4` gives a `12x12` window grid, and 2x2 region merge gives `6x6 = 36` region tokens. This may preserve small expression details while letting region self-attention model the larger face layout.

New config:

- `configs/experiments/d7a_graph_swin_region_transformer_window4.yaml`

Technical checks:

- no hard-coded `R=16` assumptions
- `region_tokens` shape should be `[B,36,64]`
- `class_region_attn` shape should be `[B,7,36]`
- visualizations must handle a `6x6` region grid
- region entropy should be interpreted against `log(36)`, not compared directly with `R=16`

Decision rule:

- macro F1 should exceed `0.5794`, or stay similar while Fear/Disgust improve clearly
- Sad and Neutral should not drop materially
- Happy should not become over-predicted
- Disgust should stay at least `0.55`
- Surprise should stay at least `0.72`

## Configs Created

- `configs/experiments/d7a_graph_swin_region_transformer_seed43.yaml`
- `configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml`
- `configs/experiments/d7a_graph_swin_region_transformer_layers2.yaml`
- `configs/experiments/d7a_graph_swin_region_transformer_window4.yaml`

All configs keep:

- full pixel graph repo
- `node_dim=7`
- `edge_dim=5`
- champion loss
- AdamW `lr=0.0005`, `weight_decay=0.0001`
- ReduceLROnPlateau on `val_macro_f1`
- `epochs=120`
- `early_stopping_patience=25`
- `class_head_type=attn_only`

## Smoke Commands

Config/debug forward:

```bash
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_seed43.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_layers2.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer.yaml
```

Train smoke:

```bash
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_layers2.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

Eval smoke for window4:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml --checkpoint output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth --max_test_batches 1
```

Visualize smoke for window4:

```bash
python scripts/visualize_d7.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml --checkpoint output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth --max_samples 4
```

## Full Run Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_seed43.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_layers2.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config <config> --checkpoint output/<experiment>/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d7.py --config <config> --checkpoint output/<experiment>/checkpoints/best.pth --max_samples 32
```

## Overall Champion Rule

A new config can replace the current champion only if it satisfies most of:

- macro F1 above `0.5794`
- accuracy at least `0.6002`, or a clear macro F1 gain with only minor accuracy loss
- weighted F1 at least `0.5969`, or no large drop
- Fear at least `0.4136`
- Sad at least `0.445`
- Neutral at least `0.58`
- Disgust at least `0.55`
- Surprise at least `0.72`
- prediction counts do not collapse
- Fear/Sad/Neutral confusion does not become clearly worse

Seed repeats are not required to beat the champion individually. They mainly decide whether the region transformer should remain the robust main line.
