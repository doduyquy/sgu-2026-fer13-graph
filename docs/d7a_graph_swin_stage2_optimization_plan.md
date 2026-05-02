# D7A Graph-Swin Stage 2 Optimization Plan

## Scope

Stage 2 optimizes the D7A Graph-Swin standalone path before adding more fusion or hybrid complexity. The current main result is D7A standalone with accuracy 0.5935, macro F1 0.5699, weighted F1 0.5874, and best epoch 96.

D7C gated fusion is not continued in this stage because the first run underperformed D7A standalone: accuracy 0.5826, macro F1 0.5677, weighted F1 0.5762. The near-term signal is therefore stronger for improving Graph-Swin itself than for widening fusion with the D6 branch.

This stage does not touch D7C/fusion, D6B, node_dim=12 variants, D6C losses, face gates, D8, D9, hard masks, entropy sharpening, or dense full-pixel attention.

## Baseline

Baseline config: `configs/experiments/d7a_graph_swin_standalone.yaml`

Key settings:

- `mode: swin_only`
- `node_dim: 7`
- `edge_dim: 5`
- `hidden_dim: 64`
- `window_size: 6`
- `shift_size: 3`
- `use_window_mha: false`
- `region_merge: true`
- `epochs: 120`
- `early_stopping_patience: 25`
- `lr: 0.0005`
- `weight_decay: 0.0001`
- `batch_size: 32`

The baseline file is kept unchanged.

## Created Configs

1. `configs/experiments/d7a_graph_swin_window_mha.yaml`

   Enables local self-attention inside each 6x6 window with `use_window_mha: true`. This tests whether richer local interaction improves difficult expressions without changing the standalone branch.

2. `configs/experiments/d7a_graph_swin_window4.yaml`

   Uses `window_size: 4`, `shift_size: 2`, producing 36 merged region tokens. This tests whether smaller windows retain fine mouth and eye-corner details.

3. `configs/experiments/d7a_graph_swin_window8.yaml`

   Uses `window_size: 8`, `shift_size: 4`, producing 9 merged region tokens. This tests whether larger windows reduce noise and improve stable classes such as Happy and Neutral.

4. `configs/experiments/d7a_graph_swin_attn_plus_mean.yaml`

   Adds `class_head_type: attn_plus_mean`, where each class representation combines class-region attention output with a global mean region summary. This tests whether the head becomes more stable when attention is close to uniform.

5. `configs/experiments/d7a_graph_swin_region_transformer.yaml`

   Adds one region-level Transformer encoder layer before the class-region head with `use_region_transformer: true`, `region_layers: 1`, and `region_heads: 4`. This tests whether region tokens benefit from eye-mouth and brow-mouth context before classification.

## Smoke Commands

Debug forward:

```bash
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_window_mha.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_window4.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_window8.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_attn_plus_mean.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer.yaml
```

Train smoke:

```bash
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_window4.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

Eval smoke:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d7a_graph_swin_window4.yaml --checkpoint output/d7a_graph_swin_window4/checkpoints/best.pth --max_test_batches 1
```

Visualize smoke:

```bash
python scripts/visualize_d7.py --config configs/experiments/d7a_graph_swin_window4.yaml --checkpoint output/d7a_graph_swin_window4/checkpoints/best.pth --max_samples 4
```

## Full Run Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_window_mha.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_window4.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_window8.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_attn_plus_mean.yaml
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_region_transformer.yaml
```

Evaluate after each run:

```bash
python scripts/evaluate_d5a.py --config <config> --checkpoint output/<experiment>/checkpoints/best.pth
```

Visualize after each run:

```bash
python scripts/visualize_d7.py --config <config> --checkpoint output/<experiment>/checkpoints/best.pth --max_samples 32
```

## Success Criteria

A config wins if it improves macro F1 above 0.5699 while keeping accuracy and weighted F1 near or above the D7A baseline. It should also avoid prediction collapse and keep weak-class behavior healthy: Fear F1 at least 0.40, Sad F1 at least 0.44, Neutral F1 at least 0.56, and Disgust F1 at least 0.55.

If a config does not win overall but improves Fear, Disgust, or Sad, record it as useful evidence for a later ensemble or D8 stage.

Also check that attention/region maps do not visibly stick to borders, training has no NaN/Inf, and AMP skip behavior is not abnormal.

## Decision Rule

Promote the best macro-F1 config if accuracy and weighted F1 remain close to baseline. If two configs are close, prefer the one with healthier weak-class F1 and less skewed `pred_count`. If all variants regress, keep the original D7A standalone baseline as the main path.

