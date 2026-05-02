# D8A Graph-Swin Pre-Part D6B

## Why After D7

D7 established a strong graph-only ceiling with Graph-Swin region context:

- Best single-model macro F1: `d7a_graph_swin_region_transformer_seed44`, macro F1 `0.5883`.
- Best single-model accuracy: `d7a_graph_swin_region_transformer_window4`, accuracy `0.6105`.
- Official graph-only ensemble: accuracy `0.6436`, macro F1 `0.6292`.

D8A-PrePart is the next research step because D6B remains more interpretable at the pixel/part motif level, but its part slots must assign from noisy per-pixel embeddings. D8A tests whether local Graph-Swin context can improve those embeddings before part assignment.

## Not D7C Fusion

D8A does not add a parallel Graph-Swin classifier and does not fuse Graph-Swin logits with D6B logits. Graph-Swin is used only as a pre-part context encoder:

`h_pixel -> h_context -> enhanced_h_pixel -> D6B part branch -> logits`

The only classifier is the D6B class-to-part path.

## Difference From D6B

D6B assigns parts directly from pixel encoder embeddings. D8A inserts a local/window context map at pixel resolution:

`enhanced_h_pixel = h_pixel + alpha * h_context`

The part masks, part features, part-to-part attention, class-to-part attention, and class pixel motifs are still D6B-style outputs.

## Architecture

- Shared edge-aware pixel encoder: node dim `7`, edge dim `5`, hidden dim `64`.
- Graph-Swin pre-part context:
  - image size `48x48`
  - regular windows and shifted windows
  - main config uses `window_size=6`, `shift_size=3`
  - window summaries are broadcast back to pixels
  - regular and shifted context are merged at pixel resolution
- Learnable scalar `context_alpha`, initialized to `0.1`.
- D6B motif branch:
  - `16` part slots
  - softmax over slots per pixel
  - weighted pooling to part features
  - part-to-part self-attention
  - class-to-part attention classifier

Registered model names:

- `graph_swin_prepart_d6b_d8a`
- `d8a_graph_swin_prepart_d6b`

## Loss

The initial D8A loss stays close to D6B:

- cross entropy
- slot diversity
- slot balance
- border penalty
- optional `context_alpha_l2`, default `0.0`

No SupCon, class attention separation, class-attended border objective, face-aware gate, D7C fusion, or input-v2 features are used.

Loss name:

- `d8a_prepart_motif_loss`

## Configs

Main config:

- `configs/experiments/d8a_graph_swin_prepart_d6b.yaml`
- window6, shift3, alpha init `0.1`

Optional follow-up configs:

- `configs/experiments/d8a_graph_swin_prepart_d6b_alpha020.yaml`
- `configs/experiments/d8a_graph_swin_prepart_d6b_window4.yaml`

The main first run should be window6 alpha0.1. Window4 is for after the main D8A run has a signal.

## Smoke Commands

```bash
python -m py_compile models/graph_swin_prepart_d6b.py scripts/debug_d8a_forward.py scripts/visualize_d8a_prepart.py
```

```bash
python scripts/debug_d8a_forward.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml
```

```bash
python scripts/train_d5a.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml --checkpoint output/d8a_graph_swin_prepart_d6b/checkpoints/best.pth --max_test_batches 1
```

```bash
python scripts/visualize_d8a_prepart.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml --checkpoint output/d8a_graph_swin_prepart_d6b/checkpoints/best.pth --max_samples 4
```

Backward compatibility checks:

```bash
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml
python scripts/debug_d7_forward.py --config configs/experiments/d7a_graph_swin_region_transformer_window4.yaml
python scripts/debug_d8a_forward.py --config configs/experiments/d6b_class_part_graph_motif_border075.yaml
```

## Full Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml
```

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml --checkpoint output/d8a_graph_swin_prepart_d6b/checkpoints/best.pth
```

```bash
python scripts/visualize_d8a_prepart.py --config configs/experiments/d8a_graph_swin_prepart_d6b.yaml --checkpoint output/d8a_graph_swin_prepart_d6b/checkpoints/best.pth --max_samples 32
```

## Success Criteria

Very good:

- single-model macro F1 above `0.5883`, or accuracy above `0.6105`

Good:

- macro F1 at least `0.57`
- accuracy at least `0.59`
- masks and class motifs cleaner than D6B

Research signal:

- clearly beats D6B border075
- motif interpretability improves
- may still trail D7 region transformer

Failure:

- does not beat D6B
- slots collapse heavily
- alpha goes to zero or context maps are uninformative
- attention focuses mostly on border/background/hair

## Analysis Focus

After a full run, inspect Fear, Sad, Neutral, Disgust, slot collapse, border focus, learned alpha, and whether context norm maps concentrate around face/mouth/eyes.

## Next Directions

If D8A has signal:

- run alpha0.2 and window4 variants
- compare learned alpha and context maps across classes
- consider a light window attention variant

If D8A has no signal:

- check whether alpha collapses
- inspect context-to-pixel norm ratio
- compare slot similarity against D6B
- keep D7 official results untouched and treat D8A as an interpretability branch experiment
