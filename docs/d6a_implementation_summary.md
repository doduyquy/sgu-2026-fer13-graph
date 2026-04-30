# D6A Implementation Summary

## Why D6A

D5B-2 showed that a flat class-level pixel motif mask `[7, 2304]` is too brittle for FER-2013. The learned masks were often small and flat, drifted toward background, hair, or image borders, and were not robust to pose, crop shifts, or off-center faces. Disgust remained unused across the D5 variants, and the light prior regularization in D5B-2B did not fix the motif behavior.

D6A tests a cleaner research question:

> Does replacing a flat pixel-level motif with a hierarchical soft part motif improve classification and produce more stable, interpretable motifs?

## Core Idea

D6A keeps the full pixel graph and does not use landmarks, face detectors, hard-coded facial parts, or hard top-K subgraphs. It learns soft part slots directly from the pixel graph:

```text
full pixel graph
-> edge-aware pixel encoder
-> pixel-to-part soft assignment
-> K learned soft parts
-> part-to-part self-attention
-> image representation
-> emotion classifier
```

The motif is now hierarchical:

```text
pixel -> soft part -> part relation -> emotion
```

## Architecture

Main class:

```text
models/slot_pixel_part_graph_motif.py
SlotPixelPartGraphMotif
```

Forward accepts either the existing trainer batch dict or direct tensors:

```text
x:          [B, 2304, 7]
edge_index: [2, 17860]
edge_attr:  [B, 17860, 5]
node_mask:  [B, 2304], optional
```

Outputs:

```text
logits:           [B, 7]
pixel_embeddings: [B, 2304, H]
part_masks:       [B, K, 2304]
part_features:    [B, K, H]
part_context:     [B, K, H]
part_attn:        [B, heads, K, K]
slot_area:        [B, K]
border_mass:      [B, K]
```

### Pixel Encoder

Node features are projected by:

```text
Linear(7, H) -> LayerNorm -> GELU -> Dropout
```

Then one or more edge-aware message passing layers aggregate neighbor messages over the fixed pixel graph. Edge attributes pass through an MLP gate, and messages are averaged into destination pixels with `index_add_`, avoiding a dependency on `torch_scatter`.

### Soft Part Slots

D6A has learnable part queries `[K, H]`. Pixel-to-part assignment uses query-key similarity and softmax over slots:

```text
part_masks = softmax(assign_logits, dim=1)
```

This creates competition between slots per pixel. Pooling weights are normalized over pixels per slot:

```text
pool_weights = part_masks / sum_pixels(part_masks)
part_features = pool_weights @ pixel_values
```

### Position and Relations

Soft slot centers are computed from the fixed 48x48 coordinate grid:

```text
part_centers: [B, K, 2]
```

A small positional MLP is added to part features, then part-to-part relationships are learned with `nn.MultiheadAttention` using `batch_first=True`.

### Classifier

D6A uses mean pooling over contextualized parts:

```text
image_feat = mean(part_context, dim=1)
```

Classifier:

```text
Linear(H, 128) -> GELU -> Dropout -> Linear(128, 7)
```

No class-to-part attention is included in D6A. That is reserved for D6B.

## Loss

Loss name:

```text
d6_hierarchical_motif
```

Total:

```text
L = CE
  + lambda_slot_div * L_slot_div
  + lambda_border * L_border
  + lambda_slot_smooth * L_slot_smooth
```

Default config:

```text
lambda_slot_div = 0.01
lambda_border = 0.005
lambda_slot_smooth = 0.0
```

Class weights use the same FER-2013 counts and `class_weight_power=0.25` as D5.

### Slot Diversity

Masks are L2-normalized over pixels, then slot cosine similarity is penalized off diagonal:

```text
sim = m @ m.T
L_slot_div = mean(off_diag(sim))
```

### Border Penalty

A soft foreground prior penalizes slot mass on the image border. This is not a hard-coded facial part; it only discourages learning the frame/background.

Default border width:

```text
border_width = 3
```

### Smoothness

Optional smoothness compares slot assignment differences across graph edges. It is implemented but disabled by default to keep D6A simple and fast.

## Config

Experiment config:

```text
configs/experiments/d6a_slot_pixel_part_graph_motif.yaml
```

The default output root is:

```text
output/d6a_slot_pixel_part_graph_motif
```

## Commands

Train:

```bash
python scripts/train_d5a.py --config configs/experiments/d6a_slot_pixel_part_graph_motif.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6a_slot_pixel_part_graph_motif.yaml --checkpoint output/d6a_slot_pixel_part_graph_motif/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d6.py --config configs/experiments/d6a_slot_pixel_part_graph_motif.yaml --checkpoint output/d6a_slot_pixel_part_graph_motif/checkpoints/best.pth
```

Smoke:

```bash
python scripts/train_d5a.py --config configs/experiments/d6a_slot_pixel_part_graph_motif.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

## Outputs

Expected run directory:

```text
output/d6a_slot_pixel_part_graph_motif/
  checkpoints/
    best.pth
    last.pth
  evaluation/
    metrics.json
    classification_report.json
    classification_report.txt
    predictions.csv
    confusion_matrix.png
    correct_examples.png
    wrong_examples.png
  figures/
    d6_part_masks/
    d6_part_attention/
    d6_slot_summary/
  training_history.json
  resolved_config.yaml
```

## Reading Visualization

Use `d6_part_masks` to check whether slots discover distinct face regions instead of collapsing into one blob. Good signs are multiple slots covering different face areas such as eyes, mouth, brows, nose, or mouth corners without any of these being hard-coded.

Use `d6_slot_summary/avg_slot_grid.png` to inspect stable average slot behavior across samples.

Use `d6_slot_summary/slot_similarity.png` to detect slot collapse. High off-diagonal similarity means many slots are learning the same region.

Use `d6_slot_summary/slot_area.csv` and `slot_area.png` to check whether any slot is unused or dominating.

Use `d6_part_attention` to inspect whether part-to-part attention is non-uniform and changes across correct and wrong examples.

## Success Criteria

Minimal classification success:

```text
test_macro_f1 > 0.30
accuracy not much below 0.40
Happy pred_count does not dominate too heavily
Fear F1 >= D5B-2A if possible
Disgust pred_count > 0 is a strong positive signal
```

Motif success:

```text
slots do not collapse into one region
slots cover different face regions
slots avoid mostly border/background mass
part-to-part attention is not uniform
some slots are interpretable as facial regions without hard-coding
```

## Later Variants

D6B should add class-to-part attention `[B, 7, K]` so each emotion can select different soft parts.

D6C can use D5B priors for slot or class attention initialization/regularization.

D6D can add quality-weighted priors, data cleaning, or augmentation after D6A/B are stable.
