# D6B Implementation Summary

## Why D6B

D6A improved the D5 line by moving from class-level pixel motifs to self-discovered pixel-to-part slots, part masks, part features, part-to-part attention, and an image classifier. It reached 53.11% accuracy and 0.4888 macro F1, with Disgust recovered compared with earlier runs.

The main remaining problems were slot collapse, a border loss that was constant under per-pixel softmax assignment, weak Fear separation, Sad over-prediction, and part attention that was not emotion-specific.

## What D6B Adds

D6B keeps the D6A pixel graph, edge-aware pixel encoder, K=16 slots, and part-to-part attention. After `part_context [B,K,H]`, it adds class-to-part attention:

```text
class_queries        [7,H]
class_key(context)   [B,K,H]
class_value(context) [B,K,H]
class_part_attn      [B,7,K]
class_repr           [B,7,H]
logits               [B,7]
```

For each emotion, a learned class query attends over the K discovered parts. The classifier is a shared `Linear(H,1)` applied to each class representation, so image logits come from emotion-specific part mixtures instead of global mean pooling.

## Border Loss Fix

D6A used a pixel-mean border penalty. Because slot probabilities sum to 1 per pixel, this term was effectively constant.

D6B uses `border_loss_type: slot_ratio`:

```text
slot_border_mass = sum_v part_masks[k,v] * border_mask[v] / sum_v part_masks[k,v]
loss_border = mean(slot_border_mass)
```

This penalizes slots whose own mass concentrates near image borders and logs `border_mass_per_slot`.

## Slot Balance Loss

D6B adds a light KL-to-uniform regularizer over slot areas:

```text
slot_area = part_masks.mean(dim=2)      # [B,K]
area_norm = slot_area / sum(slot_area)
loss_slot_balance = KL(area_norm || uniform_K)
```

The default weight is `lambda_slot_balance: 0.01`, intended only to resist collapse, not force perfectly equal semantic parts.

## Loss

Config name: `d6b_class_part_motif`

```text
L = CE
  + 0.01  * L_slot_div
  + 0.005 * L_border_slot_ratio
  + 0.01  * L_slot_balance
```

The original D6A loss path still works. D6A keeps its existing config and model name.

## Config

Main config:

```text
configs/experiments/d6b_class_part_graph_motif.yaml
```

Key model settings:

```yaml
model:
  name: slot_pixel_part_graph_motif_d6b
  hidden_dim: 64
  num_part_slots: 16
  part_heads: 4
  use_class_part_attention: true
```

## Visualization Outputs

Existing D6 outputs are preserved:

```text
figures/d6_part_masks/
figures/d6_part_attention/
figures/d6_slot_summary/
```

D6B adds:

```text
figures/d6_class_part_attention/class_part_attn_grid.png
figures/d6_class_part_attention/class_part_attn_per_sample_*.png
figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png
figures/d6_class_part_attention/class_part_attn_avg_by_pred_class.png
figures/d6_class_motif_maps/class_pixel_motif_trueclass_avg.png
figures/d6_class_motif_maps/class_pixel_motif_predclass_avg.png
figures/d6_class_motif_maps/sample_class_motif_*.png
```

Class pixel motif maps are computed as:

```text
class_pixel_attn[b,c,v] = sum_k class_part_attn[b,c,k] * part_masks[b,k,v]
```

## Diagnostics

Evaluation writes `evaluation/d6b_diagnostics.json` when D6-style tensors are present. It includes average slot areas, border mass per slot, class-part entropy, and top slots per class.

## Commands

Smoke train:

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6b_class_part_graph_motif.yaml --checkpoint output/d6b_class_part_graph_motif/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif.yaml --checkpoint output/d6b_class_part_graph_motif/checkpoints/best.pth --max_samples 32
```

Full train:

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif.yaml
```

## Success Criteria

D6B should keep accuracy near or above 0.53, push macro F1 to at least 0.49, improve Fear over 0.2966 F1, reduce Sad over-prediction or improve Sad precision, keep Disgust above 0.30 F1, reduce top-slot collapse, and show class-specific slot patterns in the new attention visualizations.
