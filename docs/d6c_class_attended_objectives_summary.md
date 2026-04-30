# D6C-Light Class-Attended Objectives

## Why D6C After D6B

D5A/D5B used flat pixel-level motif masks and saturated around 37.8-40.5% accuracy, macro F1 0.28-0.30, with Disgust often receiving zero predictions.

D6A moved to a hierarchical pixel -> part -> relation motif pipeline and reached:

- accuracy: 53.11%
- macro F1: 0.4888
- weighted F1: 0.5262

D6B added class-to-part attention and reached:

- accuracy: 54.08%
- macro F1: 0.4985
- weighted F1: 0.5389

After D6B tuning, `border075` is the main report model and `long120` is the macro/Disgust champion. The remaining failure mode is not graph input capacity, but class-conditioned objective sharpness.

## Remaining D6B Issues

- Fear, Sad, and Neutral still have similar class attention.
- Sad-Neutral attention similarity is especially high.
- Fear still confuses with Sad, Neutral, and Surprise.
- Class-to-part attention is useful but diffuse.
- Some class motifs still use boundary, background, or side-face cues.

## What D6C-Light Adds

D6C-light keeps the D6B backbone unchanged and adds three objective terms:

1. Class-attended border loss
2. Class-attention separation loss on confusion pairs
3. Supervised contrastive loss on the true `class_repr`

It does not change pixel graph input, pixel encoder, pixel-to-part attention, part-to-part attention, class-to-part attention architecture, `K=16`, `hidden_dim=64`, or `lr=0.0005`.

## Tensor Shapes

- `part_masks`: `[B, K, N]`
- `class_part_attn`: `[B, C, K]`
- `class_pixel_attn = einsum("bck,bkn->bcn", class_part_attn, part_masks)`: `[B, C, N]`
- `class_repr`: `[B, C, H]`
- `labels`: `[B]`

## Loss Formula

```text
L = lambda_cls * CE
  + lambda_slot_div * L_slot_div
  + lambda_border * L_slot_border
  + lambda_slot_balance * L_slot_balance
  + lambda_class_border * L_class_border
  + lambda_class_attn_sep * L_class_attn_sep
  + lambda_supcon * L_supcon
```

Default D6C-light weights:

- `lambda_cls: 1.0`
- `lambda_slot_div: 0.01`
- `lambda_border: 0.0075`
- `lambda_slot_balance: 0.01`
- `lambda_slot_smooth: 0.0`
- `lambda_class_border: 0.0025`
- `lambda_class_attn_sep: 0.005`
- `lambda_supcon: 0.03`
- `class_attn_sep_margin: 0.90`
- `supcon_temperature: 0.2`

## Class-Attended Border Loss

For each sample, D6C computes the motif used by the true class:

```text
true_motif = class_pixel_attn[batch_idx, label, :]
border_ratio = sum(true_motif * border_mask) / (sum(true_motif) + eps)
L_class_border = mean(border_ratio)
```

This penalizes class-specific motifs that rely too much on boundary or background pixels.

Logged keys:

- `loss_class_border`
- `diag_true_class_border_mass_mean`
- `diag_true_class_border_mass_max`

## Class-Attention Separation

The average class-to-part attention over the batch is compared only for known confusion pairs:

```text
[(Fear, Sad), (Fear, Neutral), (Fear, Surprise), (Sad, Neutral), (Angry, Disgust)]
```

For each pair:

```text
sim = cosine(avg_attn[c1], avg_attn[c2])
penalty = relu(sim - margin)
```

Logged keys include:

- `loss_class_attn_sep`
- `diag_class_attn_sim_fear_sad`
- `diag_class_attn_sim_fear_neutral`
- `diag_class_attn_sim_fear_surprise`
- `diag_class_attn_sim_sad_neutral`
- `diag_class_attn_sim_angry_disgust`
- `diag_class_part_entropy_mean`

## Supervised Contrastive On True Class Representation

D6C takes:

```text
z = class_repr[batch_idx, label, :]
z = normalize(z)
```

It applies supervised contrastive loss over batch samples with the same label as positives and all other labels as negatives. Anchors with no positive in the batch are skipped. If no anchor has positives, the term returns zero.

Logged keys:

- `loss_supcon`
- `diag_supcon_valid_anchors`
- `diag_supcon_positive_pairs`

## Configs

Main config:

```text
configs/experiments/d6c_class_attended_objectives_light.yaml
```

Ablation configs:

```text
configs/experiments/d6c_class_attended_objectives_no_supcon.yaml
configs/experiments/d6c_class_attended_objectives_no_sep.yaml
configs/experiments/d6c_class_attended_objectives_border_only.yaml
```

Main output:

```text
output/d6c_class_attended_objectives_light
```

## Smoke Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_light.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
python scripts/evaluate_d5a.py --config configs/experiments/d6c_class_attended_objectives_light.yaml --checkpoint output/d6c_class_attended_objectives_light/checkpoints/best.pth --max_test_batches 1
python scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_light.yaml --checkpoint output/d6c_class_attended_objectives_light/checkpoints/best.pth --max_samples 4
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

## Full Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_light.yaml
python scripts/evaluate_d5a.py --config configs/experiments/d6c_class_attended_objectives_light.yaml --checkpoint output/d6c_class_attended_objectives_light/checkpoints/best.pth
python scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_light.yaml --checkpoint output/d6c_class_attended_objectives_light/checkpoints/best.pth --max_samples 32
```

## Success Criteria

D6C-light is successful if it matches or improves the current D6B frontier:

- macro F1 >= 0.5376, or nearly equal with better accuracy/weighted F1
- accuracy >= 0.56, or no large drop
- Fear F1 > 0.37
- Sad/Neutral confusion decreases
- Sad F1 stays >= 0.44
- Disgust F1 stays >= 0.50 or does not drop strongly
- Fear/Sad/Neutral class attention similarity decreases versus `border075`
- class motif maps are sharper and use less border/background
- effective slots stay >= 12
- no strong overfit

Accuracy equal to `border075` is still useful if motifs become clearer, Fear/Sad/Neutral attention similarity drops, and macro F1 increases slightly.
