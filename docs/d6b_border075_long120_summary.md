# D6B Border075 Long120 Summary

## Why Border075 Long120

`d6b_class_part_graph_motif_border075_long120` tests the most conservative next step after the five D6B tuning runs: keep the best generalizing loss setting from `border075`, then extend training to 120 epochs like `long120`.

This run does not change the architecture, graph builder, data pipeline, number of slots, or objective family. It only combines:

- `lambda_border: 0.0075` from `border075`
- `epochs: 120` and `early_stopping_patience: 25` from `long120`

## What Border075 Contributes

`border075` is the current main-report candidate because it produced the best accuracy and weighted F1 among the five tuning runs:

- accuracy: `0.5606`
- macro F1: `0.5359`
- weighted F1: `0.5569`
- Disgust F1: `0.5047`
- Fear F1: `0.3702`
- Sad F1: `0.4609`
- Surprise F1: `0.7047`
- Neutral F1: `0.5243`

Its main value is stronger generalization and a better balance between weighted performance and per-class behavior. Sad F1 is especially stronger than `long120`.

## What Long120 Contributes

`long120` is the macro/Disgust champion checkpoint:

- accuracy: `0.5570`
- macro F1: `0.5376`
- weighted F1: `0.5493`
- Disgust F1: `0.5593`
- Fear F1: `0.3758`
- Sad F1: `0.4068`
- Surprise F1: `0.7168`
- Neutral F1: `0.5422`

Its main value is improved macro F1 and much better Disgust F1, but it has a larger train/validation gap and weaker Sad/Happy behavior than `border075`.

## Why Not Long140 Now

`long120` already shows signs of a larger train/validation gap. Jumping to 140 epochs would mix two questions at once: whether `border075` benefits from longer training, and whether still longer training is useful or just overfits. `border075-long120` isolates the cleaner question first.

## Why Not Border010 Or Balance015

`border010` is not the next direction because its stronger border penalty appears to suppress useful representation and reduce performance.

`balance015` is not prioritized because it makes slots more uniform but weakens semantic specialization. The current goal is not more uniform slots; it is better macro/weighted performance without slot collapse or border-heavy motifs.

## Main Config

Config file:

```bash
configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml
```

Output directory:

```bash
output/d6b_class_part_graph_motif_border075_long120
```

Key settings:

```yaml
experiment:
  name: d6b_class_part_graph_motif_border075_long120

model:
  name: slot_pixel_part_graph_motif_d6b
  num_part_slots: 16
  use_class_part_attention: true

loss:
  name: d6b_class_part_motif
  lambda_border: 0.0075
  lambda_slot_div: 0.01
  lambda_slot_balance: 0.01
  border_width: 3
  border_loss_type: slot_ratio
  slot_balance_type: kl_uniform

training:
  epochs: 120
  monitor: val_macro_f1
  early_stopping_patience: 25
  grad_clip_norm: 3.0
  amp: true
  amp_init_scale: 1024
```

## Smoke Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml --checkpoint output/d6b_class_part_graph_motif_border075_long120/checkpoints/best.pth --max_test_batches 1
```

```bash
python scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml --checkpoint output/d6b_class_part_graph_motif_border075_long120/checkpoints/best.pth --max_samples 4
```

Expected smoke artifacts:

- `output/d6b_class_part_graph_motif_border075_long120/checkpoints/best.pth`
- `output/d6b_class_part_graph_motif_border075_long120/checkpoints/last.pth`
- `output/d6b_class_part_graph_motif_border075_long120/training_history.json`
- `output/d6b_class_part_graph_motif_border075_long120/resolved_config.yaml`
- `output/d6b_class_part_graph_motif_border075_long120/figures/d6_slot_summary`
- `output/d6b_class_part_graph_motif_border075_long120/figures/d6_class_part_attention`
- `output/d6b_class_part_graph_motif_border075_long120/figures/d6_class_motif_maps`

## Full Run Commands

Train:

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml --checkpoint output/d6b_class_part_graph_motif_border075_long120/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_border075_long120.yaml --checkpoint output/d6b_class_part_graph_motif_border075_long120/checkpoints/best.pth --max_samples 32
```

## Success Criteria

Use `border075` as the accuracy/weighted baseline and `long120` as the macro/Disgust baseline.

`border075-long120` is successful if:

- accuracy is at least `0.5606`, or nearly equal with clearly higher macro F1
- macro F1 is at least `0.5376`, or at least `0.5359`
- weighted F1 is at least `0.5569`, or does not drop much
- Disgust F1 is at least `0.50`
- Fear F1 is at least `0.37`
- Sad F1 does not fall strongly below `0.44`
- Surprise F1 is at least `0.70`
- Neutral F1 is at least `0.52`
- train/validation gap is not as large as `long120`
- effective slots stay around `12` or higher
- class attention is not more uniform than earlier runs
- no slot collapse or severe border-heavy motif appears

## Reading Results After The Run

Start with `metrics.json` and `classification_report.json` for the headline comparison. Then inspect `training_history.json` for best epoch, learning-rate reductions, validation macro F1, validation accuracy, validation loss, loss components, effective slots, and class-part attention entropy.

If it beats both `border075` and `long120`, promote `border075-long120` to the main model. If it mainly wins macro F1 but loses weighted F1, Sad, or Happy, keep `border075` as the main report model and keep `border075-long120` as a macro checkpoint. If it does not improve, keep `border075` as main, keep `long120` as the macro/Disgust champion, and move the next research step to D6C.
