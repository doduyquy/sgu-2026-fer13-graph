# D6B Long100 and Light Tuning Summary

## Why D6B-long100

D6B improved over D6A but its best validation macro F1 happened at epoch 60, exactly the final epoch of the original run. That means the run did not prove a plateau. D6B-long100 keeps the same architecture and loss weights, and only extends training to test whether the model still improves with more epochs.

Original D6B reference metrics:

| metric | value |
| --- | ---: |
| accuracy | 0.5408 |
| macro F1 | 0.4985 |
| weighted F1 | 0.5389 |
| Fear F1 | 0.3698 |
| Disgust F1 | 0.3529 |
| Sad F1 | 0.4232 |
| Surprise F1 | 0.6823 |
| Neutral F1 | 0.5237 |

D6B also reduced slot collapse strongly versus D6A: top-2 slot mass went from about 85.6% to 37.1%, and effective slots increased from about 3.7 to 10.8.

## Why 100 Epochs

80 epochs may be too conservative because the original D6B was still setting the best validation macro F1 at epoch 60. 100 epochs gives a clearer read on whether improvement continues, while avoiding the higher overfit and compute risk of jumping directly to 120.

`d6b_class_part_graph_motif_long120.yaml` is prepared only as a fallback. Run it only if long100 has `best_epoch >= 95` and `val_macro_f1` is still trending upward.

## Why Not D6C Yet

This step intentionally avoids architectural or objective changes. No supervised contrastive loss, class-attended losses, focal loss, auxiliary CE, K=24, hybrid CNN-GNN, data cleaning, or augmentation is added. The goal is to isolate the effect of longer D6B training and light loss-weight tuning.

## New Configs

- `configs/experiments/d6b_class_part_graph_motif_long100.yaml`
- `configs/experiments/d6b_class_part_graph_motif_long120.yaml`
- `configs/experiments/d6b_class_part_graph_motif_border075.yaml`
- `configs/experiments/d6b_class_part_graph_motif_border010.yaml`
- `configs/experiments/d6b_class_part_graph_motif_balance015.yaml`

Run order:

1. Run long100 first.
2. Analyze validation curve, per-class metrics, LR schedule, class attention, slot/border diagnostics.
3. Run border075 only if border-heavy slots remain.
4. Run border010 only if border075 is insufficient.
5. Run balance015 only if slot balance still needs a light nudge.
6. Run long120 only if long100 is still improving near epoch 100.

## Logging Added

Training history now includes learning-rate fields per epoch:

- `lr`
- `train_lr`
- `lr_group_0`
- `lr_min`
- `lr_max`
- `lr_after_scheduler`
- `lr_min_after_scheduler`
- `lr_max_after_scheduler`
- `scheduler_monitor_value`
- `scheduler_monitor_val_macro_f1`
- `scheduler_lr_reduced`

Class-to-part attention diagnostics are logged when `class_part_attn [B,7,K]` exists:

- `train_diag_class_part_entropy_mean`
- `train_diag_class_part_max_mean`
- `train_diag_class_part_top1_mean`
- `train_diag_class_part_similarity_mean`
- `val_diag_class_part_entropy_mean`
- `val_diag_class_part_max_mean`
- `val_diag_class_part_top1_mean`
- `val_diag_class_part_similarity_mean`
- `val_diag_class_part_similarity_sad_neutral`
- `val_diag_class_part_similarity_fear_neutral`
- `val_diag_class_part_similarity_fear_surprise`
- `val_diag_class_part_similarity_disgust_angry`

Slot and border diagnostics include existing D6 diagnostics plus:

- `train_diag_effective_slots`
- `val_diag_effective_slots`
- `train_diag_border_mass_max`
- `val_diag_border_mass_max`

Validation also logs selected per-class fields:

- `val_fear_recall`
- `val_fear_f1`
- `val_sad_precision`
- `val_sad_recall`
- `val_neutral_f1`
- `val_disgust_f1`
- `val_pred_count_fear`
- `val_pred_count_sad`
- `val_pred_count_disgust`

Visualization exports additional CSV files under `figures/d6_class_part_attention/`:

- `class_part_attention_entropy.csv`
- `class_part_attention_similarity.csv`
- `top_slots_per_class.csv`

## Smoke Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

Check:

- `output/d6b_class_part_graph_motif_long100/training_history.json`
- `output/d6b_class_part_graph_motif_long100/checkpoints/best.pth`
- `output/d6b_class_part_graph_motif_long100/resolved_config.yaml`
- LR fields in `training_history.json`
- `val_diag_class_part_entropy_mean` when D6B attention output is present

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --checkpoint output/d6b_class_part_graph_motif_long100/checkpoints/best.pth --max_test_batches 1
```

```bash
python scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --checkpoint output/d6b_class_part_graph_motif_long100/checkpoints/best.pth --max_samples 4
```

## Full Run Commands

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml
```

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --checkpoint output/d6b_class_part_graph_motif_long100/checkpoints/best.pth
```

```bash
python scripts/visualize_d6.py --config configs/experiments/d6b_class_part_graph_motif_long100.yaml --checkpoint output/d6b_class_part_graph_motif_long100/checkpoints/best.pth --max_samples 32
```

Optional later:

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075.yaml
```

```bash
python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_balance015.yaml
```

## Reading D6B-long100

Success criteria versus original D6B:

- macro F1 >= 0.50
- accuracy >= 0.54
- weighted F1 >= 0.54 or close
- Fear F1 >= 0.36 or improved
- Disgust F1 not below 0.32
- Sad prediction count does not return to strong over-prediction
- top-2 slot mass remains much lower than D6A
- class-part entropy decreases slightly or attention becomes sharper
- no strong overfit

If macro F1 reaches at least 0.51, use D6B-long100 as the new main baseline. If it is stable but not better, choose between original D6B and long100 based on validation/test metrics and motif diagnostics.
