# D7 + D8B Ensemble and D8B Border020 Seed Repeat Plan

## Why Package A New Official Ensemble

The previous D7 official ensemble used:

- `d7_seed44`
- `d7_long150_resume`
- `d7_window4_region_transformer`
- method: `logit_average`

It reached accuracy `0.6436`, macro F1 `0.6292`, and weighted F1 `0.6401`.

The D8B four-run analysis found that `d8b_face_aware_graph_swin_border020` is the new single-model macro/weighted champion, while `d8b_face_aware_graph_swin_area045` is a useful companion because it keeps stronger Angry/Fear/Disgust behavior. The best checked ensemble is:

- `d7_seed44`
- `d7_long150_resume`
- `d7_window4_region_transformer`
- `d8b_border020`
- `d8b_area045`
- method: `probability_average`

Expected metrics:

- accuracy ~= `0.6461`
- macro F1 ~= `0.6342`
- weighted F1 ~= `0.6421`
- pred_count ~= `[422, 51, 407, 921, 650, 429, 709]`

Probability averaging is the new official method because the D7 and D8B logits have different calibration/scale. Averaging probabilities preserves the complementary class behavior without letting any single member dominate via logit magnitude.

## Predictions Mode

Windows:

```bat
python scripts/evaluate_d7_ensemble.py ^
  --prediction_files ^
    d7_seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv ^
    d7_long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv ^
    d7_window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv ^
    d8b_border020:output/d8b_face_aware_graph_swin_border020/evaluation/predictions.csv ^
    d8b_area045:output/d8b_face_aware_graph_swin_area045/evaluation/predictions.csv ^
  --method probability_average ^
  --output_dir output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg ^
  --split test ^
  --save_logits true
```

Linux/Kaggle:

```bash
python scripts/evaluate_d7_ensemble.py \
  --prediction_files \
    d7_seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv \
    d7_long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv \
    d7_window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv \
    d8b_border020:output/d8b_face_aware_graph_swin_border020/evaluation/predictions.csv \
    d8b_area045:output/d8b_face_aware_graph_swin_area045/evaluation/predictions.csv \
  --method probability_average \
  --output_dir output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg \
  --split test \
  --save_logits true
```

Logit-average comparison:

```bash
python scripts/evaluate_d7_ensemble.py \
  --prediction_files \
    d7_seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv \
    d7_long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv \
    d7_window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv \
    d8b_border020:output/d8b_face_aware_graph_swin_border020/evaluation/predictions.csv \
    d8b_area045:output/d8b_face_aware_graph_swin_area045/evaluation/predictions.csv \
  --method logit_average \
  --output_dir output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_logitavg \
  --split test \
  --save_logits true
```

## Checkpoint Mode

```bash
python scripts/evaluate_d7_ensemble.py \
  --members \
    d7_seed44:configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml:output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth \
    d7_long150_resume:configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml:output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
    d7_window4_region_transformer:configs/experiments/d7a_graph_swin_region_transformer_window4.yaml:output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth \
    d8b_border020:configs/experiments/d8b_face_aware_graph_swin_border020.yaml:output/d8b_face_aware_graph_swin_border020/checkpoints/best.pth \
    d8b_area045:configs/experiments/d8b_face_aware_graph_swin_area045.yaml:output/d8b_face_aware_graph_swin_area045/checkpoints/best.pth \
  --method probability_average \
  --output_dir output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg \
  --split test \
  --graph_repo_path artifacts/graph_repo \
  --chunk_cache_size 8
```

For a load/shape smoke:

```bash
python scripts/evaluate_d7_ensemble.py \
  --members \
    d7_seed44:configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml:output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth \
    d7_long150_resume:configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml:output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
    d7_window4_region_transformer:configs/experiments/d7a_graph_swin_region_transformer_window4.yaml:output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth \
    d8b_border020:configs/experiments/d8b_face_aware_graph_swin_border020.yaml:output/d8b_face_aware_graph_swin_border020/checkpoints/best.pth \
    d8b_area045:configs/experiments/d8b_face_aware_graph_swin_area045.yaml:output/d8b_face_aware_graph_swin_area045/checkpoints/best.pth \
  --method probability_average \
  --output_dir output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg_smoke_ckpt \
  --split test \
  --graph_repo_path artifacts/graph_repo \
  --chunk_cache_size 8 \
  --max_batches 1
```

## Why Border020 Seed Repeat Is Needed

`d8b_face_aware_graph_swin_border020` is the best D8B run and the new single-model macro/weighted champion, but the margin over D7 seed44/window4 is small. Seed repeat decides whether border020 is a stable single-model improvement or mainly a useful ensemble member.

New configs:

- `configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml`
- `configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml`

Both inherit `d8b_face_aware_graph_swin_border020.yaml` and only change:

- experiment name
- seed
- output directory

Architecture and loss remain unchanged: `window_size=6`, `shift_size=3`, region transformer enabled, `node_dim=7`, `edge_dim=5`, `hidden_dim=64`, `pixel_gate_area_target=0.55`, `pixel_gate_border_weight=0.02`, `pixel_gate_border_width=3`, `pixel_gate_smooth_weight=0.001`, epochs `120`, patience `25`, LR `0.0005`.

## Seed Repeat Smoke Commands

Debug forward:

```bash
python scripts/debug_d8b_forward.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml

python scripts/debug_d8b_forward.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml
```

One-batch train smoke:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --epochs 1 \
  --no_wandb

python scripts/train_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --epochs 1 \
  --no_wandb
```

## Full Seed Repeat Commands

Train:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml

python scripts/train_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml \
  --checkpoint output/d8b_face_aware_graph_swin_border020_seed43/checkpoints/best.pth

python scripts/evaluate_d5a.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml \
  --checkpoint output/d8b_face_aware_graph_swin_border020_seed44/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d8b_face_aware.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed43.yaml \
  --checkpoint output/d8b_face_aware_graph_swin_border020_seed43/checkpoints/best.pth \
  --max_samples 32

python scripts/visualize_d8b_face_aware.py \
  --config configs/experiments/d8b_face_aware_graph_swin_border020_seed44.yaml \
  --checkpoint output/d8b_face_aware_graph_swin_border020_seed44/checkpoints/best.pth \
  --max_samples 32
```

## Decision Criteria After Seed Repeat

Border020 original:

- accuracy `0.6085`
- macro F1 `0.5907`
- weighted F1 `0.6068`

Seed repeat succeeds if:

- mean macro F1 of seed42/43/44 >= `0.585`
- mean accuracy >= `0.60`
- no seed collapses below macro F1 `0.57`
- pixel gate does not collapse to all-zero/all-one
- Fear/Sad/Neutral do not drop abnormally
- `pixel_gate_border_mean` stays below `pixel_gate_center_mean`

If seed repeat is good, D8B border020 can be called a stable single-model champion and D8B-v2/D8C can be considered next. If not, keep border020 as an ensemble member but avoid calling it stable.
