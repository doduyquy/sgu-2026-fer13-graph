# D7 Ensemble Official Evaluation

## Why This Stage Exists

D7 now has several complementary region-transformer checkpoints. Before opening D8, the current D7 family needs one official, reproducible ensemble evaluation that can be run from either checkpoints or saved logits. This freezes the D7 graph-only performance champion without changing the graph repo, model architecture, node dimensions, or loss.

## Default Ensemble

The official default ensemble is:

- `seed44`: `configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml`
- `long150_resume`: `configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml`
- `window4_region_transformer`: `configs/experiments/d7a_graph_swin_region_transformer_window4.yaml`

Default output:

- `output/d7_ensemble_seed44_long150_window4`

Default method:

- `logit_average`

## Single-Model References

- Macro/research champion: `d7a_graph_swin_region_transformer_seed44`
  - accuracy `0.6069`
  - macro F1 `0.5883`
  - weighted F1 `0.6008`
- Accuracy/weighted champion: `d7a_graph_swin_region_transformer_window4`
  - accuracy `0.6105`
  - macro F1 `0.5837`
  - weighted F1 `0.6049`
- Stable macro/reference checkpoint: `d7a_graph_swin_region_transformer_long150_resume`
  - accuracy `0.6013`
  - macro F1 `0.5878`
  - weighted F1 `0.5988`

## Expected Ensemble Metrics

For `seed44 + long150_resume + window4_region_transformer` with logit average from saved predictions:

- accuracy about `0.6436`
- macro F1 about `0.6292`
- weighted F1 about `0.6401`
- pred_count about `[396, 53, 439, 926, 629, 406, 740]`

Expected per-class F1:

- Angry `0.5411`
- Disgust `0.6481`
- Fear `0.4757`
- Happy `0.8299`
- Sad `0.5070`
- Surprise `0.7786`
- Neutral `0.6237`

## Predictions Mode

Use predictions mode when `predictions.csv` files already exist and contain logits columns `score_0` through `score_6`.

```bash
python scripts/evaluate_d7_ensemble.py \
  --prediction_files \
    seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv \
    long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv \
    window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv \
  --method logit_average \
  --output_dir output/d7_ensemble_seed44_long150_window4 \
  --split test \
  --save_logits true
```

The script aligns by `sample_id`, `graph_id`, or `index`. It raises an error if IDs differ, duplicates exist, labels disagree after alignment, or logits columns are missing.

## Checkpoint Mode

Use checkpoint mode when all three checkpoints are available and a graph repo is present.

```bash
python scripts/evaluate_d7_ensemble.py \
  --members \
    seed44:configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml:output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth \
    long150_resume:configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml:output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
    window4_region_transformer:configs/experiments/d7a_graph_swin_region_transformer_window4.yaml:output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth \
  --method logit_average \
  --output_dir output/d7_ensemble_seed44_long150_window4 \
  --split test \
  --graph_repo_path artifacts/graph_repo \
  --chunk_cache_size 8
```

Smoke checkpoint mode:

```bash
python scripts/evaluate_d7_ensemble.py \
  --members \
    seed44:configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml:output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth \
    long150_resume:configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml:output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
    window4_region_transformer:configs/experiments/d7a_graph_swin_region_transformer_window4.yaml:output/d7a_graph_swin_region_transformer_window4/checkpoints/best.pth \
  --method logit_average \
  --output_dir output/d7_ensemble_seed44_long150_window4_smoke_ckpt \
  --split test \
  --graph_repo_path artifacts/graph_repo \
  --chunk_cache_size 8 \
  --max_batches 1
```

## Other Methods

Probability average:

```bash
python scripts/evaluate_d7_ensemble.py \
  --prediction_files \
    seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv \
    long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv \
    window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv \
  --method probability_average \
  --output_dir output/d7_ensemble_seed44_long150_window4_probability \
  --split test
```

Weighted logit average:

```bash
python scripts/evaluate_d7_ensemble.py \
  --prediction_files \
    seed44:output/d7a_graph_swin_region_transformer_seed44/evaluation/predictions.csv \
    long150_resume:output/d7a_graph_swin_region_transformer_long150_resume/evaluation/predictions.csv \
    window4_region_transformer:output/d7a_graph_swin_region_transformer_window4/evaluation/predictions.csv \
  --method weighted_logit_average \
  --weights 1,1,1 \
  --output_dir output/d7_ensemble_seed44_long150_window4_weighted \
  --split test
```

With weights `1,1,1`, weighted logit average must match normal logit average.

## Kaggle Usage

The Kaggle notebook has a D7 ensemble stage:

- `D7_ENSEMBLE_CHOICE = "seed44_long150_window4_logit"`
- `RUN_D7_ENSEMBLE`
- `RUN_D7_ENSEMBLE_FROM_PREDICTIONS`
- `RUN_D7_ENSEMBLE_FROM_CHECKPOINTS`
- `D7_ENSEMBLE_PREDICTION_FILES`
- `D7_ENSEMBLE_MEMBERS`

For uploaded prediction files, edit `D7_ENSEMBLE_PREDICTION_FILES` to point at Kaggle input paths. For checkpoint mode, keep the checkpoint paths under `/kaggle/working/outputs/.../checkpoints/best.pth` or edit them to Kaggle input paths.

## Output Files

The ensemble output directory contains:

- `evaluation/metrics.json`
- `evaluation/classification_report.json`
- `evaluation/classification_report.txt`
- `evaluation/predictions.csv`
- `evaluation/confusion_matrix.png`
- `evaluation/correct_examples.png`
- `evaluation/wrong_examples.png`
- `ensemble_summary.json`
- `ensemble_members.json`
- `resolved_ensemble_config.yaml`
- `d7_ensemble_report.md`
- `ensemble_per_class_metrics.csv`
- `ensemble_confusion_focus.csv`
- `ensemble_member_comparison.csv`

In predictions mode, example grids are placeholders because the CSV files do not contain pixel tensors. Checkpoint mode writes real image grids.

## Confirmation Criteria

Accept the ensemble if:

- alignment reports `3589` test samples
- accuracy is near `0.6436`
- macro F1 is near `0.6292`
- weighted F1 is near `0.6401`
- pred_count is near `[396, 53, 439, 926, 629, 406, 740]`
- Disgust does not collapse
- Sad stays above `0.45`

If metrics differ substantially, check alignment key, `y_true` consistency, class order, logits columns, and whether `logit_average` versus `probability_average` was used.

## Decision After Running

If official ensemble metrics match expectations, freeze `seed44 + long150_resume + window4_region_transformer` logit average as the D7 performance champion. Keep `seed44` as the single-model macro champion and `window4_region_transformer` as the single-model accuracy/weighted champion. Only after this D7 decision is frozen should D8A/D8B be opened.
