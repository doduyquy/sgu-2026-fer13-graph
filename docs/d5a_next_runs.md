# D5A Next Runs

## Evaluate Best Checkpoint

```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode evaluate \
  --checkpoint /kaggle/working/fer_d5_outputs/checkpoints/best.pth \
  --graph_repo_path /kaggle/input/datasets/irthn1311/graph-repo/graph_repo \
  --device cuda:0 \
  --output_root /kaggle/working/fer_d5_outputs_eval \
  --no_wandb
```

## Diagnostic Report

```bash
python scripts/diagnose_d5a_run.py \
  --output_root /kaggle/working/fer_d5_outputs \
  --eval_root /kaggle/working/fer_d5_outputs_eval/evaluation
```

Report:

```text
/kaggle/working/fer_d5_outputs/diagnostics/d5a_diagnostic_report.md
```

## 1. CE Only

```bash
python scripts/run_experiment.py \
  --config configs/experiments/d5a_ce_only.yaml \
  --environment kaggle \
  --mode train \
  --epochs 10 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/fer_d5_outputs_ce_only \
  --no_wandb
```

## 2. CE + Contrast Light

```bash
python scripts/run_experiment.py \
  --config configs/experiments/d5a_ce_contrast_light.yaml \
  --environment kaggle \
  --mode train \
  --epochs 10 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/fer_d5_outputs_ce_contrast_light \
  --no_wandb
```

## 3. Node Score Only

```bash
python scripts/run_experiment.py \
  --config configs/experiments/d5a_node_score_only.yaml \
  --environment kaggle \
  --mode train \
  --epochs 10 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/fer_d5_outputs_node_score_only \
  --no_wandb
```

## Compare Runs

```bash
python scripts/compare_d5a_runs.py \
  --runs \
  /kaggle/working/fer_d5_outputs \
  /kaggle/working/fer_d5_outputs_ce_only \
  /kaggle/working/fer_d5_outputs_ce_contrast_light \
  /kaggle/working/fer_d5_outputs_node_score_only
```

Run order recommendation:

1. `d5a_ce_only`
2. `d5a_ce_contrast_light`
3. `d5a_node_score_only`
