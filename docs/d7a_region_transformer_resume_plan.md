# D7A Region Transformer Resume Plan

## Why This Run

`d7a_graph_swin_region_transformer` is the current D7A Graph-Swin standalone main candidate after Stage 2. It is the only Stage 2 variant that improved accuracy, macro F1, and weighted F1 over the previous D7A baseline.

Key result:

- accuracy: 0.6002
- macro F1: 0.5794
- weighted F1: 0.5969
- best_epoch: 119
- best_val_macro_f1: 0.576968
- epoch 120 val_macro_f1: about 0.5766

The best validation point is at the end of the 120-epoch budget, so the next conservative step is to resume the same architecture and continue to epoch 150.

## Checkpoints

Kaggle input checkpoint paths:

```bash
/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth
/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/last.pth
```

Use `last.pth` for full training resume. Keep `best.pth` as the seeded best checkpoint in the new output directory so evaluation still has a valid best checkpoint if the resumed epochs do not improve.

## Configs

Primary resume config:

```text
configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml
```

Fallback fine-tune config:

```text
configs/experiments/d7a_graph_swin_region_transformer_finetune_best.yaml
```

Both inherit the existing `d7a_graph_swin_region_transformer.yaml` model settings. No architecture, graph repo, node dimension, or loss changes are introduced.

## Resume Vs Fine-Tune

Resume mode loads:

- model weights
- optimizer state when present
- scheduler state when present
- epoch number
- best metric metadata from config/checkpoint

Fine-tune mode loads:

- model weights only
- new optimizer and scheduler
- epoch count starts from 1
- lower LR, shorter patience

Resume is preferred. Fine-tune is only a fallback if optimizer/scheduler state cannot be restored safely.

## Smoke Commands

Checkpoint existence:

```bash
python -c "from pathlib import Path; paths=['/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth','/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/last.pth']; [print(p, Path(p).exists(), Path(p).stat().st_size if Path(p).exists() else None) for p in paths]"
```

Checkpoint keys:

```bash
python -c "import torch; p='/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/last.pth'; ckpt=torch.load(p,map_location='cpu',weights_only=False); print(sorted(ckpt.keys())); print('epoch=',ckpt.get('epoch')); print('has_model=', 'model_state_dict' in ckpt); print('has_optimizer=', 'optimizer_state_dict' in ckpt); print('has_scheduler=', 'scheduler_state_dict' in ckpt); print('metrics_keys=', sorted((ckpt.get('metrics') or {}).keys())[:20])"
```

Resume smoke:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml \
  --resume /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/last.pth \
  --resume_best_checkpoint /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --epochs 121 \
  --no_wandb
```

Fallback fine-tune smoke:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_finetune_best.yaml \
  --init_checkpoint /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --epochs 1 \
  --no_wandb
```

Eval smoke:

```bash
python scripts/evaluate_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml \
  --checkpoint output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
  --max_test_batches 1
```

## Full Commands

Primary full resume:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml \
  --resume /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/last.pth \
  --resume_best_checkpoint /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth
```

Evaluate:

```bash
python scripts/evaluate_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml \
  --checkpoint output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d7.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_long150_resume.yaml \
  --checkpoint output/d7a_graph_swin_region_transformer_long150_resume/checkpoints/best.pth \
  --max_samples 32
```

Fallback full fine-tune:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d7a_graph_swin_region_transformer_finetune_best.yaml \
  --init_checkpoint /kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-checkpoint/checkpoints/best.pth
```

## Success Criteria

Resume is successful if macro F1 exceeds 0.5794, weighted F1 reaches or exceeds 0.5969, and accuracy stays at least near 0.6002. Fear should stay at least 0.4136, Sad should not fall below 0.445, Neutral should stay at least 0.58, and Disgust should stay at least 0.55.

If no resumed checkpoint improves the Stage 2 champion, keep the original `d7a_graph_swin_region_transformer` checkpoint as main.

## Fallback

If full resume fails because an older checkpoint lacks scheduler state, the loader will still restore model and optimizer and print a scheduler warning. If optimizer restoration itself fails, use the fine-tune config with `--init_checkpoint best.pth`.

