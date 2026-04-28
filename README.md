# FER-D5: Class-Level Pixel Motif Graph Retrieval

Clean D5 project for FER-2013. It builds full 48x48 pixel graphs, learns class-level node/edge motif prototypes, retrieves soft class subgraphs, and classifies by graph matching scores.

## D5A Idea

Pipeline:

```text
FER-2013 CSV -> graph_repo -> full graph dataloader
-> class-level pixel motif prototypes T_0..T_6
-> soft retrieved subgraph S_i,c per class
-> graph matching logits [B, 7]
-> CE + motif regularization
-> evaluation + motif mask visualization
```

No CNN, no 41D descriptor motif, no candidate attention dataset, no pixel motif v2, no legacy motif bank, and no greedy top-K.

## Data Contract

Batch keys:

- `x` / `node_features`: `[B, 2304, 7]`
- `edge_index`: `[2, 17860]`
- `edge_attr`: `[B, 17860, 5]`
- `node_mask`: `[B, 2304]`
- `y` / `label`: `[B]`
- `graph_id`: `[B]`

Model outputs:

- `logits`: `[B, 7]`
- `node_attn`: `[B, 7, 2304]`
- `edge_attn`: `[B, 7, 17860]`
- `class_node_gate`: `[7, 2304]`
- `class_edge_gate`: `[7, 17860]`
- diagnostics dict

## Install

```bash
cd fer_d5
pip install -r requirements.txt
```

## Local Commands

Build graph repository:

```bash
python scripts/build_graph_repo.py --config configs/d5a_local.yaml
```

Inspect:

```bash
python scripts/inspect_graph_repo.py --config configs/d5a_local.yaml
```

Debug one batch:

```bash
python scripts/debug_d5a_batch.py --config configs/d5a_local.yaml --batch_size 2
```

Smoke run:

```bash
python scripts/run_experiment.py --config configs/d5a_local.yaml --mode smoke --max_train_batches 3 --max_val_batches 2 --max_test_batches 2 --batch_size 2
```

Train:

```bash
python scripts/train_d5a.py --config configs/d5a_local.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config configs/d5a_local.yaml --checkpoint outputs/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d5.py --config configs/d5a_local.yaml --checkpoint outputs/checkpoints/best.pth --max_samples 16
```

## Kaggle Usage

Use `notebooks/kaggle_d5_end_to_end.ipynb`.

1. Add a Kaggle input dataset containing `train.csv`, `val.csv`, and `test.csv`.
2. Clone or upload this repo.
3. In the notebook config cell, start with `MODE = "smoke"`.
4. For a full run, use `MODE = "build_and_train"`.

Recommended first run:

```bash
python scripts/run_experiment.py --config configs/d5a_kaggle.yaml --mode smoke --max_train_batches 3 --max_val_batches 2
```

Full run:

```bash
python scripts/run_experiment.py --config configs/d5a_kaggle.yaml --mode build_and_train
```

## Expected Outputs

- `outputs/checkpoints/best.pth`
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/predictions.csv`
- `outputs/figures/d5a_class_gates/*.png`
- `outputs/figures/d5a_attention/**/*.png`

## Intentionally Not Included

- candidate motif bank
- descriptor 41D pipeline
- D3.1 candidate slots
- D4A generic slot pooling classifier
- CNN classifier branches
