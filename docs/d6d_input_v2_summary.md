# D6D Input V2 Minimal Summary

## Why Move From D6C To D6D

D6C-light improved interpretability and class-attention separation, but the completed runs did not beat the D6B border075 main checkpoint. The D6C phase is therefore closed for score chasing, and the next low-risk step is improving the graph input while keeping the D6B/D6C architecture unchanged.

Current references:

- D6B border075 main: accuracy 0.5606, macro F1 0.5359, weighted F1 0.5569.
- D6B long120 macro/Disgust reference: macro F1 0.5376, Disgust F1 0.5593.

## Input Audit Conclusion

The existing graph input is clean:

- no NaN/Inf
- no shape bug
- train/val/test drift is very small
- normalization does not need to be changed before D6D

The audit also identified safe input-only feature opportunities: gradient orientation, 5x5 local context, and soft border distance.

## Feature Contract

D6D-input-v2-minimal changes only node features:

- `node_dim = 12`
- `edge_dim = 5`

The original 7 node features remain first and unchanged:

1. `intensity`
2. `x_norm`
3. `y_norm`
4. `gx`
5. `gy`
6. `grad_mag`
7. `local_contrast`

The 5 new node features are:

8. `grad_ori_cos = gx / (grad_mag + eps)`
9. `grad_ori_sin = gy / (grad_mag + eps)`
10. `local_mean_5x5`, reflect-padded 5x5 mean intensity
11. `local_std_5x5`, reflect-padded 5x5 intensity std
12. `border_distance = min(row, col, 47-row, 47-col) / 23`

The edge features stay unchanged:

1. `dx`
2. `dy`
3. `dist`
4. `delta_intensity`
5. `intensity_similarity`

## What Is Intentionally Excluded

D6D-input-v2-minimal does not add center priors, Laplacian features, multi-scale edges, patch branches, dense `[2304,2304,3]` tensors, hard masks, landmarks, CNN/pretrained features, or ensembles. The point is an input-only ablation before D7.

## Build Graph Repo V2

Smoke build:

```bash
python scripts/build_graph_repo_d6d_input_v2.py --config configs/experiments/d6d_input_v2_build.yaml --output_dir artifacts/graph_repo_d6d_input_v2_smoke --max_samples_per_split 32
```

Full build:

```bash
python scripts/build_graph_repo_d6d_input_v2.py --config configs/experiments/d6d_input_v2_build.yaml --output_dir artifacts/graph_repo_d6d_input_v2
```

The v2 repo is separate from the original graph repo and writes metadata for `node_feature_names`, `edge_feature_names`, `node_dim`, `edge_dim`, `height`, and `width`.

## Train Configs

Primary comparison config:

```bash
configs/experiments/d6d_input_v2_d6b_border075.yaml
```

It uses D6B border075 with `model.node_dim: 12`, `model.edge_dim: 5`, and `paths.graph_repo_path: artifacts/graph_repo_d6d_input_v2`.

Optional interpretability config, created but not intended for immediate full run:

```bash
configs/experiments/d6d_input_v2_d6c_light.yaml
```

## Smoke Commands

Audit smoke:

```bash
python scripts/audit_graph_input_quality.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --graph_repo_path artifacts/graph_repo_d6d_input_v2_smoke --max_samples_per_split 32 --output_dir output/graph_input_audit_d6d_v2_smoke
```

Train smoke:

```bash
python scripts/train_d5a.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --graph_repo_path artifacts/graph_repo_d6d_input_v2_smoke --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

Eval smoke:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --graph_repo_path artifacts/graph_repo_d6d_input_v2_smoke --checkpoint output/d6d_input_v2_d6b_border075/checkpoints/best.pth --max_test_batches 1
```

Visualize smoke:

```bash
python scripts/visualize_d6.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --graph_repo_path artifacts/graph_repo_d6d_input_v2_smoke --checkpoint output/d6d_input_v2_d6b_border075/checkpoints/best.pth --max_samples 4
```

## Full Run Commands

```bash
python scripts/audit_graph_input_quality.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --graph_repo_path artifacts/graph_repo_d6d_input_v2 --output_dir output/graph_input_audit_d6d_v2
python scripts/train_d5a.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml
python scripts/evaluate_d5a.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --checkpoint output/d6d_input_v2_d6b_border075/checkpoints/best.pth
python scripts/visualize_d6.py --config configs/experiments/d6d_input_v2_d6b_border075.yaml --checkpoint output/d6d_input_v2_d6b_border075/checkpoints/best.pth --max_samples 32
```

## Success Criteria

D6D-input-v2 is successful if macro F1 is at least 0.5359, or does not drop by more than 0.005, with accuracy and weighted F1 close to D6B border075. Watch the hard classes: Fear should stay at or above 0.37, Sad should not fall below 0.44, and Disgust should not fall below 0.48. Slot/motif behavior, border mass, and NaN/Inf stability must remain clean.

Decision rule:

- If node_dim=12 improves score, use it as the base for D7 Graph-Swin.
- If it is score-neutral but motif quality is cleaner, keep it as an option.
- If it drops clearly, keep node_dim=7 for D7.
