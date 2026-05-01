# D7 Dual-Branch Graph-Swin Motif Summary

## Why D7

D6B border075 remains the main checkpoint because it has the best score balance so far:
accuracy 0.5606, macro F1 0.5359, weighted F1 0.5569. D6C helped interpretability and class-attention separation but did not improve the main score. D6D-input-v2 with node_dim=12 produced clean inputs and some class gains, but it did not beat D6B and hurt Sad/Surprise.

D7 therefore tests architecture, not new input features or heavier objectives: keep the D6B full-pixel motif branch and add a learned window/region hierarchy inspired by Swin.

## Default Input

D7 defaults to the old graph repo with `node_dim=7`. The node_dim=12 D6D input is kept as a later ablation only, because its main D6B/D6C runs did not improve the current checkpoint.

## Difference From D6B

D6B has one branch:

`pixel graph encoder -> pixel-to-part slots -> part attention -> class-to-part attention -> logits`

D7 adds:

`shared pixel graph encoder -> D6B motif branch + Graph-Swin window/region branch -> fusion`

The shared encoder still uses edge-aware pixel message passing over the full 48x48 graph. No CNN, pretrained Swin, dense 2304x2304 tensors, landmarks, CLIP anchors, or semantic priors are used.

## Difference From GNN.py-Style References

D7 is not a generic graph classifier over pooled graph features. It keeps full pixel-level graph learning, keeps D6 soft motif slots, and adds a second branch that groups pixel embeddings into regular and shifted local windows before class-level region attention.

## Branch 1: D6 Motif

The D6 branch keeps:

- pixel-to-part attention over 16 slots
- part feature pooling and part position encoding
- part-to-part self attention
- class-to-part attention
- `class_repr_d6`, `logits_d6`, `part_masks`, `class_part_attn`

The D6B slot regularizers are reused for D7B/D7C.

## Branch 2: Graph-Swin

The Graph-Swin branch receives `h_pixel [B, 2304, H]` from the shared encoder and reshapes it to `[B, 48, 48, H]`.

Window flow:

- regular 6x6 windows produce 64 window token groups
- shifted windows use `torch.roll(..., shifts=(-3, -3))`
- each window is encoded with lightweight attention pooling by default
- regular and shifted window tokens are concatenated and projected
- 2x2 window merge maps the 8x8 window grid to 4x4 region tokens
- class-to-region attention creates `class_repr_swin` and `logits_swin`

Default outputs include `region_tokens [B,16,H]` and `class_region_attn [B,7,16]`.

## Fusion Modes

D7A `swin_only`:

- runs shared encoder and Graph-Swin branch
- main logits are `logits_swin`
- no D6 branch or slot regularizers are required

D7B `logits_sum`:

- runs D6 and Graph-Swin branches
- main logits are `logits_d6 + logits_swin`
- uses auxiliary CE for both branches

D7C `gated_class_repr`:

- runs both branches
- builds gate input from `[class_repr_d6, class_repr_swin, abs(diff), product]`
- default gate shape is `[B,7,1]`
- fused representation is `gate * D6 + (1 - gate) * Swin`
- class-wise head produces `logits_fused`

## Loss

Loss name: `d7_dual_branch_motif`.

Default D7B/D7C:

`CE(logits) + 0.2 CE(logits_d6) + 0.2 CE(logits_swin) + D6B slot regularizers`

Regularizers:

- `lambda_slot_div: 0.01`
- `lambda_border: 0.0075`
- `lambda_slot_balance: 0.01`
- `lambda_slot_smooth: 0.0`

D7A sets auxiliary and slot losses to zero. Missing optional branch outputs are handled safely.

## Diagnostics And Visualization

Model/training diagnostics include:

- fusion gate mean and per-class gate
- Swin class-region entropy
- region token norm
- branch auxiliary accuracies from the loss when branch logits exist
- existing D6 class-part and slot diagnostics when D6 branch runs

`scripts/visualize_d7.py` writes:

- `class_region_attention_grid.png`
- `class_region_attention_avg_by_true_class.png`
- `region_attention_maps.png`
- `region_token_grid.png`
- `fusion_gate_by_class.png`
- `fusion_gate_by_sample.png`
- `fusion_gate_by_class.csv`
- `class_region_attention_entropy.csv`
- `top_regions_per_class.csv`

The existing `scripts/visualize_d6.py` remains unchanged for D6 part/motif visualizations.

## Smoke Commands

```bash
python scripts/debug_d7_forward.py

python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_standalone.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb

python scripts/train_d5a.py --config configs/experiments/d7c_dual_branch_gated_fusion.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb

python scripts/evaluate_d5a.py --config configs/experiments/d7c_dual_branch_gated_fusion.yaml --checkpoint output/d7c_dual_branch_gated_fusion/checkpoints/best.pth --max_test_batches 1

python scripts/visualize_d7.py --config configs/experiments/d7c_dual_branch_gated_fusion.yaml --checkpoint output/d7c_dual_branch_gated_fusion/checkpoints/best.pth --max_samples 4

python scripts/train_d5a.py --config configs/experiments/d6b_class_part_graph_motif_border075.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
```

## Full Run Order

```bash
python scripts/train_d5a.py --config configs/experiments/d7a_graph_swin_standalone.yaml
python scripts/train_d5a.py --config configs/experiments/d7b_dual_branch_logits_fusion.yaml
python scripts/train_d5a.py --config configs/experiments/d7c_dual_branch_gated_fusion.yaml
```

Evaluate and visualize:

```bash
python scripts/evaluate_d5a.py --config <config> --checkpoint output/<name>/checkpoints/best.pth
python scripts/visualize_d7.py --config <config> --checkpoint output/<name>/checkpoints/best.pth --max_samples 32
```

## Success Criteria

D7A is useful if macro F1 reaches at least 0.50, accuracy reaches at least 0.52, predictions do not collapse to Happy/Neutral, and class-region attention is not uniform.

D7B/D7C should beat D6B border075 macro F1 0.5359, keep accuracy around or above 0.5606, keep weighted F1 near or above 0.5569, avoid class regressions, and show real Swin contribution through non-collapsed gates and class-dependent region attention.

## Risks

- Graph-Swin branch may be weak as a standalone learner.
- Fusion gate can collapse to one branch for all classes.
- Dual-branch capacity can overfit.
- Compute increases because D6 and Graph-Swin both run in D7B/D7C.
- Motif interpretation can become harder if fused logits hide branch behavior.
