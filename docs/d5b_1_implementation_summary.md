# D5B-1 Implementation Summary

## D5B-1 làm gì

D5B-1 là pipeline offline prior + fixed classifier cho FER-2013 full pixel graph.

Luồng end-to-end:

```text
graph_repo train
-> build node_prior [7, 2304]
-> freeze node_prior
-> train FixedMotifMLPClassifier
-> evaluate test split
```

Motif prior được build một lần từ train split và được lưu ở:

```text
artifacts/d5b_motif_prior/node_prior.pt
```

Trong quá trình train classifier, `node_prior` là buffer cố định, không có gradient và không update.

## Khác D5A thế nào

D5A học motif từ random trong quá trình train bằng class-level soft graph retrieval, gồm node/edge attention và các regularizer retrieval.

D5B-1 không gắn vào D5A. D5B-1 không dùng edge motif, không init `class_node_gate`, và không cho motif update. Classifier chỉ đọc full node features `[2304, 7]`, pooling theo fixed class prior `[7, 2304]`, rồi train MLP bằng cross entropy.

D5B-2 sau này mới dùng `node_prior` để init D5A và cho motif update.

## Công thức prior

Với mỗi sample, activation node được tính từ node feature:

```text
act = 0.2 * intensity + 0.4 * grad_mag + 0.4 * local_contrast
```

Không dùng `x_norm` hoặc `y_norm` để tính activation.

Với mỗi class `c` và node `v`:

```text
effect_c(v) = max((mean_c(v) - mean_rest(v)) / (std_all(v) + eps), 0)
fisher_c(v) = (mean_c(v) - mean_rest(v))^2 / (var_c(v) + var_rest(v) + eps)
support_c(v) = P(act(v) > percentile_75_per_node | y=c)
commonness(v) = mean_k support_k(v)
```

`effect` và `fisher` được normalize per class về `[0, 1]`.

Raw score:

```text
raw_score_c(v) =
    0.45 * effect_c(v)
  + 0.30 * fisher_c(v)
  + 0.20 * support_c(v)
  - 0.15 * commonness(v)
```

Sau đó raw score được clamp `>= 0`, reshape `[48, 48]`, smooth bằng kernel 3x3, normalize per class về `[0, 1]`, rồi clamp final prior vào `[0.05, 0.95]`.

## Classifier hoạt động thế nào

Model: `models/fixed_motif_classifier.py::FixedMotifMLPClassifier`

Input:

```text
x [B, 2304, 7]
node_prior [7, 2304]
```

Với mỗi class motif, model tính:

```text
weighted_mean
weighted_std
weighted_max
weighted_energy
```

Mỗi motif tạo feature `[node_dim * 4] = [28]`. Bảy motif tạo feature `[7, 28]`, flatten thành `[196]`, rồi đi qua MLP:

```text
Linear(196, 256)
LayerNorm(256)
GELU
Dropout(0.2)
Linear(256, 128)
GELU
Dropout(0.2)
Linear(128, 7)
```

Forward return:

```python
{
    "logits": logits,
    "motif_features": motif_features,
}
```

## Cách chạy

Build prior:

```bash
python scripts/build_d5b_motif_prior.py --config configs/experiments/d5b_1_fixed_motif_classifier.yaml
```

Train classifier:

```bash
python scripts/train_d5b_fixed_motif.py --config configs/experiments/d5b_1_fixed_motif_classifier.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5b_fixed_motif.py --config configs/experiments/d5b_1_fixed_motif_classifier.yaml --checkpoint output/d5b_1_fixed_motif_classifier/checkpoints/best.pth
```

## Output cần kiểm tra

Prior artifacts:

```text
artifacts/d5b_motif_prior/node_prior.pt
artifacts/d5b_motif_prior/node_prior_meta.json
artifacts/d5b_motif_prior/figures/class_node_prior_0_angry.png
artifacts/d5b_motif_prior/figures/class_node_prior_1_disgust.png
artifacts/d5b_motif_prior/figures/class_node_prior_2_fear.png
artifacts/d5b_motif_prior/figures/class_node_prior_3_happy.png
artifacts/d5b_motif_prior/figures/class_node_prior_4_sad.png
artifacts/d5b_motif_prior/figures/class_node_prior_5_surprise.png
artifacts/d5b_motif_prior/figures/class_node_prior_6_neutral.png
artifacts/d5b_motif_prior/figures/class_node_prior_grid.png
```

Training outputs:

```text
output/d5b_1_fixed_motif_classifier/checkpoints/best.pth
output/d5b_1_fixed_motif_classifier/checkpoints/last.pth
output/d5b_1_fixed_motif_classifier/training_history.json
output/d5b_1_fixed_motif_classifier/resolved_config.yaml
```

Evaluation outputs:

```text
output/d5b_1_fixed_motif_classifier/evaluation/metrics.json
output/d5b_1_fixed_motif_classifier/evaluation/classification_report.json
output/d5b_1_fixed_motif_classifier/evaluation/classification_report.txt
output/d5b_1_fixed_motif_classifier/evaluation/predictions.csv
output/d5b_1_fixed_motif_classifier/evaluation/confusion_matrix.png
```

## Tiêu chí thành công

So với D5A baseline hiện tại:

```text
test_macro_f1 > 0.2927
Disgust pred_count > 0
Fear F1 > 0.1591
Happy pred_count < 1683
```
