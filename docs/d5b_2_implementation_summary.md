# D5B-2 Implementation Summary

## D5B-2 là gì

D5B-2 là Prior-Initialized D5A Fine-tuning.

Pipeline:

```text
D5B-1 node_prior [7,2304]
-> init D5A class_node_gate_logits
-> train D5A normally with learnable gate
-> evaluate and visualize final motifs
```

D5B-2 kiểm tra liệu D5A khởi tạo motif bằng offline discriminative prior có cải thiện macro F1, giảm Happy bias, và cứu Disgust/Fear tốt hơn random init hay không.

## Khác D5A và D5B-1 thế nào

D5A original khởi tạo `class_node_gate_logits` bằng zero, nên `sigmoid(gate_logits)` bắt đầu quanh `0.5`.

D5B-1 dùng `node_prior` fixed và train một MLP classifier riêng. Motif không update.

D5B-2 dùng lại đúng D5A model/trainer/loss/evaluator/visualizer. Khác biệt chính là `class_node_gate_logits` được init từ `node_prior`, rồi vẫn là `nn.Parameter` learnable và được update bằng backprop.

## Cách load/init node_prior

Config bật:

```yaml
model:
  motif_prior_path: artifacts/d5b_motif_prior/node_prior.pt
  init_node_gate_from_prior: true
  prior_init_clamp_min: 0.05
  prior_init_clamp_max: 0.95
```

Khi model init:

```text
load node_prior.pt
read payload["node_prior"] [7,2304]
clamp prior vào [0.05,0.95]
logit_prior = log(prior / (1 - prior))
copy logit_prior vào class_node_gate_logits
```

`class_node_gate_logits.requires_grad` vẫn là `True`. Prior được giữ thêm trong buffer `motif_node_prior` để kiểm tra và dùng cho optional prior regularization.

## Prior có cập nhật trong train không

`node_prior` gốc không update. Nó chỉ là giá trị khởi tạo và buffer tham chiếu.

Parameter update trong train là:

```text
class_node_gate_logits
```

Do đó final gate có thể drift khỏi prior nếu gradient thấy cần.

## Configs

D5B-2A: init từ prior, không regularize prior.

```text
configs/experiments/d5b_2_prior_init_d5a.yaml
```

Các điểm chính:

```yaml
model:
  init_node_gate_from_prior: true
  edge_score_weight: 0.0

loss:
  lambda_contrast: 0.05
  lambda_smooth: 0.0
  lambda_closure: 0.0
  lambda_area: 0.0
  lambda_prior: 0.0
```

D5B-2B: giống D5B-2A nhưng giữ gate gần prior nhẹ hơn:

```text
configs/experiments/d5b_2_prior_init_reg_d5a.yaml
```

Khác biệt:

```yaml
loss:
  lambda_prior: 0.01
```

## Lệnh chạy

Train D5B-2A:

```bash
python scripts/train_d5a.py --config configs/experiments/d5b_2_prior_init_d5a.yaml
```

Evaluate:

```bash
python scripts/evaluate_d5a.py --config configs/experiments/d5b_2_prior_init_d5a.yaml --checkpoint output/d5b_2_prior_init_d5a/checkpoints/best.pth
```

Visualize:

```bash
python scripts/visualize_d5.py --config configs/experiments/d5b_2_prior_init_d5a.yaml --checkpoint output/d5b_2_prior_init_d5a/checkpoints/best.pth
```

Hoặc qua run_experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/d5b_2_prior_init_d5a.yaml --mode train
python scripts/run_experiment.py --config configs/experiments/d5b_2_prior_init_d5a.yaml --mode evaluate --checkpoint output/d5b_2_prior_init_d5a/checkpoints/best.pth
python scripts/run_experiment.py --config configs/experiments/d5b_2_prior_init_d5a.yaml --mode visualize --checkpoint output/d5b_2_prior_init_d5a/checkpoints/best.pth
```

## Verification đã chạy

Các kiểm tra cần có trước full run:

```text
config load OK
node_prior.pt exists, shape [7,2304], min/max hợp lệ
sigmoid(class_node_gate_logits) gần node_prior sau init
class_node_gate_logits.requires_grad=True
forward trả logits [B,7], node_attn [B,7,2304], class_node_gate [7,2304]
1-batch train backward OK
class_node_gate_logits có gradient
optimizer step làm gate thay đổi nhẹ
1-batch evaluate tính metrics được
```

## Output cần kiểm tra

```text
output/d5b_2_prior_init_d5a/
  checkpoints/
    best.pth
    last.pth
  evaluation/
    metrics.json
    classification_report.json
    classification_report.txt
    predictions.csv
    confusion_matrix.png
  figures/
    d5a_class_gates/
    d5a_attention/
    prior_vs_final_gate/
  training_history.json
  resolved_config.yaml
```

## Tiêu chí thành công

So với D5A original:

```text
test_accuracy = 0.4054
test_macro_f1 = 0.2927
test_weighted_f1 = 0.3587
pred_count = [120, 0, 176, 1683, 666, 344, 600]
```

So với D5B-1:

```text
test_accuracy = 0.3784
test_macro_f1 = 0.2973
test_weighted_f1 = 0.3567
pred_count = [290, 0, 251, 1097, 554, 387, 1010]
```

D5B-2 thành công nếu:

```text
test_macro_f1 > 0.2973
tốt hơn nữa nếu > 0.30
Disgust pred_count > 0
Fear F1 > 0.1591
Happy pred_count < 1683
Accuracy không giảm quá sâu so với D5A original
```

Nếu D5B-2A không tốt, thử từng thay đổi một:

```text
D5B-2B với lambda_prior=0.01
hoặc edge_score_weight=0.05
```
