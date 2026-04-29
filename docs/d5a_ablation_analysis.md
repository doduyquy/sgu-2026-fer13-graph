# D5A Ablation Analysis

Phân tích này đọc số liệu từ:

- `output/d5a_ce_only`
- `output/d5a_ce_contrast_light`
- `output/d5a_node_score_only`
- `output/d5a` để đối chiếu D5A original

Nguồn ưu tiên: `evaluation/metrics.json`, `evaluation/classification_report.json`, `evaluation/predictions.csv`, `training_history.json`, `resolved_config.yaml`, `wandb-summary.json`, `wandb/.../output.log`, các figure trong `figures/`.

Lưu ý quan trọng: 3 ablation chỉ train 10 epoch và đều best ở epoch 9. D5A original trong `output/d5a` train tối đa 80 epoch, best ở epoch 21, final epoch 41. Vì vậy so sánh original với ablation là có giá trị định hướng, nhưng không hoàn toàn là controlled comparison theo cùng số epoch. Tuy nhiên ở epoch 9, original đã đạt `val_macro_f1=0.2647`, vẫn cao hơn cả 3 ablation 10-epoch.

## I. Tóm tắt kết quả 3 ablation

### 1. `d5a_ce_only`

- Config chính: `lambda_cls=1.0`, `lambda_contrast=0.0`, `lambda_smooth=0.0`, `lambda_closure=0.0`, `lambda_area=0.0`, `edge_score_weight=0.5`.
- `best_epoch=9`
- `best_val_macro_f1=0.2196`
- Test:
  - `accuracy=0.3344`
  - `macro_f1=0.2139`
  - `weighted_f1=0.2688`
- `pred_count=[2, 0, 94, 2248, 531, 191, 523]`
- Per-class test:

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Angry | 0.5000 | 0.0020 | 0.0041 | 491 |
| Disgust | 0.0000 | 0.0000 | 0.0000 | 55 |
| Fear | 0.2979 | 0.0530 | 0.0900 | 528 |
| Happy | 0.3243 | 0.8294 | 0.4663 | 879 |
| Sad | 0.2994 | 0.2677 | 0.2827 | 594 |
| Surprise | 0.5445 | 0.2500 | 0.3427 | 416 |
| Neutral | 0.3423 | 0.2859 | 0.3116 | 626 |

- Disgust vẫn chết hoàn toàn: `pred_count=0`, recall/f1 = 0.
- Fear rất yếu: chỉ 94 prediction, recall 0.0530, f1 0.0900.
- Happy bị predict quá nhiều: 2248/3589 mẫu, khoảng 62.6%.
- AMP/non-finite grad: có warning skip optimizer ở epoch 7 batch 553, epoch 9 batch 661, epoch 10 batch 272.
- Curve: train macro F1 tăng từ 0.0774 lên 0.1926; val macro F1 tăng đến 0.2196 ở epoch 9 rồi tụt còn 0.1812 ở epoch 10. Loss train/val vẫn giảm nhẹ, nên chưa giống overfit cổ điển; vấn đề chính là prediction bias/collapse.

### 2. `d5a_ce_contrast_light`

- Config chính: `lambda_cls=1.0`, `lambda_contrast=0.05`, `lambda_smooth=0.0`, `lambda_closure=0.0`, `lambda_area=0.0`, `edge_score_weight=0.5`.
- `best_epoch=9`
- `best_val_macro_f1=0.2491`
- Test:
  - `accuracy=0.3700`
  - `macro_f1=0.2561`
  - `weighted_f1=0.3155`
- `pred_count=[26, 0, 131, 1820, 626, 312, 674]`
- Per-class test:

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Angry | 0.5385 | 0.0285 | 0.0542 | 491 |
| Disgust | 0.0000 | 0.0000 | 0.0000 | 55 |
| Fear | 0.3206 | 0.0795 | 0.1275 | 528 |
| Happy | 0.3824 | 0.7918 | 0.5157 | 879 |
| Sad | 0.3019 | 0.3182 | 0.3098 | 594 |
| Surprise | 0.5032 | 0.3774 | 0.4313 | 416 |
| Neutral | 0.3412 | 0.3674 | 0.3538 | 626 |

- Đây là ablation tốt nhất trong 3 ablation.
- Disgust vẫn chết hoàn toàn: `pred_count=0`, recall/f1 = 0.
- Fear cải thiện so với CE-only và node-score-only, nhưng vẫn thấp: f1 0.1275.
- Happy bias giảm so với CE-only/node-score-only nhưng vẫn nặng: 1820/3589 mẫu, khoảng 50.7%.
- AMP/non-finite grad: có warning skip optimizer ở epoch 7 batch 505, epoch 9 batch 655 và 842.
- Curve: train macro F1 tăng đều đến 0.2344; val macro F1 đạt 0.2491 ở epoch 9 rồi tụt 0.2193 ở epoch 10. Không thấy overfit mạnh; giống under-separation/prediction bias hơn.

### 3. `d5a_node_score_only`

- Config chính: `lambda_cls=1.0`, `lambda_contrast=0.05`, `lambda_smooth=0.0`, `lambda_closure=0.0`, `lambda_area=0.0`, `edge_score_weight=0.0`.
- `best_epoch=9`
- `best_val_macro_f1=0.2152`
- Test:
  - `accuracy=0.3213`
  - `macro_f1=0.2076`
  - `weighted_f1=0.2577`
- `pred_count=[12, 0, 142, 2195, 387, 452, 401]`
- Per-class test:

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Angry | 0.5000 | 0.0122 | 0.0239 | 491 |
| Disgust | 0.0000 | 0.0000 | 0.0000 | 55 |
| Fear | 0.2746 | 0.0739 | 0.1164 | 528 |
| Happy | 0.3207 | 0.8009 | 0.4580 | 879 |
| Sad | 0.2920 | 0.1902 | 0.2304 | 594 |
| Surprise | 0.3606 | 0.3918 | 0.3756 | 416 |
| Neutral | 0.3192 | 0.2045 | 0.2493 | 626 |

- Node-only không tốt hơn bản có edge; đây là tín hiệu chống lại giả thuyết edge score là nguyên nhân chính gây hại.
- Disgust vẫn chết hoàn toàn.
- Fear thấp hơn CE+contrast-light, cao hơn CE-only một chút theo f1.
- Happy vẫn bị predict quá nhiều: 2195/3589 mẫu, khoảng 61.2%.
- AMP/non-finite grad: có warning skip optimizer ở epoch 5 batch 742, epoch 8 batch 189, epoch 10 batch 272.
- Curve: train macro F1 tăng đến 0.1894; val macro F1 đạt 0.2152 ở epoch 9 rồi tụt 0.1782 ở epoch 10. Underfit/bias rõ hơn overfit.

## II. Bảng so sánh metric

| run | best_epoch | best_val_macro_f1 | test_accuracy | test_macro_f1 | test_weighted_f1 | Angry F1 | Disgust F1 | Fear F1 | Happy F1 | Happy pred_count | Disgust pred_count | nhận xét ngắn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D5A original | 21 | 0.2954 | 0.4054 | 0.2927 | 0.3587 | 0.1342 | 0.0000 | 0.1591 | 0.5738 | 1683 | 0 | Tốt nhất tổng thể; vẫn chết Disgust và bias Happy. |
| `d5a_ce_only` | 9 | 0.2196 | 0.3344 | 0.2139 | 0.2688 | 0.0041 | 0.0000 | 0.0900 | 0.4663 | 2248 | 0 | Bỏ aux làm kết quả giảm mạnh; Happy collapse nặng nhất. |
| `d5a_ce_contrast_light` | 9 | 0.2491 | 0.3700 | 0.2561 | 0.3155 | 0.0542 | 0.0000 | 0.1275 | 0.5157 | 1820 | 0 | Tốt nhất trong 3 ablation; contrast nhẹ giúp cân bằng hơn. |
| `d5a_node_score_only` | 9 | 0.2152 | 0.3213 | 0.2076 | 0.2577 | 0.0239 | 0.0000 | 0.1164 | 0.4580 | 2195 | 0 | Tắt edge score không giúp; node-only yếu. |

So sánh cùng mốc epoch 9 trên validation:

| run | val_macro_f1 epoch 9 |
|---|---:|
| D5A original | 0.2647 |
| `d5a_ce_only` | 0.2196 |
| `d5a_ce_contrast_light` | 0.2491 |
| `d5a_node_score_only` | 0.2152 |

Điều này làm kết luận "auxiliary losses đang hại model" không được ủng hộ bởi số hiện có.

## III. Phân tích từng ablation

### CE-only

CE-only là phép thử quan trọng nhất cho câu hỏi aux loss có bó motif quá sớm không. Nếu đúng như giả thuyết này, CE-only phải tốt hơn hoặc ít nhất xấp xỉ original. Thực tế ngược lại:

- Test macro F1 chỉ 0.2139, thấp hơn original 0.2927.
- Val macro F1 best 0.2196, thấp hơn original ở epoch 9 là 0.2647.
- Prediction phân bố kém hơn: Happy tăng từ 1683 ở original lên 2248; Angry gần như biến mất, chỉ 2 prediction.

Kết luận cục bộ: bỏ toàn bộ aux loss làm motif/logit kém hơn, không phải tốt hơn.

### CE + contrast light

Contrast nhẹ là ablation tốt nhất:

- Test macro F1 tăng từ 0.2139 lên 0.2561 so với CE-only.
- Test accuracy tăng từ 0.3344 lên 0.3700.
- Happy pred_count giảm từ 2248 xuống 1820.
- Fear f1 tăng từ 0.0900 lên 0.1275.
- Surprise và Neutral cũng tăng rõ.

Nhưng nó vẫn kém original:

- Test macro F1 0.2561 < 0.2927.
- Best val macro F1 0.2491 < 0.2954 của original, và cũng < 0.2647 của original ở epoch 9.
- Disgust vẫn hoàn toàn không được predict.

Kết luận cục bộ: contrast nhẹ giúp class separation so với CE-only, nhưng chưa giải quyết collapse class nhỏ và chưa thay thế được full D5A loss.

### Node-score-only

Node-score-only tắt đóng góp edge score bằng `edge_score_weight=0.0`. Kỳ vọng nếu edge motif gây nhiễu thì node-only phải tốt hơn. Thực tế:

- Test macro F1 0.2076, thấp nhất trong 3 ablation.
- Best val macro F1 0.2152, cũng thấp nhất.
- Happy pred_count 2195, gần CE-only và tệ hơn CE+contrast-light.
- Sad/Neutral giảm mạnh so với CE+contrast-light.
- `class_edge_gate` giữ nguyên đúng kỳ vọng: mean=min=max=0.5 trong history và wandb summary.

Kết luận cục bộ: số hiện tại không ủng hộ giả thuyết edge motif là nguồn nhiễu chính. Edge score/edge attention có vẻ vẫn đóng góp thông tin, hoặc ít nhất tắt edge không làm model tốt hơn.

## IV. Kết luận nguyên nhân D5A yếu

### A. Auxiliary losses có đang làm hại model không?

Không có bằng chứng ủng hộ. CE-only kém hơn rõ:

- `d5a_ce_only` test macro F1 0.2139 < original 0.2927.
- `d5a_ce_only` best val macro F1 0.2196 < original epoch 9 val macro F1 0.2647.
- CE-only làm Happy bias nặng hơn: 2248 prediction Happy so với 1683 của original.

Vì vậy aux loss không phải nguyên nhân chính làm D5A yếu. Ngược lại, full D5A loss có vẻ giúp giảm bias và cải thiện macro F1.

### B. Contrast nhẹ có giúp không?

Có, trong phạm vi ablation:

- CE+contrast-light tốt hơn CE-only trên val macro F1, test accuracy, test macro F1, weighted F1.
- Fear, Happy, Sad, Surprise, Neutral đều tốt hơn CE-only.
- Happy bias giảm từ 2248 xuống 1820 prediction.

Nhưng contrast nhẹ vẫn chưa đủ:

- Disgust vẫn `pred_count=0`.
- Fear vẫn thấp.
- CE+contrast-light vẫn kém original.

Kết luận: contrast nhẹ giúp class separation, nhưng signal motif/class score vẫn yếu.

### C. Edge motif có gây nhiễu không?

Không có bằng chứng rõ rằng edge motif gây nhiễu. Node-score-only là run kém nhất:

- Node-score-only test macro F1 0.2076 < CE+contrast-light 0.2561.
- Node-score-only best val macro F1 0.2152 < CE+contrast-light 0.2491.
- Node-score-only Happy bias nặng hơn CE+contrast-light.

Kết luận: không nên tạm bỏ edge score chỉ dựa trên 3 ablation này. Nếu muốn giảm edge, nên giảm nhẹ/tune `edge_score_weight`, không phải kết luận edge là thủ phạm chính.

### D. Vấn đề chính có phải motif random khó học không?

Có khả năng cao. Các dấu hiệu:

- Cả 3 ablation đều thấp hơn original và đều thấp tuyệt đối: test macro F1 chỉ 0.2076-0.2561.
- Disgust luôn chết hoàn toàn ở cả original và 3 ablation: `pred_count=0`, f1=0.
- Fear luôn yếu: f1 chỉ 0.0900-0.1591.
- Happy luôn bị predict quá nhiều: 1683-2248 prediction trên 3589 mẫu.
- Gate/attention có học hình dạng không hoàn toàn ngẫu nhiên, nhưng việc class nhỏ không được predict cho thấy motif học từ label ảnh đơn thuần chưa đủ discriminative.

Kết luận chính: D5A random motif có học được một phần signal, nhưng không đủ mạnh để tự thoát khỏi class imbalance và Happy bias. Nên chuyển trọng tâm sang D5B motif prior thay vì tiếp tục mở rộng D5A random.

## V. Phân tích class collapse / prediction bias

### Prediction bias theo run

| run | pred_count Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral | Happy share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D5A original | 120 | 0 | 176 | 1683 | 666 | 344 | 600 | 46.9% |
| `d5a_ce_only` | 2 | 0 | 94 | 2248 | 531 | 191 | 523 | 62.6% |
| `d5a_ce_contrast_light` | 26 | 0 | 131 | 1820 | 626 | 312 | 674 | 50.7% |
| `d5a_node_score_only` | 12 | 0 | 142 | 2195 | 387 | 452 | 401 | 61.2% |

### Confusion nổi bật

Từ confusion matrix/predictions:

- `d5a_ce_only`: Happy là sink chính. Angry gần như không được predict. Disgust thật chủ yếu bị đẩy sang Happy.
- `d5a_ce_contrast_light`: bias Happy giảm tốt nhất trong ablation, nhưng vẫn rất lớn:
  - Angry -> Happy: 227
  - Disgust -> Happy: 35/55
  - Fear -> Happy: 222
  - Sad -> Happy: 247
  - Neutral -> Happy: 264
- `d5a_node_score_only`: Happy lại tăng:
  - Angry -> Happy: 294
  - Disgust -> Happy: 39/55
  - Fear -> Happy: 296
  - Sad -> Happy: 335
  - Neutral -> Happy: 359

Macro F1 thấp chủ yếu do:

- Disgust f1 = 0 ở mọi run.
- Angry rất thấp ở ablation, đặc biệt CE-only f1 0.0041.
- Fear recall thấp.
- Happy có recall cao nhưng precision thấp do bị over-predict.

## VI. Attention/class gate diagnostic

Các ablation đều có:

- 16 figure `figures/d5a_attention/all_class_grids`
- 16 figure `figures/d5a_attention/top_edges`
- 16 figure `figures/d5a_attention/true_pred`
- 7 figure `figures/d5a_class_gates`

Không thấy file diagnostic dạng JSON riêng cho 3 ablation; chỉ có history metrics và PNG.

Quan sát từ history:

| run | final class_node_gate mean/min/max | final class_edge_gate mean/min/max | ghi chú |
|---|---|---|---|
| `d5a_ce_only` | 0.4714 / 0.1284 / 0.9494 | 0.4645 / 0.1332 / 0.9257 | Gate có phân hóa, không đứng yên ở 0.5. |
| `d5a_ce_contrast_light` | 0.4775 / 0.1390 / 0.9363 | 0.4674 / 0.1395 / 0.9272 | Gate có phân hóa tương tự CE-only. |
| `d5a_node_score_only` | 0.4739 / 0.0627 / 0.9442 | 0.5000 / 0.5000 / 0.5000 | Edge gate đứng yên đúng kỳ vọng do `edge_score_weight=0`. |

Quan sát trực quan nhanh:

- Class gates không hoàn toàn đều; có hình dạng vùng mặt, đặc biệt Happy gate có vùng miệng rất mạnh.
- Disgust gate cũng có phân bố không phẳng, nhưng vẫn không tạo prediction Disgust. Nghĩa là gate có học pattern nào đó, nhưng class score/logit chưa discriminative đủ.
- Attention sample có vùng tập trung quanh mắt/mũi/miệng, nhưng cũng có nhiễu biên/nền và khác class chưa đủ ổn định để cứu class nhỏ.
- Với node-score-only, edge gate đứng yên 0.5 là đúng với config, không phải lỗi logging.

## VII. Quyết định tiếp theo

Chọn **Hướng 4**:

> Dừng cải D5A random theo hướng ablation nhỏ; chuyển sang D5B: Offline Discriminative Full-Graph Motif Prior + End-to-End Fine-tuning.

Lý do:

- Không ablation nào vượt original.
- CE-only không chứng minh aux loss gây hại.
- Contrast nhẹ giúp nhưng chưa đủ.
- Node-score-only không chứng minh edge là nguyên nhân.
- Class collapse gốc vẫn còn: Disgust chết, Fear yếu, Happy bias.
- D5A original tốt nhất nhưng vẫn macro F1 chỉ khoảng 0.293, nghĩa là random motif vẫn thiếu prior discriminative.

Không nên train dài D5A gốc thêm chỉ dựa trên kết quả này. Original đã train đến best epoch 21 và early-stop/final epoch 41; vấn đề còn lại là chất lượng motif/class separation, không phải thiếu vài epoch ở ablation.

## VIII. Nếu đi D5B: thiết kế D5B-node-prior tối giản

Bản nên triển khai trước là **D5B-node-prior only**.

### Mục tiêu

Tạo `node_prior [7, 2304]` offline từ train set, class-specific và discriminative, dùng để khởi tạo hoặc regularize `class_node_gate`.

### Thiết kế tối giản

1. Tính thống kê node theo class từ train set:
   - Với mỗi class `c`, lấy trung bình node/pixel feature hoặc intensity map trên toàn bộ ảnh train thuộc class `c`.
   - Chuẩn hóa từng class để prior có scale ổn định.

2. Làm prior discriminative:
   - Không chỉ dùng mean class thô.
   - Dùng dạng one-vs-rest, ví dụ `prior_c = normalize(mean_c - mean_not_c)` hoặc một biến thể có clamp/sigmoid để giữ trong [0, 1].
   - Có thể cân bằng theo class count để Disgust không bị nuốt bởi prior của class lớn.

3. Init model:
   - `class_node_gate` init từ `node_prior [7, 2304]`.
   - Không cần edge prior ở bước đầu.
   - `edge_score_weight=0` hoặc rất nhỏ, ví dụ 0.05-0.1, để kiểm tra node prior trước.

4. Loss:
   - Dùng `CE + contrast light`.
   - Khởi điểm hợp lý: `lambda_cls=1.0`, `lambda_contrast=0.05`.
   - Có thể thêm prior regularization nhẹ:
     - giữ `class_node_gate` không trôi quá xa prior trong vài epoch đầu;
     - hoặc decay regularization theo epoch.

5. Visualization bắt buộc:
   - Lưu `node_prior` từng class trước fine-tune.
   - Lưu `class_node_gate` sau fine-tune.
   - So sánh prior vs learned gate, đặc biệt Disgust/Fear/Happy.

6. Metric cần theo dõi:
   - `pred_count` từng class mỗi epoch.
   - Disgust recall/f1.
   - Fear recall/f1.
   - Happy pred_count và precision.
   - `val_macro_f1` làm monitor chính.

### Vì sao chưa nên làm edge_prior ngay?

Node-score-only không thắng, nên không thể kết luận edge vô ích. Nhưng edge prior trên 17860 edge sẽ tăng độ phức tạp và khó debug. D5B-node-prior giúp kiểm tra câu hỏi cốt lõi trước: prior class-specific trên full pixel grid có phá được Happy bias và cứu Disgust/Fear không.

## IX. Những file/code cần kiểm tra thêm nếu thiếu dữ liệu

Không thiếu các file metric chính cho 3 ablation. Tuy nhiên nếu muốn kết luận chắc hơn trước khi code D5B, nên kiểm tra thêm:

- `output/d5a_*` chưa có diagnostic report dạng `.md`/`.json` như original; có thể tạo script đọc tự động gate stats, pred entropy, confusion notes cho mọi run.
- Chưa có số định lượng similarity giữa class gates. Nên tính cosine similarity/correlation giữa `class_node_gate` các class nếu checkpoint dễ load.
- Chưa có attention aggregate theo toàn test set. Figure hiện tại chỉ 16 sample, đủ để xem nhanh nhưng chưa đủ kết luận thống kê.
- `wandb-summary.json` chỉ phản ánh epoch cuối, không phải best epoch; khi tổng hợp phải luôn lấy best từ `training_history.json`.
- Original không có wandb log trong thư mục `output/d5a`, nên không xác nhận được AMP/non-finite warning của original từ local files.

