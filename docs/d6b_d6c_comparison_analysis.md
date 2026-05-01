# D6B vs D6C Comparison Analysis

## 1. Executive summary

Hai run D6 đều vượt D5A/D5B rất rõ trên FER-2013 full pixel graph. D6B đạt `test_macro_f1=0.5252`, D6C đạt `test_macro_f1=0.5264`; trong khi D5A original khoảng `0.2927` và D5B-1 khoảng `0.2973`. Đây là bước nhảy lớn nhất của project hiện tại.

Kết luận ngắn:

- D6B tốt hơn nhẹ về `accuracy`, `weighted_f1`, `best val_macro_f1`, và phân bố `pred_count` tổng thể.
- D6C tốt hơn nhẹ về `macro_f1`, Disgust/Fear/Happy F1, và đặc biệt tốt hơn về class-part motif separation.
- D6B nên giữ làm numeric baseline chính hiện tại vì ổn định hơn và cân bằng pred_count hơn.
- D6C là hướng nghiên cứu đáng tiếp tục vì class-attended objectives làm class attention bớt đồng dạng rất mạnh, nhưng cần chạy/tune thêm trước khi thay D6B làm baseline chính.

File đã đọc/kiểm tra:

- Có: `resolved_config.yaml`, `training_history.json`, `evaluation/metrics.json`, `evaluation/classification_report.json`, `evaluation/classification_report.txt`, `evaluation/predictions.csv`, confusion/correct/wrong images, slot summary, class-part attention CSV/images, class motif maps, `d6_part_masks/`, `d6_part_attention/`.
- D6C có thêm `class_motif_border_mass_by_class.csv`, `true_class_border_mass_by_class.csv`, `confusion_pair_attention_similarity.csv`.
- Thiếu ở D6B: `class_motif_border_mass_by_class.csv`, `true_class_border_mass_by_class.csv`, `confusion_pair_attention_similarity.csv`. Vì vậy phần class-level border mass của D6B chỉ dùng history diagnostic và visual, không kết luận bằng CSV class motif.

## 2. Metric comparison table

| Metric | D6B border075 long150 | D6C class-attended light | Better |
|---|---:|---:|---|
| best epoch | 130 | 93 | D6C sớm hơn |
| best val_macro_f1 | 0.5231 | 0.5097 | D6B |
| test accuracy | 0.5589 | 0.5581 | D6B rất nhẹ |
| test macro F1 | 0.5252 | 0.5264 | D6C rất nhẹ |
| test weighted F1 | 0.5520 | 0.5487 | D6B |
| Angry F1 | 0.4487 | 0.4339 | D6B |
| Disgust F1 | 0.4559 | 0.4960 | D6C |
| Fear F1 | 0.3664 | 0.3954 | D6C |
| Happy F1 | 0.7748 | 0.7849 | D6C |
| Sad F1 | 0.3951 | 0.3698 | D6B |
| Surprise F1 | 0.6998 | 0.6838 | D6B |
| Neutral F1 | 0.5355 | 0.5208 | D6B |
| pred_count | [396, 81, 449, 933, 555, 490, 685] | [334, 70, 509, 976, 531, 520, 649] | D6B tổng thể gần support hơn |
| val effective_slots | 11.6218 | 12.7535 | D6C |
| eval slot_area_entropy | 2.4612 | 2.5585 | D6C |
| val class_part_entropy_mean | 2.6694 | 4.7361 | khác định nghĩa/log; D6C objective tạo entropy aggregate cao hơn |
| eval class attention entropy avg | ~2.7437 | ~2.4610 | D6C sắc hơn |
| val class attention similarity mean | 0.9419 | 0.5630 | D6C |
| val slot border mass mean | 0.1415 | 0.1519 | D6B nhẹ hơn |
| D6C true class border mass mean | missing | 0.2112 avg from CSV | chỉ có D6C |

Per-class report:

| Run | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| D6B | Angry | 0.5025 | 0.4053 | 0.4487 | 491 |
| D6B | Disgust | 0.3827 | 0.5636 | 0.4559 | 55 |
| D6B | Fear | 0.3987 | 0.3390 | 0.3664 | 528 |
| D6B | Happy | 0.7524 | 0.7986 | 0.7748 | 879 |
| D6B | Sad | 0.4090 | 0.3822 | 0.3951 | 594 |
| D6B | Surprise | 0.6469 | 0.7620 | 0.6998 | 416 |
| D6B | Neutral | 0.5124 | 0.5607 | 0.5355 | 626 |
| D6C | Angry | 0.5359 | 0.3646 | 0.4339 | 491 |
| D6C | Disgust | 0.4429 | 0.5636 | 0.4960 | 55 |
| D6C | Fear | 0.4028 | 0.3883 | 0.3954 | 528 |
| D6C | Happy | 0.7459 | 0.8282 | 0.7849 | 879 |
| D6C | Sad | 0.3917 | 0.3502 | 0.3698 | 594 |
| D6C | Surprise | 0.6154 | 0.7692 | 0.6838 | 416 |
| D6C | Neutral | 0.5116 | 0.5304 | 0.5208 | 626 |

## 3. Per-class analysis

### Angry

- D6B: F1 `0.4487`, pred_count `396`.
- D6C: F1 `0.4339`, pred_count `334`.
- D6B xử lý Angry tốt hơn vì recall cao hơn: `0.4053` vs `0.3646`.
- Lỗi chính:
  - D6B: Angry -> Neutral `77`, Sad `72`, Fear `62`, Happy `48`.
  - D6C: Angry -> Sad `80`, Fear `75`, Neutral `75`, Happy `51`.
- D6C có precision Angry cao hơn, nhưng dự đoán Angry ít hơn nên mất recall.

### Disgust

- D6B: F1 `0.4559`, pred_count `81`.
- D6C: F1 `0.4960`, pred_count `70`.
- Cả hai đều recall `0.5636` với `31/55` mẫu Disgust đúng.
- D6C tốt hơn vì precision tăng từ `0.3827` lên `0.4429`.
- Lỗi chính:
  - D6B: Disgust -> Angry `9`, Happy `4`, Sad `4`, Neutral `3`.
  - D6C: Disgust -> Angry `8`, Sad `5`, Fear/Happy/Neutral `3`.

Disgust hồi phục rất mạnh so với D5 vì D6 không còn bắt class gate pixel phẳng phải tự giải quyết mọi thứ. Soft part slots tạo tầng trung gian để gom các pattern cục bộ, rồi class-part attention chọn tổ hợp part cho từng emotion. Với class nhỏ như Disgust, việc có part-level representation giúp model có "đơn vị biểu đạt" tái sử dụng từ toàn bộ dữ liệu, thay vì cần học riêng một mask pixel-class rất yếu và dễ bị Happy nuốt.

### Fear

- D6B: F1 `0.3664`, pred_count `449`.
- D6C: F1 `0.3954`, pred_count `509`.
- D6C tốt hơn vì recall tăng từ `0.3390` lên `0.3883`.
- Lỗi chính:
  - D6B: Fear -> Sad `106`, Surprise `84`, Neutral `71`, Angry `43`, Happy `40`.
  - D6C: Fear -> Sad `100`, Surprise `78`, Neutral `64`, Happy `42`, Angry `36`.
- Fear vẫn khó do motif còn gần Sad/Surprise. D6C giảm một phần lỗi Fear nhưng chưa tách triệt để.

### Sad

- D6B: F1 `0.3951`, pred_count `555`.
- D6C: F1 `0.3698`, pred_count `531`.
- D6B tốt hơn Sad vì recall `0.3822` vs `0.3502`.
- Lỗi chính:
  - D6B: Sad -> Neutral `130`, Fear `85`, Angry `66`, Happy `53`.
  - D6C: Sad -> Neutral `125`, Fear `101`, Happy `60`, Angry `55`.
- D6C làm Fear mạnh hơn nhưng kéo thêm Sad -> Fear, phù hợp với similarity Fear-Sad còn cao.

### Class còn yếu

Các class yếu nhất theo F1:

- D6B: Fear `0.3664`, Sad `0.3951`, Angry `0.4487`, Disgust `0.4559`.
- D6C: Sad `0.3698`, Fear `0.3954`, Angry `0.4339`, Disgust `0.4960`.

Disgust không còn là class chết. Vấn đề chính hiện tại chuyển sang tách nhóm Fear/Sad/Neutral và Angry/Sad/Fear.

## 4. Slot/part discovery analysis

D6B slot statistics:

- `val_diag_effective_slots=11.6218`.
- `eval slot_area_entropy=2.4612`; `log(16)=2.7726`.
- Eval slot area min/max: `0.0116` / `0.1567`.
- Một số slot lớn hoặc bám border/nền mạnh: slot 0/1/3/10 khá giống nhau; slot 11 lớn nhất; avg border mass per slot cao ở slot 0/1/3/10/11.

D6C slot statistics:

- `val_diag_effective_slots=12.7535`.
- `eval slot_area_entropy=2.5585`; gần uniform hơn D6B.
- Eval slot area min/max: `0.0131` / `0.1280`.
- Slot ít collapse hơn theo số liệu, nhưng visual vẫn có nhóm giống nhau: 1/4/9/11, 2/3, 7/8, 5/13, 0/15.

Kết luận:

- Không có hard slot collapse kiểu chỉ còn 1-2 slot hoạt động. Effective slots `11.6-12.75/16` nghĩa là đa số slot còn được dùng.
- Có partial redundancy. Một số slot học vùng nền/tóc/border hoặc biên mặt thay vì chỉ vùng biểu cảm.
- D6C ít collapse hơn theo `effective_slots`, `slot_area_entropy`, và val slot similarity mean (`0.1810` vs D6B `0.1943` tại best epoch).
- Border loss đang hoạt động: slot border mass giảm rõ so với epoch đầu. D6B best val `val_diag_border_mass_mean=0.1415`; D6C best val `0.1519`. Tuy nhiên slot/motif vẫn còn dùng biên mặt và nền.

## 5. Class-part motif analysis

Top slots per class từ CSV:

| Class | D6B top slots | D6C top slots |
|---|---|---|
| Angry | 0, 13, 10, 1, 3 | 13, 5, 9, 11, 1 |
| Disgust | 5, 15, 7, 12, 6 | 1, 4, 11, 9, 2 |
| Fear | 2, 9, 10, 3, 0 | 5, 13, 3, 2, 15 |
| Happy | 1, 0, 10, 3, 11 | 7, 11, 4, 8, 1 |
| Sad | 9, 2, 13, 0, 10 | 13, 5, 14, 10, 6 |
| Surprise | 2, 9, 12, 6, 4 | 3, 2, 15, 0, 4 |
| Neutral | 9, 11, 2, 7, 10 | 10, 14, 6, 8, 7 |

D6B class attention gần uniform:

- Entropy mỗi class quanh `2.76`, gần `log(16)=2.77`.
- Nhiều cosine similarity class-class rất cao: Fear-Sad `0.9962`, Fear-Neutral `0.9956`, Angry-Happy `0.9969`, Angry-Fear `0.9951`.
- Điều này nói rằng D6B thắng metric chủ yếu nhờ representation/part pooling tốt hơn D5, nhưng class-part attention chưa thật sự tách mạnh emotion motif.

D6C class-attended objectives tạo motif rõ hơn:

- Entropy giảm: Disgust `2.3038`, Fear `2.2953`, Surprise `2.2589`, Neutral `2.5191`.
- Top probability tăng: Surprise `0.2335`, Fear `0.1945`, Neutral `0.1941`, Disgust `0.1706`.
- Similarity giảm mạnh: val class attention similarity mean từ D6B `0.9419` xuống D6C `0.5630`.
- Class motifs nhìn khác nhau hơn, đặc biệt Surprise/Fear/Neutral/Disgust có top slot riêng rõ hơn.

Confusion-pair similarity của D6C:

| Pair | Similarity |
|---|---:|
| Fear-Sad | 0.7882 |
| Fear-Neutral | 0.3352 |
| Fear-Surprise | 0.6794 |
| Sad-Neutral | 0.7238 |
| Angry-Disgust | 0.7533 |

Các similarity này giải thích confusion:

- Fear-Sad cao (`0.7882`) đi cùng lỗi Fear -> Sad `100` và Sad -> Fear `101`.
- Fear-Surprise còn khá cao (`0.6794`) đi cùng Fear -> Surprise `78` và Surprise -> Fear `38`.
- Sad-Neutral cao (`0.7238`) đi cùng Sad -> Neutral `125` và Neutral -> Sad `101`.
- Angry-Disgust cao (`0.7533`) chưa làm Disgust chết, nhưng vẫn còn Disgust -> Angry `8` và Angry/Disgust motif gần nhau.

Kết luận: D6C class-attended objectives thật sự giúp class motif về mặt giải thích, nhưng lợi ích metric chưa bật lên rõ. Nhiều cặp khó vẫn có attention similarity cao và confusion tương ứng.

## 6. Border/background analysis

D6B:

- Có `lambda_border=0.0075`, `border_loss_type=slot_ratio`, `border_width=3`.
- Best val `val_diag_border_mass_mean=0.1415`, `val_diag_border_mass_max=0.3540`.
- Eval diagnostics `diag_border_mass_mean=0.1412`.
- Thiếu `class_motif_border_mass_by_class.csv` và `true_class_border_mass_by_class.csv`, nên không có số class-level border mass chính thức.
- Visual slot/motif cho thấy một số slot lớn học biên mặt/nền/tóc, đặc biệt các slot lớn giống nhau.

D6C:

- Có cùng slot border loss như D6B, thêm `lambda_class_border=0.0025`.
- Best val `val_loss_class_border=0.2057`, `val_diag_true_class_border_mass_mean=0.2057`, max `0.2882`.
- True-class border mass từ CSV:
  - Angry `0.2661`
  - Disgust `0.2303`
  - Fear `0.2636`
  - Happy `0.1806`
  - Sad `0.2594`
  - Surprise `0.1162`
  - Neutral `0.1619`
- Pred-class border mass cũng cao nhất ở Fear/Angry/Sad, thấp nhất Surprise/Neutral/Happy.

Border mass hiện tại chấp nhận được cho một run nghiên cứu, nhưng chưa "sạch". D6C class_border loss có vẻ hoạt động vì mean class border giảm từ khoảng `0.2329` epoch 1 xuống `0.2057` ở best epoch. Tuy nhiên Angry/Fear/Sad/Disgust vẫn còn bám biên/nền đáng kể.

Không nên tăng border penalty quá mạnh ngay. Nếu tăng, nên tăng nhẹ `lambda_class_border` sau khi giữ `lambda_border=0.0075`, vì penalty quá mạnh có thể xóa thông tin biên mặt hợp lệ và làm tụt metric.

## 7. Comparison with D5A/D5B

Mốc D5:

| Run | Accuracy | Macro F1 | Weighted F1 | pred_count |
|---|---:|---:|---:|---|
| D5A original | 0.4054 | 0.2927 | 0.3587 | [120, 0, 176, 1683, 666, 344, 600] |
| D5B-1 fixed motif | 0.3784 | 0.2973 | 0.3567 | [290, 0, 251, 1097, 554, 387, 1010] |
| D6B | 0.5589 | 0.5252 | 0.5520 | [396, 81, 449, 933, 555, 490, 685] |
| D6C | 0.5581 | 0.5264 | 0.5487 | [334, 70, 509, 976, 531, 520, 649] |

Gain so với D5A:

- D6B: accuracy `+0.1535`, macro F1 `+0.2325`, weighted F1 `+0.1932`.
- D6C: accuracy `+0.1527`, macro F1 `+0.2337`, weighted F1 `+0.1899`.

Gain so với D5B-1:

- D6B: accuracy `+0.1805`, macro F1 `+0.2279`, weighted F1 `+0.1953`.
- D6C: accuracy `+0.1797`, macro F1 `+0.2291`, weighted F1 `+0.1920`.

Happy bias:

- D5A Happy pred_count `1683` -> D6B `933` (`-750`, giảm khoảng `44.6%`).
- D5A Happy pred_count `1683` -> D6C `976` (`-707`, giảm khoảng `42.0%`).
- D5B-1 Happy pred_count `1097` -> D6B `933`; D6C `976`.

Disgust:

- D5A/D5B-1 pred_count `0`.
- D6B pred_count `81`, F1 `0.4559`.
- D6C pred_count `70`, F1 `0.4960`.
- Đây là bằng chứng mạnh nhất rằng D6 phá được collapse class nhỏ.

Fear:

- D5A pred_count `176`, F1 `0.1591`.
- D6B pred_count `449`, F1 `0.3664`.
- D6C pred_count `509`, F1 `0.3954`.
- Fear tăng mạnh nhưng vẫn là class khó vì còn nhầm Sad/Surprise/Neutral.

Per-class so với D5A original:

| Class | D5A F1 | D6B F1 | D6C F1 | Best gain vs D5A |
|---|---:|---:|---:|---:|
| Angry | 0.1342 | 0.4487 | 0.4339 | +0.3145 |
| Disgust | 0.0000 | 0.4559 | 0.4960 | +0.4960 |
| Fear | 0.1591 | 0.3664 | 0.3954 | +0.2363 |
| Happy | 0.5738 | 0.7748 | 0.7849 | +0.2111 |
| Sad | 0.3190 | 0.3951 | 0.3698 | +0.0761 |
| Surprise | 0.4579 | 0.6998 | 0.6838 | +0.2419 |
| Neutral | 0.4046 | 0.5355 | 0.5208 | +0.1309 |

Class tăng mạnh nhất so với D5A là Disgust, sau đó Angry/Surprise/Fear. Với D5B-1, local docs hiện có global metrics và pred_count nhưng không có per-class report đầy đủ, nên không kết luận per-class F1 gain chi tiết so với D5B-1.

Điều này chứng minh hướng hierarchical pixel-part motif đang giải quyết đúng nút thắt của D5: D5 chỉ có motif pixel/class phẳng, dễ collapse vào Happy và bỏ class nhỏ; D6 giữ full pixel graph nhưng thêm tầng soft part và class-part composition, làm motif có cấu trúc nội bộ hơn.

## 8. Decision: D6B vs D6C

### D6B hay D6C tốt hơn về metric?

Gần như hòa. D6C hơn macro F1 rất nhẹ (`+0.0012`), nhưng D6B hơn accuracy (`+0.0008`), weighted F1 (`+0.0033`), và best val macro F1 (`+0.0134`). Về metric ổn định, D6B nhỉnh hơn.

### D6B hay D6C tốt hơn về cân bằng class?

D6B cân bằng pred_count tổng thể tốt hơn nếu so với support test. D6C tốt hơn cho Disgust/Fear nhưng trả giá ở Angry/Sad và tăng Happy/Surprise prediction.

### D6B hay D6C tốt hơn về giải thích motif?

D6C tốt hơn rõ. Class attention của D6B còn gần uniform và similarity class-class rất cao. D6C làm attention sắc hơn, top slots rõ hơn, similarity giảm mạnh, và có thêm class border diagnostics.

### D6B hay D6C đáng làm baseline chính hơn?

Nên chọn D6B làm numeric baseline chính hiện tại. D6C nên được trình bày là phiên bản class-attended/explainability ablation rất hứa hẹn, gần ngang metric và tốt hơn motif separation. Nếu D6C long/tuned vượt D6B ổn định, khi đó có thể thay baseline.

## 9. Recommended next experiments

1. Giữ D6B `d6b_class_part_graph_motif_border075_long150` làm baseline chính hiện tại.
2. Chạy lại D6C longer hoặc cùng setup epoch/patience với D6B. D6C best ở epoch 93, final epoch 118 còn `val_macro_f1=0.5074`, không sụp mạnh; có thể cần schedule/patience tương đương để kiểm tra công bằng.
3. Tune class-attended objectives: D6C đã giảm similarity rất tốt nhưng metric chưa tăng rõ. Ưu tiên tune `lambda_class_attn_sep`, `class_attn_sep_margin`, `lambda_supcon` thay vì đổi kiến trúc.
4. Border loss: giữ `lambda_border=0.0075`; thử tăng nhẹ `lambda_class_border` từ `0.0025` lên `0.0035-0.005` cho D6C, nhưng theo dõi Angry/Fear/Sad vì đây là các class đang bám border cao.
5. Thêm/tune loss giảm similarity cho cặp hay nhầm, nhưng chỉ sau khi có D6C long. Ưu tiên Fear-Sad, Fear-Surprise, Sad-Neutral, Angry-Disgust. Không hard-code landmark; chỉ regularize class-part distributions hoặc pairwise motif similarity.
6. Có thể chạy checkpoint averaging/ensemble giữa D6B và D6C vì lỗi của chúng khác nhau: D6C tốt Disgust/Fear/Happy, D6B tốt Angry/Sad/Surprise/Neutral.
7. Nên dừng mở rộng D5 hướng cũ và tập trung D6. D5 có thể giữ làm baseline lịch sử/thesis, nhưng không nên quay lại làm hướng chính vì D6 đã vượt rất rõ và cứu được Disgust.

## 10. Notes for thesis/report writing

Luận điểm chính nên viết:

- D6 vẫn giữ lõi full pixel graph: mỗi ảnh vẫn là 2304 nodes, 17860 edges, node features `[2304,7]`, edge attributes `[17860,5]`.
- Điểm mới không phải hard-code mắt/mũi/miệng, mà là học soft part slots tự giám sát từ quan hệ pixel.
- Tầng class-part attention biến motif từ pixel mask phẳng thành tổ hợp part theo class, tức motif có cấu trúc nội bộ.
- Kết quả thực nghiệm ủng hộ giả thuyết: macro F1 tăng từ khoảng `0.293/0.297` lên `0.525/0.526`, Disgust từ `pred_count=0` lên `70-81`, Fear F1 tăng từ `0.1591` lên `0.366-0.395`, Happy bias giảm hơn 40%.
- D6C chứng minh class-attended objective có tác dụng về interpretability: class attention similarity mean giảm mạnh và top slots phân biệt hơn.
- Hạn chế còn lại: slot redundancy, motif còn bám border/nền ở một số class, và cặp Fear-Sad/Sad-Neutral/Fear-Surprise vẫn khó.

Một câu kết luận học thuật gợi ý:

> The D6 hierarchical motif framework substantially outperforms flat pixel-level class gates by preserving the full pixel graph while introducing learned soft part slots and class-specific part composition. The results suggest that emotion motifs in FER-2013 are better modeled as structured part configurations rather than independent class-level pixel masks.
