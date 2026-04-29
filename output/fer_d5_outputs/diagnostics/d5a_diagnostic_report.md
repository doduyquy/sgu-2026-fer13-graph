# D5A Diagnostic Report

## Summary
- output_root: `output\fer_d5_outputs`
- eval_root: `output\fer_d5_outputs\evaluation`
- best_epoch: 21.0
- best_val_macro_f1: 0.2953961309362905
- final_epoch: 41.0
- test_accuracy: 0.40540540540540543
- test_macro_f1: 0.2926539537047689
- test_weighted_f1: 0.35873429710672905

## Prediction Distribution
- true_count: {'Angry': 491, 'Disgust': 55, 'Fear': 528, 'Happy': 879, 'Sad': 594, 'Surprise': 416, 'Neutral': 626}
- pred_count: {'Angry': 120, 'Disgust': 0, 'Fear': 176, 'Happy': 1683, 'Sad': 666, 'Surprise': 344, 'Neutral': 600}
- pred_entropy: 1.4529
- No severe prediction collapse detected.

## Per-Class Quality
| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Angry | 0.3417 | 0.0835 | 0.1342 | 491 |
| Disgust | 0.0000 | 0.0000 | 0.0000 | 55 |
| Fear | 0.3182 | 0.1061 | 0.1591 | 528 |
| Happy | 0.4367 | 0.8362 | 0.5738 | 879 |
| Sad | 0.3018 | 0.3384 | 0.3190 | 594 |
| Surprise | 0.5058 | 0.4183 | 0.4579 | 416 |
| Neutral | 0.4133 | 0.3962 | 0.4046 | 626 |

- highest_f1: Happy (0.5738)
- lowest_f1: Disgust (0.0000)
- Disgust status: f1=0.0000, recall=0.0000
- Fear status: f1=0.1591, recall=0.1061
- Happy prediction share: 46.9%

## Confusion Notes
- Sad -> Happy: 225
- Fear -> Happy: 211
- Neutral -> Happy: 193
- Angry -> Happy: 185
- Angry -> Sad: 124

## Attention/Gate Quick Check
- class_gate_png_count: 0
- attention_png_count: 0
- total_figure_png_count: 0
- figures_dir: `output\fer_d5_outputs\figures`

## Conclusion
- D5A is learning a non-trivial signal, but quality is still weak/moderate.
- Main suspected issues to test next:
  - A. class collapse if pred_count is concentrated in 1-2 classes
  - B. edge motif noise if node_score_only improves macro F1
  - C. aux loss too strong if ce_only improves macro F1
  - D. prototype score too weak if all ablations stay flat
