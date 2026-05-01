# D6C Ablation Plan Summary

## Why Ablate D6C

D6C-light improves the interpretability side of D6B by making class-to-part attention much less diffuse. The strongest evidence is the drop in confusion-pair attention similarity:

- Fear-Sad: about `0.9976 -> 0.7882`
- Fear-Neutral: about `0.9962 -> 0.3352`
- Fear-Surprise: about `0.9332 -> 0.6794`
- Sad-Neutral: about `0.9970 -> 0.7238`

It also improves Fear and Happy F1 compared with the D6B `border075` main model:

- Fear: `0.3702 -> 0.3954`
- Happy: `0.7681 -> 0.7849`

However, D6C-light is not the metric champion because Sad drops strongly:

- Sad: `0.4609 -> 0.3698`

The goal of this ablation set is to identify which D6C objective causes the Sad/Fear trade-off without changing graph input, architecture, K, optimizer, or training recipe.

## What D6C-Light Adds

D6C-light keeps the D6B backbone and adds:

1. `class-attended border loss`
2. `class-attention separation loss`
3. supervised contrastive loss on true `class_repr`

The ablations remove these terms selectively while keeping the D6B backbone unchanged.

## Why Run `no_supcon` First

`lambda_supcon=0.03` may pull true-class representations into a space that improves Fear but shifts the decision boundary against Sad. Since Sad is the largest regression in D6C-light, `no_supcon` is the cleanest first test.

Config:

```text
configs/experiments/d6c_class_attended_objectives_no_supcon.yaml
```

Key change:

```yaml
loss:
  lambda_supcon: 0.0
```

Decision rule:

- If Sad F1 recovers to `>= 0.42` while Fear stays `>= 0.37`, SupCon is likely the main trade-off source.
- If Disgust stays `>= 0.45` and macro F1 improves over D6C-light, the next D6C variant should remove SupCon or reduce it to around `0.01`.

## Why Run `no_sep` Second

If `no_supcon` does not recover Sad, the next suspect is `class_attn_sep`. It may force Fear/Sad/Neutral attention to be different even when they need some shared facial parts.

Config:

```text
configs/experiments/d6c_class_attended_objectives_no_sep.yaml
```

Key changes:

```yaml
loss:
  lambda_class_attn_sep: 0.0
  lambda_supcon: 0.03
```

Decision rule:

- If `no_sep` recovers Sad more than `no_supcon`, attention separation is likely too strong or too broad.
- If `no_sep` keeps Fear/Disgust and improves macro F1, the next D6C variant should drop or reduce class-attention separation.

## Role Of `border_only`

`border_only` isolates class-attended border loss from SupCon and attention separation. It is a report/diagnostic ablation, not the first full run priority.

Config:

```text
configs/experiments/d6c_class_attended_objectives_border_only.yaml
```

Key changes:

```yaml
loss:
  lambda_class_border: 0.0025
  lambda_class_attn_sep: 0.0
  lambda_supcon: 0.0
```

Decision rule:

- If `border_only` is close to D6B `border075`, class-attended border is safe but not sufficient.
- If `border_only` improves motif cleanliness without hurting Sad, it can remain as a low-risk D6C objective.

## Smoke Commands

Primary smoke:

```bash
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb
python scripts/evaluate_d5a.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --checkpoint output/d6c_class_attended_objectives_no_supcon/checkpoints/best.pth --max_test_batches 1
python scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --checkpoint output/d6c_class_attended_objectives_no_supcon/checkpoints/best.pth --max_samples 4
```

YAML sanity:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); from common import load_config; [print(p, load_config(p)['loss']) for p in ['configs/experiments/d6c_class_attended_objectives_no_sep.yaml','configs/experiments/d6c_class_attended_objectives_border_only.yaml']]"
```

## Full Commands

Run first:

```bash
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml
python scripts/evaluate_d5a.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --checkpoint output/d6c_class_attended_objectives_no_supcon/checkpoints/best.pth
python scripts/visualize_d6.py --config configs/experiments/d6c_class_attended_objectives_no_supcon.yaml --checkpoint output/d6c_class_attended_objectives_no_supcon/checkpoints/best.pth --max_samples 32
```

Only after analyzing `no_supcon`:

```bash
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_no_sep.yaml
python scripts/train_d5a.py --config configs/experiments/d6c_class_attended_objectives_border_only.yaml
```

## Evaluation Criteria For `no_supcon`

Compare with D6C-light:

- accuracy: `0.5581`
- macro F1: `0.5264`
- weighted F1: `0.5487`
- Disgust F1: `0.4960`
- Fear F1: `0.3954`
- Sad F1: `0.3698`

Compare with D6B `border075`:

- accuracy: `0.5606`
- macro F1: `0.5359`
- weighted F1: `0.5569`
- Disgust F1: `0.5047`
- Fear F1: `0.3702`
- Sad F1: `0.4609`

`no_supcon` is successful if:

- Sad F1 recovers to `>= 0.42`
- Fear F1 remains `>= 0.37`
- Disgust F1 remains `>= 0.45`, better if `>= 0.48`
- macro F1 is at least D6C-light or moves closer to `border075`
- accuracy/weighted F1 do not drop strongly
- class attention stays more separated than D6B
- pred_count does not shift too strongly toward Fear/Happy
- Fear-Sad and Sad-Neutral confusion does not worsen

## Decision Rules

If `no_supcon` is good:

- D6C should drop SupCon or reduce `lambda_supcon` to `0.01` in a later experiment.

If `no_supcon` is not good:

- Run `no_sep` to test whether class-attention separation is the main cause.

If `no_sep` is good:

- D6C should drop or reduce `lambda_class_attn_sep`, while possibly keeping class-attended border and light SupCon.

If neither is good:

- Keep D6C-light as an interpretability ablation.
- Keep D6B `border075` as the main metric model.
- Move later work toward D6D input fixes rather than adding more D6C variants.
