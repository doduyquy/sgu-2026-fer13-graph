# Báo Cáo Tối Ưu Training – FER D5A (Kaggle T4)

> Đây là hướng dẫn đầy đủ về các thay đổi đã thực hiện và cách chạy từng bước thực nghiệm để đo tốc độ.

---

## Tóm tắt thay đổi code

| File | Thay đổi |
|------|----------|
| `training/trainer.py` | + Profiling chi tiết 7 metrics/batch, CUDA memory stats, AMP + GradScaler, `non_blocking=True` cho move_to_device |
| `scripts/common.py` | + `apply_cli_overrides` xử lý 8 args mới; `build_dataloader` hỗ trợ `persistent_workers`, `prefetch_factor`, log config |
| `scripts/run_experiment.py` | + 8 CLI args mới pass-through |
| `scripts/train_d5a.py` | + 8 CLI args mới pass-through |
| `configs/d5a.yaml` | + `amp: false`, `profile_batches: 0` trong training section |

---

## A. Baseline – Chạy đầu tiên để có số tham chiếu

```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size 16 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --no_wandb
```

Ghi lại: `train_sec_per_batch`, GPU util, GPU memory.

---

## B. Profiling – batch_size=16

```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size 16 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --profile_batches 10 \
  --no_wandb
```

Output mong đợi:
```
[PROFILE batch=0]
  data_time      =X.XXXXs
  to_device_time =X.XXXXs
  forward_time   =X.XXXXs
  loss_time      =X.XXXXs
  backward_time  =X.XXXXs
  optimizer_time =X.XXXXs
  batch_time     =X.XXXXs
  cuda_allocated_gb    =X.XXX
  cuda_reserved_gb     =X.XXX
  cuda_max_allocated_gb=X.XXX

[PROFILE average first 10 batches]
  avg_data_time      =...
  ...
  estimated_full_epoch_minutes=...
```

---

## C. Thử Batch Size Lớn Hơn

### batch_size=32
```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size 32 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --profile_batches 10 \
  --no_wandb
```

### batch_size=64
```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size 64 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --profile_batches 10 \
  --no_wandb
```

### batch_size=128 (nếu 64 vẫn dùng ít mem)
```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size 128 \
  --max_train_batches 20 \
  --max_val_batches 5 \
  --max_test_batches 5 \
  --profile_batches 5 \
  --no_wandb
```

> **Công thức tính:** `batches_per_epoch = ceil(28709 / batch_size)`
> - bs=16 → 1795 batches → nếu 1.5s/batch → ~44 phút
> - bs=32 → 898 batches → nếu 2.0s/batch → ~30 phút
> - bs=64 → 449 batches → nếu 3.0s/batch → ~22 phút
> - bs=128 → 225 batches → nếu 5.0s/batch → ~19 phút

---

## D. DataLoader Optimization (nếu data_time > 30% batch_time)

```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size <BEST_BS> \
  --num_workers 2 \
  --pin_memory true \
  --persistent_workers true \
  --prefetch_factor 2 \
  --graph_cache_chunks 4 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --profile_batches 10 \
  --no_wandb
```

> [!NOTE]
> `persistent_workers=True` tự động bị ignore nếu `num_workers=0` (có warning trong log).

---

## E. AMP Test

```bash
python scripts/run_experiment.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --mode train \
  --epochs 1 \
  --device cuda:0 \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --batch_size <BEST_BS> \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --max_test_batches 10 \
  --profile_batches 10 \
  --amp \
  --no_wandb
```

> [!IMPORTANT]
> AMP tự tắt nếu device không phải CUDA. Log sẽ in `[AMP] amp_enabled=True/False`.

---

## G. Bảng Kết Quả (điền sau khi chạy)

### 1. Baseline / Batch-size test

| batch_size | num_workers | amp | sec/batch | batches/epoch | est train min/epoch | cuda max GB | OOM | note |
|---:|---:|---|---:|---:|---:|---:|---|---|
| 16 | 0 | no | ? | 1795 | ? | ? | no | baseline |
| 32 | 0 | no | ? | 898 | ? | ? | ? | |
| 64 | 0 | no | ? | 449 | ? | ? | ? | |
| 128 | 0 | no | ? | 225 | ? | ? | ? | |

### 2. Profiling breakdown (best run)

| metric | seconds | percent |
|---|---:|---:|
| data_time | | |
| to_device_time | | |
| forward_time | | |
| loss_time | | |
| backward_time | | |
| optimizer_time | | |
| batch_time | | |

### 3. DataLoader test

| batch_size | num_workers | pin_memory | persistent_workers | prefetch | cache_chunks | sec/batch | data_time | note |
|---:|---:|---|---|---:|---:|---:|---:|---|

### 4. AMP test

| batch_size | amp | sec/batch | cuda max GB | NaN/Inf | note |
|---|---|---:|---:|---|---|

---

## H. Câu hỏi kết luận (điền sau khi có số liệu)

| Câu hỏi | Kết quả |
|---------|---------|
| A. Bottleneck chính | ? (DataLoader/forward/backward/loss) |
| B. Batch size tốt nhất | ? |
| C. DataLoader setting tốt nhất | ? |
| D. AMP có nên bật không | ? |
| E. Thời gian 1 epoch mới | ? phút |
| F. Có cần DDP không | Chưa đủ dữ liệu – đánh giá sau bước C/D/E |
| G. Có cần D5A-node-only không | Phụ thuộc vào loss_time |

---

## Chi tiết kỹ thuật các thay đổi

### `training/trainer.py`

- **`move_to_device`**: Thêm `non_blocking=True` để GPU transfer không block CPU.
- **`_sync(device)`**: Gọi `torch.cuda.synchronize()` trước/sau mỗi phase để timing chính xác.
- **`_cuda_mem_stats(device)`**: Log `allocated/reserved/max_allocated` GiB.
- **`_autocast()`**: Hỗ trợ cả `torch.amp.autocast` (mới) lẫn `torch.cuda.amp.autocast` (cũ).
- **`GradScaler`**: Dùng `scaler.scale(loss).backward()` + `scaler.unscale_(optimizer)` + `clip_grad_norm_` + `scaler.step()` + `scaler.update()`.
- **`fit()`**: Tính `total_train_batches` để estimate full epoch time trong profile.
- **Epoch log**: Thêm `train_sec/batch` vào cuối mỗi epoch print.

### `scripts/common.py`

- **`apply_cli_overrides`**: Xử lý `pin_memory`/`persistent_workers` dạng string `"true"/"false"`.
- **`build_dataloader`**: Guard `persistent_workers` và `prefetch_factor` khi `num_workers=0`.
- **`create_trainer`**: Forward `amp` và `profile_batches` từ config vào `D5Trainer`.

### `scripts/run_experiment.py` & `scripts/train_d5a.py`

- Thêm 8 args mới: `--profile_batches`, `--num_workers`, `--pin_memory`, `--persistent_workers`, `--prefetch_factor`, `--graph_cache_chunks`, `--amp`, `--no_amp`.
