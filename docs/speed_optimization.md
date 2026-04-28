# Speed Optimization Guide for FER D5A

## 1. Vì sao cần benchmark

Việc huấn luyện D5A trên đồ thị ở cấp độ điểm ảnh (pixel-level graph) rất nặng về mặt tính toán và bộ nhớ. Một số điểm nghẽn (bottleneck) phổ biến có thể xảy ra:
- **DataLoader Bottleneck**: CPU không chuẩn bị batch kịp (data_time cao).
- **GPU Computation Bottleneck**: Forward/backward tốn nhiều thời gian (loss_time, backward_time cao).
- **CPU-GPU Transfer Bottleneck**: Thời gian copy data sang GPU tốn nhiều thời gian.

Vì vậy, chúng ta cần benchmark các kịch bản (scenario) có sẵn để tìm ra thiết lập tối ưu nhất (batch size, num_workers, AMP, v.v.) nhằm giảm thời gian của một epoch mà không cần thay đổi kiến trúc hoặc sử dụng DDP.

## 2. Các scenario có sẵn

Đã thiết lập sẵn các config trong `configs/scenarios/`:
- `d5a_speed_bs16.yaml`: Baseline với batch size 16.



- `d5a_speed_bs32.yaml`: Tăng batch size lên 32.
- `d5a_speed_bs64.yaml`: Tăng batch size lên 64.
- `d5a_speed_bs128.yaml`: Tăng batch size lên 128 (nguy cơ OOM cao nhất).
- `d5a_speed_bs64_loader2.yaml`: Batch 64 với 2 workers và prefetch.
- `d5a_speed_bs64_loader4.yaml`: Batch 64 với 4 workers và prefetch.
- `d5a_speed_bs64_amp.yaml`: Batch 64 sử dụng AMP (Automatic Mixed Precision).
- `d5a_speed_bs64_loader2_amp.yaml`: Kết hợp DataLoader tối ưu và AMP.

## 3. Cách chạy từng scenario

Chạy một kịch bản đơn lẻ và log lại `metrics`, `profile`:
```bash
python scripts/run_speed_scenario.py \
  --config configs/d5a.yaml \
  --scenario configs/scenarios/d5a_speed_bs64.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --device cuda:0 \
  --output_dir outputs/speed_benchmark \
  --no_wandb
```

## 4. Cách chạy toàn bộ benchmark

Chạy tự động tất cả các scenario, bắt lỗi OOM và sinh ra bảng kết quả `.csv` / `.md`:
```bash
python scripts/run_speed_benchmark.py \
  --config configs/d5a.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --device cuda:0 \
  --output_dir outputs/speed_benchmark \
  --no_wandb
```
Trong notebook Kaggle, có thể bật `RUN_SPEED_BENCHMARK = True` trong cell số 3.5 để chạy tự động.

## 5. Cách đọc bảng kết quả

Báo cáo benchmark sẽ bao gồm các chỉ số:
- **sec/batch**: Thời gian trung bình để xử lý 1 batch.
- **estimated_train_min_per_epoch**: Ước tính số phút cần để hoàn thành 1 epoch đầy đủ.
- **data_time**: Phần trăm/thời gian dùng để load dữ liệu.
- **forward/backward**: Phần trăm/thời gian GPU tính toán.
- **cuda max GB**: Đỉnh bộ nhớ VRAM đã cấp phát. Nếu chạm ngưỡng của T4 (khoảng 15GB), script sẽ log là OOM.

## 6. Khi nào chọn batch lớn

Nếu **cuda max GB** còn thấp và **sec/batch** giảm dần theo batch size, bạn có thể tăng batch size. Việc này tận dụng tối đa GPU stream processor. Tuy nhiên, nếu batch size quá lớn gây OOM, hãy lui về mức an toàn trước đó.

## 7. Khi nào bật AMP

Nên bật **AMP** (`amp=True`) khi phần lớn thời gian tiêu tốn vào **forward** và **backward**. AMP sử dụng Float16 cho các phép tính tương thích, giúp tiết kiệm bộ nhớ và tăng tốc độ xử lý TensorCore. Tuy nhiên, đôi khi nó gây gradient NaNs. Nếu benchmark với AMP chạy thành công, đó là tín hiệu tốt.

## 8. Khi nào tối ưu DataLoader

Nếu **data_time** chiếm phần trăm đáng kể (>30% tổng thời gian batch), DataLoader chính là điểm nghẽn. Hãy sử dụng scenario bật `num_workers > 0`, `pin_memory = True` và tăng `prefetch_factor`.

## 9. Khi nào mới nghĩ đến DDP

Khi batch size lớn nhất, dataloader tối ưu nhất và AMP đã được sử dụng nhưng thời gian 1 epoch vẫn quá lâu (ví dụ >30 phút), hoặc bạn cần tăng kích thước kiến trúc, lúc đó mới cần chia tải ra 2 GPU T4 bằng DDP (Distributed Data Parallel).

## 10. Khi nào cân nhắc D5A-node-only

Nếu việc truyền và tính toán `edge_index`, `edge_attr` trên GPU làm ngốn VRAM quá nhanh dù batch size nhỏ, bạn có thể bỏ qua cấu trúc cạnh và dùng model "Node-Only" (MLP, transformer) để giảm mạnh kích thước batch được tải vào bộ nhớ.
