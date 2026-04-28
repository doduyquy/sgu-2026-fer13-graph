# =============================================================================
# Speed Benchmark v3 – 4 mandatory scenarios
# Fixes:
#   1. est_epoch_min computed correctly from sec/batch * batches/epoch (no 0.0)
#   2. first_train_batch_x_shape captured from batch_idx=0 inside train loop
#   3. bs_mismatch check for ALL configured batch sizes (not just 128)
#   4. profile_batches_recorded < requested is flagged with ⚠
# Copy cell này vào Kaggle notebook và chạy sau Cell 1 (clone repo)
# =============================================================================

import subprocess, sys, os
from pathlib import Path

# --- Chỉnh nếu cần ---
REPO_PATH   = Path("/kaggle/working/sgu-2026-facial-expression-recognition")
OUTPUT_DIR  = "/kaggle/working/fer_d5_outputs/speed_benchmark_v3"
DEVICE      = "cuda:0"
ENVIRONMENT = "kaggle"

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# 4 scenarios bắt buộc
MANDATORY = [
    "d5a_speed_bs16",
    "d5a_speed_bs32",
    "d5a_speed_bs64",
    "d5a_speed_bs64_amp",
]

cmd = [
    sys.executable, "scripts/run_speed_benchmark.py",
    "--config",      "configs/d5a.yaml",
    "--environment", ENVIRONMENT,
    "--device",      DEVICE,
    "--output_dir",  OUTPUT_DIR,
    "--no_wandb",
    "--only",        *MANDATORY,
]

print("Command:", " ".join(cmd))
print("=" * 70)
result = subprocess.run(cmd, text=True)
print("Exit code:", result.returncode)

# Hiển thị kết quả
md_path = Path(OUTPUT_DIR) / "speed_benchmark_results.md"
if md_path.exists():
    print("\n" + "=" * 70)
    print(md_path.read_text(encoding="utf-8"))
