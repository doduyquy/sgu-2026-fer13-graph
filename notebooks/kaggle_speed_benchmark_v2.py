# =============================================================================
# Speed Benchmark v2 – 3 mandatory scenarios
# Copy cell này vào Kaggle notebook và chạy sau Cell 1 (clone repo)
# =============================================================================

import subprocess, sys, os
from pathlib import Path

# --- Chỉnh nếu cần ---
REPO_PATH   = Path("/kaggle/working/sgu-2026-facial-expression-recognition")
OUTPUT_DIR  = "/kaggle/working/fer_d5_outputs/speed_benchmark_v2"
DEVICE      = "cuda:0"
ENVIRONMENT = "kaggle"

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# Chạy 3 scenario bắt buộc: bs64, bs128, bs64_amp
MANDATORY = [
    "d5a_speed_bs64",
    "d5a_speed_bs128",
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
