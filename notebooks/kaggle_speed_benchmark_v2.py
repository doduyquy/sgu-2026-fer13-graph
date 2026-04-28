# =============================================================================
# Speed Benchmark v4 – 4 mandatory scenarios (full graph repo required)
#
# Fixes:
#   1. Inspects graph repo BEFORE training → logs train_samples, len_train_loader.
#   2. FAIL_TINY_REPO if train_samples < 1000.
#   3. batches_per_epoch = len(train_loader) (actual, not hardcoded 28709/bs).
#   4. est_epoch_min = sec/batch × len(train_loader) – never 0.0 or TINY.
#   5. bs_mismatch uses min(configured_bs, train_samples) as expected batch size.
#   6. --require_full_repo aborts if train_samples != 28709.
#   7. graph_repo_path defaults to /kaggle/working/artifacts/graph_repo.
#
# If this fails with FAIL_TINY_REPO, rebuild the full graph repo first:
#   python scripts/run_experiment.py \
#     --config configs/d5a.yaml --environment kaggle --mode build_graph \
#     --device cuda:0
#
# Copy this cell into the Kaggle notebook and run after Cell 1 (clone repo).
# =============================================================================

import subprocess, sys, os
from pathlib import Path

# --- Chỉnh nếu cần ---
REPO_PATH        = Path("/kaggle/working/sgu-2026-facial-expression-recognition")
GRAPH_REPO_PATH  = "/kaggle/working/artifacts/graph_repo"
OUTPUT_DIR       = "/kaggle/working/fer_d5_outputs/speed_benchmark_v4"
DEVICE           = "cuda:0"
ENVIRONMENT      = "kaggle"
REQUIRE_FULL     = True   # Set False chỉ khi muốn test trên smoke repo

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
    "--config",           "configs/d5a.yaml",
    "--environment",      ENVIRONMENT,
    "--device",           DEVICE,
    "--graph_repo_path",  GRAPH_REPO_PATH,
    "--output_dir",       OUTPUT_DIR,
    "--no_wandb",
    "--only",             *MANDATORY,
]
if REQUIRE_FULL:
    cmd.append("--require_full_repo")

print("Command:", " ".join(cmd))
print("=" * 70)
result = subprocess.run(cmd, text=True)
print("Exit code:", result.returncode)

# Hiển thị kết quả
md_path = Path(OUTPUT_DIR) / "speed_benchmark_results.md"
if md_path.exists():
    print("\n" + "=" * 70)
    print(md_path.read_text(encoding="utf-8"))
else:
    print("[WARNING] No markdown results found. Check for errors above.")
