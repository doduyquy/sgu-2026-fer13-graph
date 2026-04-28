# =============================================================================
# Speed Benchmark v4 – 4 mandatory scenarios (full graph repo required)
#
# Fixes v4:
#   1. Auto-detects actual graph repo root (handles Kaggle extra-nesting).
#   2. Prints directory tree when no chunks found.
#   3. FAIL_TINY_REPO if train_samples < 1000.
#   4. batches_per_epoch = len(train_loader) (actual).
#   5. est_epoch_min = sec/batch × len(train_loader) – never 0.0.
#   6. bs_mismatch uses min(configured_bs, train_samples).
#   7. --require_full_repo aborts if train_samples != 28709.
#
# If fails with "No chunk files found", run the diagnostic cell below first
# to check what's actually at the Kaggle input path, then fix GRAPH_REPO_PATH.
#
# Copy this cell into the Kaggle notebook and run after Cell 1 (clone repo).
# =============================================================================

import subprocess, sys, os
from pathlib import Path

# --- Chỉnh nếu cần ---
REPO_PATH        = Path("/kaggle/working/sgu-2026-fer13-graph")
GRAPH_REPO_PATH  = "/kaggle/input/datasets/irthn1311/graph-repo"
OUTPUT_DIR       = "/kaggle/working/fer_d5_outputs/speed_benchmark_v4"
DEVICE           = "cuda:0"
ENVIRONMENT      = "kaggle"
REQUIRE_FULL     = True   # Set False chỉ khi muốn test trên smoke repo

# =============================================================================
# DIAGNOSTIC: inspect actual directory tree at GRAPH_REPO_PATH
# =============================================================================
def _print_tree(base, max_depth=3, prefix=""):
    base = Path(base)
    if not base.exists():
        print(f"  [NOT FOUND] {base}")
        return
    try:
        children = sorted(base.iterdir())
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    for i, child in enumerate(children[:25]):
        connector = "└── " if i == len(children) - 1 else "├── "
        extra = f"  ({child.stat().st_size:,} B)" if child.is_file() else "/"
        print(f"  {prefix}{connector}{child.name}{extra}")
        if child.is_dir() and max_depth > 1:
            ext = "    " if i == len(children) - 1 else "│   "
            _print_tree(child, max_depth - 1, prefix + ext)
    if len(children) > 25:
        print(f"  {prefix}  ... ({len(children) - 25} more entries)")

print("=" * 70)
print(f"GRAPH_REPO_PATH inspection: {GRAPH_REPO_PATH}")
print("=" * 70)
_print_tree(GRAPH_REPO_PATH)

# Check for manifest.pt anywhere under the path
import subprocess as _sp
result_find = _sp.run(
    ["find", GRAPH_REPO_PATH, "-name", "manifest.pt", "-o", "-name", "chunk_000.pt"],
    capture_output=True, text=True
)
if result_find.stdout.strip():
    print("\nFound key files:")
    for line in result_find.stdout.strip().split("\n")[:10]:
        print(f"  {line}")
else:
    print("\n[WARNING] No manifest.pt or chunk_000.pt found under GRAPH_REPO_PATH.")
    print("The graph repo may not be built, or the path is wrong.")
print("=" * 70)

# =============================================================================
# RUN BENCHMARK
# =============================================================================
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
    print("\nIf 'No chunk files found': check the tree above and fix GRAPH_REPO_PATH.")
    print("If the repo is not built yet:")
    print("  python scripts/run_experiment.py \\")
    print("    --config configs/d5a.yaml \\")
    print("    --environment kaggle --mode build_graph --device cuda:0")
