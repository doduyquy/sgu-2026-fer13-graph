# =============================================================================
# Speed Benchmark v4 – Full Pipeline
#
# SITUATION:
#   The Kaggle input dataset "irthn1311/graph-repo" only has 48 smoke samples.
#   The full FER-2013 graph repo (28709 train samples) must be built from the
#   raw CSVs on Kaggle. This notebook does it in two steps:
#
#   CELL A – Build full graph repo → /kaggle/working/artifacts/graph_repo
#   CELL B – Run speed benchmark   → /kaggle/working/fer_d5_outputs/speed_benchmark_v4
#
# Run CELL A first. After it completes (~5-15 min), run CELL B.
# Do NOT run CELL B if CELL A failed.
# =============================================================================

import subprocess, sys, os
from pathlib import Path

# --- Paths (do not change) ---
REPO_PATH    = Path("/kaggle/working/sgu-2026-fer13-graph")
GRAPH_REPO   = "/kaggle/working/artifacts/graph_repo"
OUTPUT_DIR   = "/kaggle/working/fer_d5_outputs/speed_benchmark_v4"
DEVICE       = "cuda:0"
ENVIRONMENT  = "kaggle"

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# =============================================================================
# CELL A: Build full graph repo
# Writes to /kaggle/working/artifacts/graph_repo
# Takes ~5-15 minutes depending on Kaggle GPU/CPU.
# =============================================================================

def cell_a_build_graph():
    """Build full FER-2013 graph repo from raw CSVs."""
    manifest = Path(GRAPH_REPO) / "manifest.pt"
    if manifest.exists():
        # Quick check: is it full?
        import torch
        m = torch.load(manifest, map_location="cpu", weights_only=False)
        train_n = m.get("splits", {}).get("train", {}).get("num_samples", 0)
        if train_n >= 28709:
            print(f"[SKIP] Full graph repo already exists: {GRAPH_REPO}")
            print(f"  train_samples={train_n}  (OK)")
            return True
        else:
            print(f"[REBUILD] Existing repo has only {train_n} train samples – rebuilding.")

    print("=" * 70)
    print("Building full FER-2013 graph repo...")
    print(f"Output: {GRAPH_REPO}")
    print("=" * 70)

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config",      "configs/d5a.yaml",
        "--environment", ENVIRONMENT,
        "--device",      DEVICE,
        "--mode",        "build_graph",
        "--no_wandb",
        "--graph_repo_path", GRAPH_REPO,
    ]
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    print("Exit code:", result.returncode)

    if result.returncode != 0:
        print("[ERROR] build_graph failed. Check errors above.")
        return False

    # Verify
    if manifest.exists():
        import torch
        m = torch.load(manifest, map_location="cpu", weights_only=False)
        train_n = m.get("splits", {}).get("train", {}).get("num_samples", 0)
        print(f"\n[OK] Graph repo built: train_samples={train_n}")
        return train_n >= 28709
    return False

build_ok = cell_a_build_graph()
print(f"\nCell A done. build_ok={build_ok}")

# =============================================================================
# CELL B: Run speed benchmark (only if build_ok)
# =============================================================================

def cell_b_benchmark():
    if not build_ok:
        print("[SKIP] CELL B skipped because CELL A failed. Fix CELL A first.")
        return

    # Confirm the full repo
    manifest = Path(GRAPH_REPO) / "manifest.pt"
    if not manifest.exists():
        print(f"[ERROR] {GRAPH_REPO}/manifest.pt not found. Run CELL A first.")
        return

    import torch
    m = torch.load(manifest, map_location="cpu", weights_only=False)
    train_n = m.get("splits", {}).get("train", {}).get("num_samples", 0)
    print(f"[CHECK] graph_repo train_samples={train_n}")
    if train_n < 28709:
        print(f"[ERROR] repo is still tiny ({train_n} samples). Re-run CELL A.")
        return

    MANDATORY = [
        "d5a_speed_bs16",
        "d5a_speed_bs32",
        "d5a_speed_bs64",
        "d5a_speed_bs64_amp",
    ]

    cmd = [
        sys.executable, "scripts/run_speed_benchmark.py",
        "--config",          "configs/d5a.yaml",
        "--environment",     ENVIRONMENT,
        "--device",          DEVICE,
        "--graph_repo_path", GRAPH_REPO,
        "--output_dir",      OUTPUT_DIR,
        "--no_wandb",
        "--require_full_repo",
        "--only",            *MANDATORY,
    ]
    print("\nCommand:", " ".join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, text=True)
    print("Exit code:", result.returncode)

    md_path = Path(OUTPUT_DIR) / "speed_benchmark_results.md"
    if md_path.exists():
        print("\n" + "=" * 70)
        print(md_path.read_text(encoding="utf-8"))
    else:
        print("[WARNING] No markdown results found.")

cell_b_benchmark()
