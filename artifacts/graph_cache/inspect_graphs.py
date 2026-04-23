import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from graph.structures import PixelGraph

pt_path = PROJECT_ROOT / "artifacts" / "graph_cache" / "train_graphs.pt"

data = torch.load(pt_path, map_location="cpu", weights_only=False)

print("type(data):", type(data))

if isinstance(data, list):
    print("len(data):", len(data))
    if len(data) > 0:
        g = data[0]
        print("type(data[0]):", type(g))
        print("\n=== first graph fields ===")
        for k, v in vars(g).items():
            if hasattr(v, "shape"):
                print(f"{k}: shape={v.shape}, type={type(v)}")
            else:
                print(f"{k}: {v} ({type(v)})")

elif isinstance(data, dict):
    print("keys:", list(data.keys()))
else:
    print(data)