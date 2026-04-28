# D5 Code Extraction Report

Project: FER-2013 Graph/Motif GNN  
Target redesign: D5 - Class-Level Pixel Motif Graph Retrieval  
Date: 2026-04-28  
Scope: read-only technical extraction from the current repository. No D5 implementation and no source-code refactor.

---

## 1. Executive Summary

Repo hien tai co hai lop code graph quan trong:

- Canonical graph pipeline moi nam trong `data/*`: `RawSample -> PixelGraphSample + SharedGraphStructure -> GraphResolver -> ResolvedPixelGraph`. Day la phan nen giu cho D5.
- Legacy graph/motif/candidate pipeline nam trong `src/graph/*`, `src/motif*`, `src/data/pixel_motif_dataset.py`, `src/data/candidate_attention_dataset.py`, `legacy/*`. Phan nay co nhieu y tuong huu ich nho, nhung khong nen copy vao project D5 vi D5 khong dung descriptor 41D, motif bank cu, candidate slots, greedy top-K, hay pixel_motif_dataset_v2.

D5 nen ke thua truc tiep cac contract sau:

- Raw CSV reading: `data/raw_types.py`, `data/raw_fer_dataset.py`.
- Graph construction: `configs/graph_config.py`, `data/shared_graph_builder.py`, `data/canonical_graph_builder.py`, `data/graph_types.py`.
- Graph repository: `data/graph_repository.py`, `data/chunked_graph_dataset.py`, `scripts/build_graph_repository.py`.
- Graph resolver: `data/graph_resolver.py`.
- Full graph dataloader D4A: `src/data/full_graph_dataset.py`, mot phan route `src/data/dataloader.py`.
- Reusable model primitive: `EdgeAwarePixelGNNLayer` trong `src/models/full_graph_adaptive_motif_slot_gnn.py`, rewrite de tach rieng `models/edge_gnn.py`.
- Training utilities: weighted CE/class weights, AdamW/ReduceLROnPlateau, metric computation, batch limit smoke-test, checkpoint/early stopping.

D5 data contract nen co dinh:

- `node_features` or `x`: `[B, 2304, 7]`, `float32`
- `edge_index`: `[2, E]`, `int64`, shared across batch
- `edge_attr`: `[B, E, 5]`, `float32`
- `node_mask`: `[B, 2304]`, `bool`
- `y`/`label`: `[B]`, `int64`
- For 48x48 8-neighbor directed graph: `E = 17,860`; no self-loop.

Quan trong: comment trong `data/shared_graph_builder.py` ghi "M approx 16,704" la khong chinh xac cho 48x48 8-neighbor directed. Cong thuc dung:  
`E_8 = 2*H*(W-1) + 2*(H-1)*W + 4*(H-1)*(W-1) = 17,860` voi `H=W=48`.

---

## 2. Repo Map

Bang duoi chi liet ke file active lien quan truc tiep den viec dung lai cho D5.

| Nhom | File | Vai tro hien tai | Ke thua? | Ly do |
|---|---|---:|---|---|
| Raw CSV | `data/raw_types.py` | Dataclass `RawSample` cho mot row FER-2013 da parse | YES | Contract gon, khong dinh graph |
| Raw CSV | `data/raw_fer_dataset.py` | `RawFERDataset`: doc `emotion`, `pixels`, optional `Usage`; parse image 48x48 | YES | Nen dung lam raw layer moi |
| Raw CSV legacy | `src/data/fer_split_dataset.py` | Dataset CSV cu tra dict | NO/PARTIAL | Logic giong `RawFERDataset` nhung contract kem sach hon |
| Graph config | `configs/graph_config.py` | Dataclass `GraphConfig`, feature names, repo root/chunk size | YES | Single source of truth tot cho graph v2 |
| Graph construction | `data/graph_types.py` | `SharedGraphStructure`, `PixelGraphSample`, `ResolvedPixelGraph` | YES | Data contract chinh cho D5 |
| Graph construction | `data/shared_graph_builder.py` | Build shared `edge_index` va static edge attrs | YES | Can giu gan nguyen, sua comment edge count |
| Graph construction | `data/canonical_graph_builder.py` | Build node features va dynamic edge attrs tung anh | YES | Cong thuc node/edge features dung voi D5 |
| Graph legacy | `src/graph/image_to_graph.py` | Builder cu, moi graph luu ca `edge_index/edge_attr`, local_contrast khac | NO | Trung lap, semantics khac canonical |
| Graph repository | `data/graph_repository.py` | Writer/Reader chunked graph repo | YES/PARTIAL | Writer/Reader tot; manifest nen clean path tuong doi |
| Graph repository | `data/chunked_graph_dataset.py` | Dataset lazy chunk loading + optional resolve | YES/PARTIAL | Nen giu, toi uu index build neu can |
| Build repo | `scripts/build_graph_repository.py` | CLI CSV -> graph_repo | YES/PARTIAL | Dung lam seed, rewrite thanh script D5 gon hon |
| Inspect repo | `scripts/inspect_graph_repository.py` | Validate shape, dtype, finite, feature names | YES/PARTIAL | Rat huu ich cho smoke/debug D5 |
| Resolver | `data/graph_resolver.py` | Merge shared topology + per-image dynamic features | YES | Contract cuc quan trong, nen copy gan nguyen |
| Full graph dataset | `src/data/full_graph_dataset.py` | `FullGraphDataset`, `collate_fn_full_graph` | YES | Contract batch gan dung D5 |
| Dataloader factory | `src/data/dataloader.py` | Multi-mode dataloader, trong do co route `full_graph` | PARTIAL | Lay full_graph route, bo cac mode legacy |
| D4A model | `src/models/full_graph_adaptive_motif_slot_gnn.py` | Edge-aware GNN + adaptive motif slots | PARTIAL | Chi lay `EdgeAwarePixelGNNLayer`, diagnostics/loss formulas |
| Model registry | `src/models/__init__.py` | Registry cho tat ca models | PARTIAL | D5 nen co registry nho hon |
| Loss | `src/training/losses.py` | Weighted CE, focal, motif consistency cu | PARTIAL | Giu class weights/WeightedCE; bo motif consistency cu |
| Optimizer | `src/training/optimizer.py` | Adam/AdamW/SGD + schedulers | YES/PARTIAL | Giu AdamW/ReduceLROnPlateau; clean config |
| Trainer | `src/training/trainer.py` | Generic trainer nhieu route + D4A stats | PARTIAL | Giu loop idea, move/forward/logits, limits, monitor; rewrite gon |
| Train entry | `scripts/train.py` | CLI train multi-mode | PARTIAL | Lay device resolve, config override, checkpoint/eval flow |
| Metrics | `src/evaluation/metrics.py` | accuracy, macro F1, weighted F1, report, confusion matrix | YES | Copy gan nguyen |
| Evaluator | `src/evaluation/evaluator.py` | Test eval, dict/tensor output, max_test_batches, figures | PARTIAL | Giu minimal evaluator; bo prediction grid neu khong can |
| Config load | `src/utils/config.py` | Load `base.yaml + model.yaml + env.yaml` | PARTIAL | Hien co merge flat env, de gay path confusion |
| Experiment runner | `scripts/run_experiment.py`, `src/pipeline/experiment_runner.py` | Build/train/debug modes | PARTIAL/NO | Huu ich ve modes, nhung qua gan artifact cu |
| Artifact builder | `src/pipeline/artifact_builder.py` | Build graph/candidate/motif artifacts | PARTIAL | Chi lay `build_graph_repo`, path resolve cho full_graph |
| Debug full graph | `scripts/debug_full_graph_d4_batch.py` | Shape/forward/backward sanity D4A | YES/PARTIAL | Lam seed cho `debug_d5a_batch.py` |
| Visualization | `scripts/visualize_candidate_attention.py` | Overlay candidate attention bbox | PARTIAL | Lay image loading, CSV/PNG pattern; D5 can heatmap moi |
| Visualization legacy | `legacy/scripts/visualize_pixel_motif_evidence.py` | Visualize selected motif bbox/center | PARTIAL | Lay bbox overlay idea only |

---

## 3. Raw Data / CSV Reading

### 3.1 Files

- `data/raw_types.py`
  - `RawSample` at line 19.
- `data/raw_fer_dataset.py`
  - `RawFERDataset` at line 40.
  - `_parse_pixels` at line 91.
  - `__getitem__` at line 108.
  - class distribution utilities at lines 130+.
- Legacy equivalent: `src/data/fer_split_dataset.py`.

### 3.2 RawSample Contract

Important code:

```python
@dataclass
class RawSample:
    sample_id: int
    label: int
    split: str
    usage: str
    image: np.ndarray  # shape (H, W), dtype float32, [0,255]
    metadata: Dict = field(default_factory=dict)
```

Input/output:

- Input source: one row in `train.csv`, `val.csv`, or `test.csv`.
- Output image: `np.ndarray`, shape `(48, 48)`, dtype `float32`, raw pixel range `[0,255]`.
- Output label: `int` in `[0,6]`.
- `split`: logical split from caller, not inferred from `Usage`.

### 3.3 Pixel Parsing

Important code:

```python
def _parse_pixels(self, pixel_str: str) -> np.ndarray:
    expected = self.image_size * self.image_size
    arr = np.fromstring(str(pixel_str), sep=" ", dtype=np.float32)
    if arr.size != expected:
        raise ValueError(
            f"Pixel count mismatch: expected {expected}, got {arr.size}"
        )
    return arr.reshape(self.image_size, self.image_size)
```

Assumptions:

- CSV has space-delimited `pixels` column.
- `image_size=48` by default, so expected token count is `2304`.
- No normalization in raw layer.

### 3.4 Label, Usage, Split

Important code:

```python
required = {"emotion", "pixels"}
missing = required - set(self._df.columns)
...
self._df["emotion"] = self._df["emotion"].astype(int)
if "Usage" not in self._df.columns:
    self._df["Usage"] = self.split
```

```python
sample = RawSample(
    sample_id=int(idx),
    label=int(row["emotion"]),
    split=self.split,
    usage=str(row["Usage"]),
    image=image,
)
```

CSV assumptions:

- Required columns: `emotion`, `pixels`.
- Optional column: `Usage`. If missing, auto-fill with split name.
- Split is controlled by the file path/caller (`train.csv`, `val.csv`, `test.csv`), not by `Usage`.

### 3.5 Recommended Clean Changes For D5

- Keep `RawSample` and `RawFERDataset`.
- Add explicit `csv_schema_version` or `source_csv` in metadata if needed.
- Keep raw image unnormalized; graph builder owns normalization.
- Make `EMOTION_NAMES` a single shared constant in `data/labels.py` to avoid mismatch:
  - `data/raw_fer_dataset.py` uses `Angry`, `Disgust`, ...
  - `src/data/emotions_dict.py` uses lowercase.
- Do not copy `src/data/fer_split_dataset.py`; it duplicates raw reading but returns less strict dict contract.

---

## 4. Graph Construction

### 4.1 Core Dataclasses

File: `data/graph_types.py`

`SharedGraphStructure`:

```python
@dataclass
class SharedGraphStructure:
    height: int
    width: int
    connectivity: int
    edge_index: torch.Tensor          # [2, M], int64
    edge_attr_static: torch.Tensor    # [M, S], float32
    static_feature_names: List[str] = field(default_factory=list)
    config_dict: Dict[str, Any] = field(default_factory=dict)
```

`PixelGraphSample`:

```python
@dataclass
class PixelGraphSample:
    graph_id: int
    label: int
    split: str
    usage: str
    height: int
    width: int
    node_features: torch.Tensor       # [N, d], float32
    edge_attr_dynamic: torch.Tensor   # [M, D], float32
    node_feature_names: List[str] = field(default_factory=list)
    dynamic_feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

`ResolvedPixelGraph`:

```python
@dataclass
class ResolvedPixelGraph:
    graph_id: int
    label: int
    split: str
    node_features: torch.Tensor       # [N, d]
    edge_index: torch.Tensor          # [2, M]
    edge_attr: torch.Tensor           # [M, S+D]
    node_feature_names: List[str] = field(default_factory=list)
    edge_feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Recommendation: copy these dataclasses into D5 with same names and shapes.

### 4.2 GraphConfig

File: `configs/graph_config.py`

Important defaults:

```python
height: int = 48
width: int = 48
connectivity: int = 8
normalize_pixels: bool = True
node_feature_names = [
    "intensity", "x_norm", "y_norm", "gx", "gy",
    "grad_mag", "local_contrast",
]
edge_static_feature_names = ["dx", "dy", "dist"]
edge_dynamic_feature_names = ["delta_intensity", "intensity_similarity"]
intensity_similarity_alpha: float = 1.0
chunk_size: int = 500
repo_root: str = "artifacts/graph_repo"
```

D5 should keep:

- `height`, `width`, `connectivity`
- `normalize_pixels`
- ordered node/edge feature names
- `intensity_similarity_alpha`
- `chunk_size`
- `version`

D5 can simplify:

- Move path fields out of graph feature config into a `paths` or `artifacts` section.

### 4.3 Node Features

File: `data/canonical_graph_builder.py`

| Feature | Computed in | Formula / logic | Shape | dtype | Expected range | Keep for D5? |
|---|---|---|---:|---|---|---|
| `intensity` | `_normalize`, `_build_node_features` | `I = raw / 255` if `normalize_pixels`, clipped `[0,1]`, flattened | `[N]` | float32 | `[0,1]` | YES |
| `x_norm` | `_build_coord_grids` | `x / (W - 1)` | `[N]` | float32 | `[0,1]` | YES |
| `y_norm` | `_build_coord_grids` | `y / (H - 1)` | `[N]` | float32 | `[0,1]` | YES |
| `gx` | `compute_gradients` | central diff x: `(I[:,x+1]-I[:,x-1])*0.5`; boundary one-sided | `[N]` | float32 | clipped `[-1,1]` | YES |
| `gy` | `compute_gradients` | central diff y: `(I[y+1]-I[y-1])*0.5`; boundary one-sided | `[N]` | float32 | clipped `[-1,1]` | YES |
| `grad_mag` | `compute_gradients` | `sqrt(gx^2 + gy^2)`, clipped `[0,1]` | `[N]` | float32 | `[0,1]` | YES |
| `local_contrast` | `compute_local_contrast` | `abs(I - mean_3x3_edge_padded)` | `[N]` | float32 | `[0,1]` | YES |

Important code:

```python
def _normalize(self, image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    if self.config.normalize_pixels:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    if not np.isfinite(img).all():
        raise ValueError("Non-finite values detected after image normalization")
    return img
```

```python
gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
gx[:, 0] = img[:, 1] - img[:, 0]
gx[:, -1] = img[:, -1] - img[:, -2]

gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
gy[0, :] = img[1, :] - img[0, :]
gy[-1, :] = img[-1, :] - img[-2, :]

grad_mag = np.sqrt(np.clip(gx * gx + gy * gy, a_min=0.0, a_max=None))
```

```python
padded = np.pad(img, pad_width=pad, mode="edge")
local_sum = np.zeros_like(img, dtype=np.float32)
for dy in range(window_size):
    for dx in range(window_size):
        local_sum += padded[dy:dy + img.shape[0], dx:dx + img.shape[1]]
local_mean = local_sum / float(window_size * window_size)
contrast = np.abs(img - local_mean)
```

### 4.4 Edge Construction

File: `data/shared_graph_builder.py`

Neighbor offsets:

```python
if connectivity == 4:
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]
if connectivity == 8:
    return [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
```

Edge index:

```python
for y in range(self.height):
    for x in range(self.width):
        u = self._node_id(y, x)
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                v = self._node_id(ny, nx)
                rows.append(u)
                cols.append(v)
edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
```

Contract:

- `edge_index`: `[2, E]`, `torch.int64`.
- Directed: yes. If `(u,v)` is a neighbor pair, both directions appear through offset enumeration.
- Self-loop: no.
- 48x48, 8-neighbor directed: `E = 17,860`.
- 48x48, 4-neighbor directed: `E = 9,024`.

### 4.5 Static Edge Features

File: `data/shared_graph_builder.py`

| Feature | Formula | Shape | Per-image? | Keep for D5? |
|---|---|---:|---:|---|
| `dx` | `dst_x - src_x` | `[E]` | No | YES |
| `dy` | `dst_y - src_y` | `[E]` | No | YES |
| `dist` | `sqrt(dx^2 + dy^2)` | `[E]` | No | YES |

Important code:

```python
src_y = src // self.width
src_x = src % self.width
dst_y = dst // self.width
dst_x = dst % self.width

dx = (dst_x - src_x).astype(np.float32)
dy = (dst_y - src_y).astype(np.float32)
dist = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
```

Expected ranges for 8-neighbor:

- `dx`, `dy`: `{-1, 0, 1}`
- `dist`: `{1, sqrt(2)}`

### 4.6 Dynamic Edge Features

File: `data/canonical_graph_builder.py`

| Feature | Formula | Shape | Per-image? | Keep for D5? |
|---|---|---:|---:|---|
| `delta_intensity` | `abs(I_src - I_dst)` | `[E]` | Yes | YES |
| `intensity_similarity` | `exp(-alpha * abs(I_src - I_dst))` | `[E]` | Yes | YES |

Important code:

```python
flat = image.ravel()
I_src = flat[self._src_ids]
I_dst = flat[self._dst_ids]

delta = np.abs(I_src - I_dst).astype(np.float32)
alpha = self.config.intensity_similarity_alpha

feature_map = {
    "delta_intensity": delta,
    "intensity_similarity": np.exp(-alpha * delta).astype(np.float32),
}
```

Expected ranges:

- `delta_intensity`: `[0,1]`
- `intensity_similarity`: `[exp(-alpha), 1]`; with alpha=1, roughly `[0.3679, 1]`.

### 4.7 Legacy/Conflict Warning

Do not copy `src/graph/image_to_graph.py` into D5:

- It computes full `edge_attr` per image and does not split static/dynamic edge attrs.
- It uses legacy `GraphConfig` in `src/graph/graph_config.py`.
- It has feature name `contrast`, while canonical graph uses `local_contrast`.
- Legacy `_compute_local_contrast` uses `mode="reflect"` and signed `image[y,x] - patch.mean()`, while canonical uses `mode="edge"` and absolute contrast.
- It stores numpy arrays in `PixelGraph`, not torch tensors.

For D5, choose canonical `data/*` version because it is cleaner and supports shared topology.

---

## 5. Graph Repository

### 5.1 Files

- `data/graph_repository.py`
  - `GraphRepositoryWriter` at line 54.
  - `_SplitWriter` at line 125.
  - `GraphRepositoryReader` at line 182.
- `data/chunked_graph_dataset.py`
  - `ChunkedGraphDataset` at line 51.
- `scripts/build_graph_repository.py`
  - `build_split` at line 94.
  - validation helpers at lines 142 and 154.
- `src/pipeline/artifact_builder.py`
  - `build_graph_repo` at line 179.
  - `ensure_pixel_motif_artifacts` handles `recipe == "full_graph_d4"` at line 709+.

### 5.2 On-Disk Format

Canonical layout:

```text
<repo_root>/
  manifest.pt
  shared/
    shared_graph.pt
  train/
    chunk_000.pt
    chunk_001.pt
    ...
  val/
    chunk_000.pt
    ...
  test/
    chunk_000.pt
    ...
```

Files:

- `shared/shared_graph.pt`: one `SharedGraphStructure`.
- `<split>/chunk_XXX.pt`: `List[PixelGraphSample]`.
- `manifest.pt`: dict:
  - `version`
  - `chunk_size`
  - `built_at`
  - `splits[split].num_samples`
  - `splits[split].num_chunks`
  - `splits[split].chunk_files`

Current caveat:

- `_SplitWriter` stores `str(path)` in `chunk_files`. If the repo is moved/uploaded, these paths may become stale.
- Reader does not depend on `chunk_files`; it uses `glob("chunk_*.pt")`. In D5 manifest, prefer relative chunk filenames.

### 5.3 Writer Flow

Important code:

```python
writer = GraphRepositoryWriter(repo_root=args.repo_root, config=cfg)
writer.write_shared(shared)
with writer.open_split("train") as sw:
    for raw_sample in raw_ds:
        pixel_graph = graph_builder.build(raw_sample)
        sw.add(pixel_graph)
writer.save_manifest()
```

`_SplitWriter.add`:

```python
self._buf.append(sample)
self._total += 1
if len(self._buf) >= self._chunk_size:
    self._flush()
```

`_flush`:

```python
name = CHUNK_PATTERN.format(idx=self._chunk_idx)
path = self._dir / name
torch.save(self._buf, path)
self._chunk_paths.append(str(path))
self._chunk_idx += 1
self._buf = []
```

### 5.4 Reader Flow

Important code:

```python
reader = GraphRepositoryReader(repo_root)
shared = reader.load_shared()
chunk = reader.load_chunk("train", 0)
for sample in reader.iter_split("train"):
    ...
```

`load_shared`:

```python
path = self.repo_root / SHARED_DIR / SHARED_FILENAME
shared = torch.load(path, map_location="cpu", weights_only=False)
```

`chunk_paths`:

```python
split_dir = self.repo_root / split
paths = sorted(split_dir.glob("chunk_*.pt"))
```

`iter_split`:

```python
for path in self.chunk_paths(split)[start_chunk:]:
    chunk = torch.load(path, map_location="cpu", weights_only=False)
    yield from chunk
```

### 5.5 ChunkedGraphDataset

Important code:

```python
self._reader = GraphRepositoryReader(repo_root)
self._chunk_paths = self._reader.chunk_paths(split)
self._index = self._build_global_index()
if resolve:
    self._shared = self._reader.load_shared()
    self._resolver = GraphResolver(self._shared)
```

`__getitem__`:

```python
chunk_idx, local_idx = self._index[idx]
chunk = self._get_chunk(chunk_idx)
sample = chunk[local_idx]
item = self._resolver.resolve(sample) if self.resolve else sample
```

Pros:

- Lazy chunk loading for actual samples.
- Optional resolution to full graph.
- Simple cache.

Cons / clean recommendation:

- `_build_global_index` loads every chunk once at dataset init just to get sizes. This is okay for FER-2013 but can be cleaned by storing chunk sizes in manifest.
- Cache eviction uses insertion order; acceptable with Python 3.7+ dict, but D5 could use `OrderedDict`.

### 5.6 Minimal D5 Read Code

```python
from data.chunked_graph_dataset import ChunkedGraphDataset

ds = ChunkedGraphDataset(
    repo_root="artifacts/graph_repo",
    split="train",
    resolve=True,
    cache_chunks=1,
)
g = ds[0]
assert g.node_features.shape == (2304, 7)
assert g.edge_index.shape[0] == 2
assert g.edge_attr.shape[1] == 5
```

### 5.7 Minimal D5 Build Code

```python
cfg = GraphConfig(connectivity=8, chunk_size=500, repo_root=repo_root)
shared = SharedGraphBuilder(cfg).build()
writer = GraphRepositoryWriter(repo_root=repo_root, config=cfg)
writer.write_shared(shared)

for split, csv_path in {"train": train_csv, "val": val_csv, "test": test_csv}.items():
    raw_ds = RawFERDataset(csv_path=csv_path, split=split, validate=False)
    builder = CanonicalGraphBuilder(cfg, shared)
    with writer.open_split(split) as sw:
        for raw in raw_ds:
            raw.validate(cfg.height, cfg.width)
            sw.add(builder.build(raw))

writer.save_manifest()
```

### 5.8 What To Keep vs Drop

Keep:

- `GraphRepositoryWriter`
- `GraphRepositoryReader`
- `ChunkedGraphDataset`
- `GraphResolver`
- `scripts/build_graph_repository.py` validation logic

Rewrite/clean:

- Manifest should store relative paths and chunk sample counts.
- `artifact_builder.py` should be split: D5 only needs graph repo builder, not candidate/motif stages.
- Add graph repo version/feature hash validation before training D5.

Drop:

- Candidate/motif stages in `artifact_builder.py`.
- `manifest.json` logic for pixel motif/candidate attention artifacts.

---

## 6. Graph Resolver

File: `data/graph_resolver.py`

### 6.1 Resolve Contract

`GraphResolver.resolve(sample)`:

1. Validates sample height/width match shared graph.
2. Validates dynamic edge rows match shared `num_edges`.
3. Concats static + dynamic edge attributes.
4. Returns `ResolvedPixelGraph`.

Important code:

```python
def resolve(self, sample: PixelGraphSample) -> ResolvedPixelGraph:
    self._validate_compatibility(sample)
    edge_attr = self._merge_edge_attrs(sample)
    edge_feature_names = (
        list(self.shared.static_feature_names)
        + list(sample.dynamic_feature_names)
    )
    return ResolvedPixelGraph(
        graph_id=sample.graph_id,
        label=sample.label,
        split=sample.split,
        node_features=sample.node_features,
        edge_index=self.shared.edge_index,
        edge_attr=edge_attr,
        node_feature_names=list(sample.node_feature_names),
        edge_feature_names=edge_feature_names,
        metadata={**sample.metadata, "height": sample.height, "width": sample.width,
                  "usage": sample.usage, "resolved": True},
    )
```

Merge logic:

```python
static = self.shared.edge_attr_static      # [M, S]
dynamic = sample.edge_attr_dynamic         # [M, D]
if static.shape[1] == 0:
    return dynamic
if dynamic.shape[1] == 0:
    return static
return torch.cat([static, dynamic], dim=1) # [M, S+D]
```

Validation:

```python
if sample.height != self.shared.height or sample.width != self.shared.width:
    raise ValueError(...)
if self.shared.num_edges != int(sample.edge_attr_dynamic.shape[0]):
    raise ValueError(...)
```

### 6.2 Output Shapes

For canonical D5 full graph:

- `node_features`: `[2304, 7]`
- `edge_index`: `[2, 17860]`
- `edge_attr`: `[17860, 5]`
- `edge_feature_names`: `["dx", "dy", "dist", "delta_intensity", "intensity_similarity"]`

### 6.3 Common Manual-Merge Errors

- Concatenating dynamic before static, changing feature order.
- Expanding `edge_index` to `[B,2,E]` unnecessarily; D4A/D5 can keep shared `[2,E]`.
- Using stale `edge_index` from old `src/graph` pipeline.
- Forgetting `weights_only=False` when loading dataclass objects with newer PyTorch.
- Recomputing edge attrs in a different order, causing dynamic edge rows not aligned with shared `edge_index`.

Recommendation: copy `GraphResolver` almost unchanged.

---

## 7. Full Graph Dataset / Dataloader D4A

### 7.1 FullGraphDataset

File: `src/data/full_graph_dataset.py`

Important code:

```python
class FullGraphDataset(Dataset):
    def __init__(self, repo_root: str, split: str, cache_chunks: int = 1) -> None:
        self._ds = ChunkedGraphDataset(
            repo_root=repo_root,
            split=split,
            resolve=True,
            cache_chunks=cache_chunks,
        )
        self.shared = self._ds.shared

    def __getitem__(self, idx: int) -> dict[str, Any]:
        graph: ResolvedPixelGraph = self._ds[idx]
        node_features = graph.node_features.float()
        return {
            "graph_id": int(graph.graph_id),
            "node_features": node_features,
            "x": node_features,
            "edge_index": graph.edge_index.long(),
            "edge_attr": graph.edge_attr.float(),
            "node_mask": torch.ones(graph.num_nodes, dtype=torch.bool),
            "label": torch.tensor(int(graph.label), dtype=torch.long),
            "y": torch.tensor(int(graph.label), dtype=torch.long),
        }
```

`__getitem__` keys:

- `graph_id`: int
- `node_features`: `[2304,7]`
- `x`: alias to `node_features`
- `edge_index`: `[2,E]`
- `edge_attr`: `[E,5]`
- `node_mask`: `[2304]`, all true
- `label`: scalar tensor
- `y`: scalar tensor

### 7.2 Collate Function

Important code:

```python
def collate_fn_full_graph(batch: list[dict[str, Any]]) -> dict[str, Any]:
    node_features = torch.stack([s["node_features"] for s in batch])
    edge_attr = torch.stack([s["edge_attr"] for s in batch])
    node_mask = torch.stack([s["node_mask"] for s in batch])
    labels = torch.stack([s["y"] for s in batch])
    return {
        "graph_id": torch.tensor([int(s["graph_id"]) for s in batch], dtype=torch.long),
        "node_features": node_features,
        "x": node_features,
        "edge_index": batch[0]["edge_index"],
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "label": labels,
        "y": labels,
    }
```

Batch contract:

- `node_features`/`x`: `[B, 2304, 7]`
- `edge_index`: `[2, 17860]`
- `edge_attr`: `[B, 17860, 5]`
- `node_mask`: `[B, 2304]`
- `label`/`y`: `[B]`
- `graph_id`: `[B]`

### 7.3 Dataloader Route

File: `src/data/dataloader.py`

Route logic:

```python
mode = config.get("dataloader_mode", data_cfg_top.get("mode", "graph_vector"))
if mode == "graph_vector" and data_cfg_top.get("recipe") == "full_graph_d4":
    mode = "full_graph"
...
elif mode == "full_graph":
    _validate_repo(graph_repo_path)
    return _build_full_graph_loaders(graph_repo_path, config, batch_size, num_workers)
```

Full graph loader:

```python
train_ds = FullGraphDataset(graph_repo_path, "train", cache_chunks=cache_chunks)
val_ds = FullGraphDataset(graph_repo_path, "val", cache_chunks=cache_chunks)
test_ds = FullGraphDataset(graph_repo_path, "test", cache_chunks=cache_chunks)
...
DataLoader(..., collate_fn=collate_fn_full_graph)
```

Config keys:

- `data.mode: full_graph`
- or top-level `dataloader_mode: full_graph`
- D4A fallback: `data.recipe: full_graph_d4`

### 7.4 Can Reuse Directly For D5?

Yes, mostly. D5 should copy/rewrite `FullGraphDataset` and `collate_fn_full_graph`.

Recommended clean changes:

- Move from `src/data/full_graph_dataset.py` into new project `data/full_graph_dataset.py`.
- Remove multi-mode dataloader factory; D5 only needs `build_full_graph_loaders`.
- Validate all samples share same `edge_index` shape.
- Optionally return `image_intensity = node_features[..., 0].reshape(B,48,48)` only in visualization, not training batch.

---

## 8. D4A Model - Reusable Parts Only

File: `src/models/full_graph_adaptive_motif_slot_gnn.py`

### 8.1 Node Encoder

Important code:

```python
self.node_encoder = nn.Sequential(
    nn.Linear(self.node_dim, self.hidden_dim),
    nn.LayerNorm(self.hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

Contract:

- Input: `[B, N, node_dim]`, default `node_dim=7`.
- Output: `[B, N, hidden_dim]`, default `hidden_dim=128`.

D5 reuse: YES. Copy into D5 model or `NodeEncoder`.

### 8.2 Edge Encoder

Inside `EdgeAwarePixelGNNLayer`:

```python
edge_input_dim = self.edge_dim if self.use_edge_attr and self.edge_dim > 0 else 1
self.edge_encoder = nn.Sequential(
    nn.Linear(edge_input_dim, hidden_dim),
    nn.GELU(),
    nn.LayerNorm(hidden_dim),
)
```

Contract:

- Input: edge_attr `[E, edge_dim]` or `[B,E,edge_dim]`, default `edge_dim=5`.
- Output: edge embedding `[E,H]` or per batch `[B,E,H]`.

D5 reuse: YES, but rewrite into standalone `models/edge_gnn.py`.

### 8.3 Edge-Aware Message Passing

Important code:

```python
src = edge_index[0].long().clamp(0, N - 1)
dst = edge_index[1].long().clamp(0, N - 1)
out = torch.zeros_like(h)
deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)

for b in range(B):
    e = self.edge_encoder(edge_attr_batched[b])
    msg_in = torch.cat([h[b, src], e], dim=-1)
    msg = self.message_mlp(msg_in)
    out[b].index_add_(0, dst, msg)
    deg[b].index_add_(0, dst, torch.ones((E, 1), device=h.device, dtype=h.dtype))

agg = out / deg.clamp_min(1.0)
h = self.msg_norm(h + self.dropout(agg))
h = self.ffn_norm(h + self.dropout(self.ffn(h)))
return h * node_mask.unsqueeze(-1).to(dtype=h.dtype)
```

Formula:

- Edge embedding: `e_uv = EdgeMLP(a_uv)`
- Message: `m_uv = MessageMLP([h_u, e_uv])`
- Aggregate: `m_v = (1 / deg(v)) * sum_{u -> v} m_uv`
- Residual update: `h'_v = LN(h_v + Dropout(m_v))`
- FFN: `h''_v = LN(h'_v + Dropout(FFN(h'_v)))`
- Mask: `h'' = h'' * node_mask`

D5 reuse:

- Good baseline encoder for graph matching scores.
- Performance caveat: current implementation loops over batch dimension. For `B=16`, `E=17860`, OK but not ideal. In clean D5, consider vectorizing with flattened batch offsets or `torch_scatter` if allowed.

### 8.4 D4A Diagnostics

Reusable diagnostics:

```python
entropy = -(assignments.clamp_min(1e-8) * assignments.clamp_min(1e-8).log()).sum(dim=-1)
assignment_entropy = (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)
```

```python
diff = assignments[:, src, :] - assignments[:, dst, :]
per_edge = diff.pow(2).mean(dim=-1)
smoothness = per_edge.masked_select(valid).mean()
```

Trainer D4A accumulator computes:

- `d4a/null_mass`
- `d4a/motif_mass_total`
- `d4a/assignment_entropy_mean`
- `d4a/assignment_entropy_std`
- `d4a/slot_mass_mean/min/max/std`
- `d4a/active_slot_count_soft`
- `d4a/slot_gate_mean/min/max`
- `d4a/logits_nan_count`
- `d4a/assignments_nan_count`

D5 adaptation:

- Rename diagnostics from `d4a/*` to `d5/*`.
- For class-level node attention `A = node_attn [B,C,N]`:
  - `d5/class_attn_entropy`
  - `d5/class_attn_mass`
  - `d5/class_gate_mean/min/max`
  - `d5/node_attn_nan_count`
  - `d5/edge_attn_nan_count`
  - `d5/smoothness_node_attn`

### 8.5 Do Not Bring To D5 As Main Architecture

Do not copy:

- `assignment_head` producing generic slots.
- `use_null_slot` behavior as D5 core.
- `_slot_pool` generic slot pooling.
- `slot_gates` as generic slots.
- `readout_mode = slots_global / slots_only / global_only` classifier as main architecture.
- Global residual shortcut if it lets model classify without motif retrieval.

Reason: D5 target is class-level pixel motif graph prototypes `T_0...T_6` and soft matching subgraphs `S_i,c`, not image-level generic slot pooling.

---

## 9. Loss / Class Weights / Regularization

### 9.1 Weighted CE

File: `src/training/losses.py`

Train counts:

```python
FER2013_TRAIN_COUNTS = [3995, 436, 4097, 7215, 4830, 3171, 4965]
```

Class weight formula:

```python
weights = total / (len(counts) * counts)
weights = weights.pow(power)
weights = weights / weights.mean().clamp_min(1e-8)
```

Mathematically:

`w_c = ((sum_j n_j) / (C * n_c))^p`, then normalize `w_c <- w_c / mean(w)`.

With `class_weight_power=0.25`, current weights are approximately:

```text
[0.926369, 1.611727, 0.920548, 0.799105, 0.883439, 0.981441, 0.877371]
```

Weighted CE:

```python
class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights=None, label_smoothing: float = 0.0):
        ...

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
```

Recommendation: copy `compute_class_weights` and `WeightedCrossEntropy` into D5.

### 9.2 Loss Build System

Current `build_loss(config)` supports:

- `cross_entropy` / `ce`
- `weighted_ce` / `weighted_cross_entropy`
- `focal`
- `weighted_focal`
- `weighted_ce_motif`
- `ce_motif`

D5 clean recommendation:

- Keep only:
  - `ce`
  - `weighted_ce`
  - optional `focal`
  - D5-specific graph losses
- Remove `CombinedMotifLoss` tied to old `motif_score_vector`.

### 9.3 Existing Regularization

| Loss / regularization | Exists? | File/function | Reusable? | Notes |
|---|---:|---|---|---|
| Weighted CE | Yes | `src/training/losses.py::WeightedCrossEntropy` | YES | Copy |
| Focal loss | Yes | `FocalLoss` | PARTIAL | Optional |
| Motif consistency | Yes | `MotifConsistencyLoss` | NO | Tied to old motif score vector |
| Assignment entropy | Yes | D4A `_assignment_entropy` | YES/PARTIAL | Reuse formula for D5 attention entropy |
| Slot diversity | Yes | D4A `_slot_diversity` | PARTIAL | Replace slots with class prototypes/gates |
| Assignment smoothness | Yes | D4A `_assignment_smoothness` | YES | Reuse for class node attention |
| Sparsity | Indirect | entropy minimization via `lambda_sparse` | PARTIAL | Better define explicit area control |
| Contrastive class loss | No | N/A | WRITE NEW | Needed for D5 |
| Edge-node closure | No | N/A | WRITE NEW | Needed for D5 motif subgraphs |
| Area control | No | N/A | WRITE NEW | Needed for D5 |
| Class gate diversity | No | N/A | WRITE NEW | Needed for D5 |

### 9.4 Proposed D5 Loss Formulas

Assume:

- `logits`: `[B,C]`
- `y`: `[B]`
- `node_attn`: `[B,C,N]`
- `edge_attn`: `[B,C,E]`
- `class_proto`: `[C,K,H]` or `[C,H]`
- `edge_index`: `[2,E]`
- `node_mask`: `[B,N]`

Graph smoothness over class attention:

```python
def graph_smoothness(node_attn, edge_index, node_mask=None):
    # node_attn: [B,C,N]
    src, dst = edge_index.long()
    diff = node_attn[:, :, src] - node_attn[:, :, dst]  # [B,C,E]
    per_edge = diff.pow(2)
    if node_mask is not None:
        valid = node_mask[:, src] & node_mask[:, dst]   # [B,E]
        per_edge = per_edge.masked_select(valid[:, None, :])
    return per_edge.mean()
```

Area control:

```python
def area_control(node_attn, target_area=0.10):
    # Encourage per-class selected mass not to collapse/all-image.
    mass = node_attn.mean(dim=-1)  # [B,C]
    return (mass - target_area).pow(2).mean()
```

Class gate/prototype diversity:

```python
def prototype_diversity(proto):
    # proto: [C,H] or [C,K,H] flattened per class
    p = proto.reshape(proto.shape[0], -1)
    p = F.normalize(p, dim=-1)
    sim = p @ p.t()
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye).pow(2).mean()
```

Class contrast loss:

```python
def class_score_margin(scores, y, margin=0.2):
    # scores/logits: [B,C]
    B, C = scores.shape
    row = torch.arange(B, device=scores.device)
    pos = scores[row, y]
    neg = scores.masked_fill(F.one_hot(y, C).bool(), -1e9).max(dim=1).values
    return F.relu(margin - pos + neg).mean()
```

Edge-node closure:

```python
def edge_node_closure(edge_attn, node_attn, edge_index):
    # Penalize high edge attention when endpoints have low node attention.
    src, dst = edge_index.long()
    endpoint = torch.minimum(node_attn[:, :, src], node_attn[:, :, dst])  # [B,C,E]
    return F.relu(edge_attn - endpoint).pow(2).mean()
```

Recommended D5 clean module:

```text
training/losses.py
  compute_class_weights
  WeightedCrossEntropy
  D5RetrievalLoss:
    ce
    graph_smoothness
    area_control
    prototype_diversity
    edge_node_closure
    optional class_score_margin
```

---

## 10. Optimizer / Scheduler

File: `src/training/optimizer.py`

### 10.1 Current Contract

`build_optimizer(model, config)`:

```python
opt_cfg = config.get("optimizer", {}) or {}
opt_name = opt_cfg.get("name", train_cfg.get("optimizer", "adam")).lower()
lr = opt_cfg.get("lr", train_cfg.get("lr", 0.001))
weight_decay = opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0001))
```

Supported:

- `adam`
- `adamw`
- `sgd`

`build_scheduler(optimizer, config)`:

- `none`
- `ReduceLROnPlateau`
- `step`
- `cosine`

ReduceLROnPlateau:

```python
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=mode,
    factor=factor,
    patience=patience,
    min_lr=min_lr,
)
scheduler.monitor_key = monitor
scheduler.monitor_mode = mode
```

D4A config:

```yaml
optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.0001

scheduler:
  name: ReduceLROnPlateau
  mode: max
  monitor: val_macro_f1
  factor: 0.5
  patience: 5
  min_lr: 0.000001
```

### 10.2 Recommended D5A Defaults

Start conservative:

```yaml
optimizer:
  name: adamw
  lr: 0.0007
  weight_decay: 0.0001

scheduler:
  name: ReduceLROnPlateau
  mode: max
  monitor: val_macro_f1
  factor: 0.5
  patience: 5
  min_lr: 0.000001
```

If D5 prototypes/gates are unstable, try:

- `lr: 0.0005`
- `grad_clip_norm: 5.0`
- warmup 3-5 epochs only if adding custom scheduler.

Clean D5 contract:

```python
optimizer = build_optimizer(model, cfg["optimizer"])
scheduler, monitor_key = build_scheduler(optimizer, cfg["scheduler"])
```

Avoid relying on mixed `training.*` and `optimizer.*` fallback fields in the new project.

---

## 11. Trainer Loop

File: `src/training/trainer.py`

### 11.1 Useful Features To Keep

- `Trainer.fit`
- `train_one_epoch`
- `validate`
- `_move_batch_to_device`
- `_extract_logits`
- `_compute_loss` supporting dict loss and model `aux_loss`
- `_forward_batch` dispatch for full graph dict
- `max_train_batches`, `max_val_batches`, `max_test_batches`
- `grad_clip_norm`
- `pred_count`
- early stopping by configurable monitor
- checkpoint save with `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_monitor`
- optional WandB logging
- finite diagnostics pattern

### 11.2 Current Forward/Output Handling

Logits extraction:

```python
def _extract_logits(self, model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    return model_out
```

Full graph route:

```python
if "node_features" in batch and "edge_index" in batch and "edge_attr" in batch:
    return self.model(batch)
```

Loss with `aux_loss`:

```python
loss_out = self.criterion(logits, y, batch=batch)
...
aux_loss = model_out.get("aux_loss") if isinstance(model_out, dict) else None
...
return loss_out + aux_loss
```

Batch move:

```python
return {
    key: value.to(self.device) if torch.is_tensor(value) else value
    for key, value in batch.items()
}
```

### 11.3 What Is Too Busy

Current trainer is crowded by:

- Candidate attention route
- Pixel motif route
- Hierarchical route
- D4A-specific print formatting
- WandB payload assembled inline
- Multiple loss conventions

D5 should rewrite trainer around one primary batch contract.

### 11.4 Minimal Clean Trainer Skeleton For D5

```python
class Trainer:
    def train_one_epoch(self):
        self.model.train()
        y_true, y_pred = [], []
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(self.train_loader):
            if self.max_train_batches and step >= self.max_train_batches:
                break
            batch = move_to_device(batch, self.device)
            out = self.model(batch)
            logits = out["logits"] if isinstance(out, dict) else out
            y = batch["y"].long()
            loss_out = self.criterion(out, y, batch)  # D5 loss can inspect attentions
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            bs = y.shape[0]
            total_loss += float(loss.detach()) * bs
            seen += bs
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        return {"loss": total_loss / max(1, seen), **metrics_without_report(metrics)}
```

### 11.5 D5 Diagnostics To Add

Replace D4A accumulator with D5-specific:

- `d5/node_attn_entropy`
- `d5/node_attn_mass_mean/min/max`
- `d5/edge_attn_mass_mean`
- `d5/class_proto_norm_mean`
- `d5/class_gate_mean/min/max`
- `d5/logits_nan_count`
- `d5/node_attn_nan_count`
- `d5/edge_attn_nan_count`
- `d5/grad_norm`

---

## 12. Evaluator / Metrics

### 12.1 Metrics

File: `src/evaluation/metrics.py`

Important code:

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
```

Return keys:

- `accuracy`
- `macro_f1`
- `weighted_f1`
- `confusion_matrix`
- `report`

Recommendation: copy as-is.

### 12.2 Evaluator

File: `src/evaluation/evaluator.py`

Useful behavior:

- Handles dict model output with `out["logits"]`.
- Moves tensor batch to device.
- Supports `training.max_test_batches`.
- Saves confusion matrix to `save_dir/confusion_matrix.png`.
- Prints classification report.
- Collects `graph_id` for later visualization.

D5 minimal evaluator:

```python
@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    y_true, y_pred, graph_ids = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        logits = out["logits"] if isinstance(out, dict) else out
        pred = logits.argmax(dim=1)
        y_true.extend(batch["y"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        graph_ids.extend(batch.get("graph_id", torch.arange(pred.numel())).cpu().tolist())
    return compute_classification_metrics(y_true, y_pred)
```

Clean changes:

- Separate scalar metric computation from plotting.
- Separate prediction grid raw-image visualization from evaluator.
- Add optional saving of D5 attention tensors for selected samples.

---

## 13. Config System / Experiment Runner

### 13.1 Current Config Loading

File: `src/utils/config.py`

Current flow:

```python
model_config = yaml.safe_load(open(configs/<model>.yaml))
env_config = yaml.safe_load(open(configs/env.yaml))
base_config = yaml.safe_load(open(configs/base.yaml))
config = _deep_update(base_config, model_config)
config = {**config, **env_config[env]}
```

Issue:

- Env config is merged flat into top-level config.
- Paths like `graph_repo_path`, `data_path`, `root_path` become top-level keys.
- `base.yaml` also contains `kaggle` and `local` sections, while `env.yaml` duplicates path fields. This can confuse Kaggle vs Windows/local runs.

Recommendation for D5:

- Use one config schema with explicit sections:
  - `paths`
  - `data`
  - `graph`
  - `model`
  - `loss`
  - `optimizer`
  - `scheduler`
  - `training`
  - `logging`
- Resolve environment only once.
- Do not deep-merge hidden base configs unless necessary.

### 13.2 Current Experiment Runner

Files:

- `scripts/run_experiment.py`
- `src/pipeline/experiment_runner.py`
- `src/pipeline/artifact_builder.py`

Modes:

- `build_and_train`
- `train_from_artifact`
- `build_only`
- `train_only`
- `debug_only`

Full graph D4A path:

- Experiment config `configs/experiments/d4a_full_graph_adaptive_slot.yaml`:
  - `data.recipe: full_graph_d4`
  - `data.stage: graph_repo`
  - `training.config: full_graph_adaptive_motif_slot_gnn_d4a`
  - `training.debug_batch: true`
- `artifact_builder.ensure_pixel_motif_artifacts` special-cases `recipe == "full_graph_d4"` and only builds graph repo.
- `experiment_runner.debug_batch_for_experiment` calls `debug_full_graph_d4_batch` if config name includes `full_graph_adaptive_motif_slot_gnn`.

Good ideas to keep:

- Explicit modes.
- Build graph repo once, then train.
- `--smoke`, `--max_train_batches`, `--max_val_batches`, `--max_test_batches`.
- `debug_batch` before long training.

Drop/simplify:

- Pixel motif/candidate/motif bank branches.
- Artifact auto-detection for old artifacts.
- `pixel_motif_dir` path passing into train command.

### 13.3 Proposed D5 Config Schema

`configs/d5a_model.yaml`:

```yaml
model:
  name: class_pixel_motif_graph_retrieval
  num_classes: 7
  node_dim: 7
  edge_dim: 5
  hidden_dim: 128
  gnn_layers: 2
  prototype_dim: 128
  prototypes_per_class: 4
  retrieval_temperature: 0.2
  use_edge_attr: true
  dropout: 0.2

loss:
  name: d5_retrieval
  ce_weight: 1.0
  use_class_weights: true
  class_counts: [3995, 436, 4097, 7215, 4830, 3171, 4965]
  class_weight_power: 0.25
  lambda_smooth: 0.02
  lambda_area: 0.01
  target_area: 0.10
  lambda_edge_closure: 0.01
  lambda_proto_diversity: 0.01
```

`configs/d5a_experiment.yaml`:

```yaml
experiment:
  name: d5a_class_level_pixel_motif_graph_retrieval
  mode: build_and_train

paths:
  csv_root: auto
  artifact_root: artifacts
  graph_repo_path: artifacts/graph_repo
  output_root: outputs

data:
  dataset: fer2013
  dataloader: full_graph
  image_size: 48
  num_classes: 7
  batch_size: 16
  num_workers: 0
  pin_memory: true
  graph_cache_chunks: 1

graph:
  connectivity: 8
  normalize_pixels: true
  node_feature_names: [intensity, x_norm, y_norm, gx, gy, grad_mag, local_contrast]
  edge_static_feature_names: [dx, dy, dist]
  edge_dynamic_feature_names: [delta_intensity, intensity_similarity]
  intensity_similarity_alpha: 1.0
  chunk_size: 500

optimizer:
  name: adamw
  lr: 0.0007
  weight_decay: 0.0001

scheduler:
  name: ReduceLROnPlateau
  mode: max
  monitor: val_macro_f1
  factor: 0.5
  patience: 5
  min_lr: 0.000001

training:
  epochs: 80
  monitor: val_macro_f1
  early_stopping_patience: 20
  grad_clip_norm: 5.0
  max_train_batches: null
  max_val_batches: null
  max_test_batches: null
  seed: 42

logging:
  use_wandb: false
  project: FER2013-D5
```

Path handling warning:

- Windows local default should use relative `artifacts/graph_repo`, `num_workers: 0`.
- Kaggle input graph repo is read-only: `/kaggle/input/fer-graph-repo/graph_repo`.
- Kaggle working outputs: `/kaggle/working/...`.
- Do not mix `/kaggle/input` paths as output dirs.

---

## 14. Debug / Smoke Test

### 14.1 Current Full Graph Debug

File: `scripts/debug_full_graph_d4_batch.py`

Checks:

- Forces `dataloader_mode = "full_graph"`.
- Loads one train batch.
- Prints shapes and stats for:
  - `node_features`
  - `x`
  - `edge_index`
  - `edge_attr`
  - `node_mask`
  - `y`
- Builds model and criterion.
- Runs forward/backward.
- Asserts:
  - `logits.shape == (B, num_classes)`
  - D4A-specific `slot_assignments.shape == (B,2304,num_slots+1)`
  - finite logits.

Reusable helper:

```python
def _stats(name: str, value: torch.Tensor) -> None:
    v = value.detach().float()
    print(
        f"{name:<28}: min={v.min().item():.6f} max={v.max().item():.6f} "
        f"mean={v.mean().item():.6f} std={v.std(unbiased=False).item():.6f}"
    )
```

### 14.2 D5 Debug Checklist

`debug_d5a_batch.py` should check:

- Data:
  - `node_features` / `x`: `[B, 2304, 7]`
  - `edge_index`: `[2, 17860]`
  - `edge_attr`: `[B, 17860, 5]`
  - `node_mask`: `[B, 2304]`
  - `y`: `[B]`
- Model output:
  - `logits` or class scores: `[B, 7]`
  - `node_attn`: `[B, 7, 2304]`
  - `edge_attn`: `[B, 7, 17860]`
  - optional `class_graph_scores`: `[B, 7]`
  - optional `prototype_scores`: `[B, 7, K]`
- Numerical:
  - all finite: `logits`, `node_attn`, `edge_attn`, total loss
  - backward OK
  - grad norm finite
- Semantics:
  - attention sums/mass not all zero
  - per-class attention not identical at initialization after first update smoke
  - edge attention does not exceed endpoint node attention if closure projection is used

Smoke train command after new project:

```bash
python scripts/debug_d5a_batch.py --config configs/d5a_experiment.yaml --batch_size 2
python scripts/train_d5a.py --config configs/d5a_experiment.yaml --epochs 2 --max_train_batches 3 --max_val_batches 2 --max_test_batches 2
```

---

## 15. Visualization / Attention / Motif Mask

### 15.1 Existing Scripts

`scripts/visualize_candidate_attention.py`

Reusable:

- Load graph image from graph repo:

```python
for sample in reader.iter_split(split):
    images[int(sample.graph_id)] = sample.node_features[:, 0].float().reshape(48, 48)
```

- Save PNG and CSV.
- Overlay top attention bboxes.

Not reusable directly:

- Depends on `CandidateAttentionDataset`.
- Visualizes candidate bboxes, not D5 full node/edge attention.

`legacy/scripts/visualize_pixel_motif_evidence.py`

Reusable:

- Bbox/center overlay pattern.
- `fig.savefig(out_path, dpi=180)`.

Not reusable directly:

- Depends on `pixel_motif_dataset_v2`.
- Uses selected motif centers/bboxes.

`src/evaluation/metrics.py` and `src/evaluation/evaluator.py`

Reusable:

- Confusion matrix save path: `save_dir/confusion_matrix.png`.

### 15.2 Mapping Node IDs To Image

Canonical order:

- `node_id = y * width + x`
- `node_features[:,0].reshape(48,48)` recovers intensity image.
- Any `node_attn[i,c]` with shape `[2304]` can be visualized by `.reshape(48,48)`.

### 15.3 Proposed `visualize_d5_motifs.py`

Required outputs:

- `class_node_gate[c].reshape(48,48)` for learned class-level prototype/gate.
- `node_attn[i,c].reshape(48,48)` for a sample and class.
- Grid of all 7 class attentions for one sample.
- Overlay predicted and true class heatmaps on intensity image.
- Edge attention optional:
  - convert top edges to line segments `(src_x,src_y)->(dst_x,dst_y)`.
- Deletion/insertion tests:
  - remove top-attended pixels and measure score drop.
  - insert top-attended pixels over baseline image and measure score rise.

Minimal D5 heatmap code:

```python
def node_map(vec, H=48, W=48):
    return torch.as_tensor(vec).detach().cpu().float().reshape(H, W).numpy()

def save_attention_grid(image, node_attn, label, pred, out_path):
    # image: [48,48], node_attn: [7,2304]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"image y={label} pred={pred}")
    axes[0].axis("off")
    for c in range(7):
        ax = axes[c + 1]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.imshow(node_map(node_attn[c]), cmap="magma", alpha=0.55)
        ax.set_title(f"class {c}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
```

---

## 16. Legacy / Do Not Bring To New D5 Project

| Component | Bring? | Why not | Small useful part |
|---|---:|---|---|
| `legacy/*` | NO | Deprecated scripts/docs, old assumptions | Some visualization overlay ideas |
| `src/graph/*` old graph builder | NO | Duplicates canonical graph, different contrast semantics, numpy graph contract | Formula cross-check only |
| `src/data/pixel_motif_dataset.py` | NO | D5 does not use pixel_motif_dataset_v2, descriptor/top-K selected motifs | `build_subgraph_tensor_from_node_indices` idea if D5 later extracts hard subgraphs |
| `src/data/candidate_attention_dataset.py` | NO | D5 does not use candidate_attention_dataset_v1 or candidate slots | Train-only scaler pattern for future descriptor normalization |
| `scripts/precompute_pixel_candidate_subgraphs.py` | NO | D5 learns class-level pixel motif retrieval from full graph, not precomputed candidates | Topology helpers maybe reference only |
| `scripts/precompute_pixel_motif_dataset.py` | NO | Greedy top-K selection and old motif bank matching | Finite checks, output audit patterns |
| `src/motif/*`, `src/motif_v2/*` | NO | Old motif bank, cosine descriptor matching, discriminative metadata | None for D5 core |
| `src/graph/subgraph_descriptor.py` | NO | 41D descriptor pipeline (`4*node_dim + 3 + 2*edge_dim`) not D5 motif representation | `_safe_feature_stats` if doing analysis only |
| `src/models/learnable_slot_candidate_motif_gnn.py` | NO | D3.1 candidate slot model; wrong input contract | Attention diagnostics ideas only |
| `src/models/hierarchical_motif_gnn.py` | NO | Built around pixel motif dataset/hierarchical selected subgraphs | Internal subgraph encoder idea only |
| D4A slot pooling classifier | NO/PARTIAL | Generic adaptive slots are not class-level motif prototypes | Edge-aware GNN layer, entropy/smoothness formulas |
| `CombinedMotifLoss` / `MotifConsistencyLoss` | NO | Tied to old `motif_score_vector` | Margin idea only |

---

## 17. Proposed Clean Project Structure For D5

```text
fer_d5/
  configs/
    d5a_model.yaml
    d5a_experiment.yaml
  data/
    labels.py
    raw_dataset.py
    graph_types.py
    graph_config.py
    graph_builder.py
    graph_repository.py
    graph_resolver.py
    full_graph_dataset.py
  models/
    edge_gnn.py
    class_pixel_motif_graph_retrieval.py
    registry.py
  training/
    losses.py
    optimizer.py
    trainer.py
  evaluation/
    metrics.py
    evaluator.py
  visualization/
    visualize_d5_motifs.py
  scripts/
    build_graph_repo.py
    inspect_graph_repo.py
    debug_d5a_batch.py
    train_d5a.py
    evaluate_d5a.py
    visualize_d5.py
```

Module mapping:

| New module | Source in old repo | Action | Clean design note |
|---|---|---|---|
| `data/labels.py` | `data/raw_fer_dataset.py`, `src/data/emotions_dict.py` | REWRITE | Single canonical label names |
| `data/raw_dataset.py` | `data/raw_types.py`, `data/raw_fer_dataset.py` | COPY/PARTIAL | Keep `RawSample`, `RawFERDataset` |
| `data/graph_config.py` | `configs/graph_config.py` | COPY/PARTIAL | Move from `configs` to `data` or keep typed config |
| `data/graph_types.py` | `data/graph_types.py` | COPY | Same dataclasses |
| `data/graph_builder.py` | `data/shared_graph_builder.py`, `data/canonical_graph_builder.py` | COPY/PARTIAL | Combine or keep two builders |
| `data/graph_repository.py` | `data/graph_repository.py`, `data/chunked_graph_dataset.py` | COPY/PARTIAL | Relative manifest, chunk sizes |
| `data/graph_resolver.py` | `data/graph_resolver.py` | COPY | Critical |
| `data/full_graph_dataset.py` | `src/data/full_graph_dataset.py` | COPY/PARTIAL | D5 batch contract |
| `models/edge_gnn.py` | D4A `EdgeAwarePixelGNNLayer` | REWRITE RECOMMENDED | Standalone, optional vectorization |
| `models/class_pixel_motif_graph_retrieval.py` | New | WRITE NEW | D5 core |
| `training/losses.py` | `src/training/losses.py`, D4A formulas | COPY/PARTIAL + NEW | Weighted CE + D5 losses |
| `training/optimizer.py` | `src/training/optimizer.py` | COPY/PARTIAL | Clean config-only API |
| `training/trainer.py` | `src/training/trainer.py` | REWRITE RECOMMENDED | One full-graph path |
| `evaluation/metrics.py` | `src/evaluation/metrics.py` | COPY | Good enough |
| `evaluation/evaluator.py` | `src/evaluation/evaluator.py` | REWRITE/PARTIAL | Minimal full-graph evaluator |
| `visualization/visualize_d5_motifs.py` | `scripts/visualize_candidate_attention.py` | WRITE NEW | Heatmap not bbox-first |
| `scripts/build_graph_repo.py` | `scripts/build_graph_repository.py` | COPY/PARTIAL | Slim CLI |
| `scripts/debug_d5a_batch.py` | `scripts/debug_full_graph_d4_batch.py` | REWRITE/PARTIAL | D5 output shape asserts |
| `scripts/train_d5a.py` | `scripts/train.py` | REWRITE RECOMMENDED | Remove legacy routes |

---

## 18. Minimal Implementation Checklist For D5A

Data layer:

- [ ] Copy raw CSV reader.
- [ ] Copy canonical graph config/types/builders.
- [ ] Build graph repo from CSV.
- [ ] Inspect graph repo: shapes, finite values, feature names.
- [ ] Copy full graph dataset/collate.
- [ ] Confirm batch contract `[B,2304,7]`, `[2,17860]`, `[B,17860,5]`.

Model:

- [ ] Implement `EdgeAwarePixelGNNEncoder`.
- [ ] Implement class-level prototypes `T_0...T_6`.
- [ ] Compute soft node matching `node_attn [B,7,2304]`.
- [ ] Compute soft edge matching `edge_attn [B,7,E]`.
- [ ] Compute graph matching scores/logits `[B,7]`.
- [ ] Return diagnostic dict.

Loss:

- [ ] Weighted CE.
- [ ] Graph smoothness over `node_attn`.
- [ ] Edge-node closure.
- [ ] Area control.
- [ ] Prototype/class diversity.
- [ ] Finite checks in debug.

Training:

- [ ] Clean trainer for dict batches.
- [ ] Monitor `val_macro_f1`.
- [ ] Grad clipping.
- [ ] Early stopping.
- [ ] Checkpoint best.
- [ ] Smoke train with max batches.

Visualization:

- [ ] Class prototype/gate heatmaps.
- [ ] Per-sample true/pred/all-class attention grids.
- [ ] Top edge attention overlay.
- [ ] Deletion/insertion evaluation.

---

## 19. Commands / Checks After Building New Project

Build graph repo:

```bash
python scripts/build_graph_repo.py \
  --train_csv data/fer13-split/train.csv \
  --val_csv data/fer13-split/val.csv \
  --test_csv data/fer13-split/test.csv \
  --repo_root artifacts/graph_repo \
  --chunk_size 500 \
  --connectivity 8
```

Inspect graph repo:

```bash
python scripts/inspect_graph_repo.py --repo_root artifacts/graph_repo
```

Expected:

- `num_nodes = 2304`
- `edge_index = (2, 17860)`
- `edge_attr_static = (17860, 3)`
- first sample `node_features = (2304, 7)`
- first sample `edge_attr_dynamic = (17860, 2)`
- resolved `edge_attr = (17860, 5)`
- no NaN/Inf

Debug one D5 batch:

```bash
python scripts/debug_d5a_batch.py \
  --config configs/d5a_experiment.yaml \
  --graph_repo_path artifacts/graph_repo \
  --batch_size 2
```

Smoke train:

```bash
python scripts/train_d5a.py \
  --config configs/d5a_experiment.yaml \
  --epochs 2 \
  --max_train_batches 3 \
  --max_val_batches 2 \
  --max_test_batches 2 \
  --no_wandb
```

Full train:

```bash
python scripts/train_d5a.py --config configs/d5a_experiment.yaml
```

Visualize:

```bash
python scripts/visualize_d5.py \
  --checkpoint outputs/checkpoints/d5a/best.pth \
  --graph_repo_path artifacts/graph_repo \
  --split test \
  --max_samples 32 \
  --out_dir outputs/figures/d5a_attention
```

---

## Appendix A - Important Code Snippets

### A1. Raw CSV Minimal

```python
class RawFERDataset(Dataset):
    def _validate_columns(self):
        required = {"emotion", "pixels"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {missing}")
        self._df["emotion"] = self._df["emotion"].astype(int)
        if "Usage" not in self._df.columns:
            self._df["Usage"] = self.split

    def _parse_pixels(self, pixel_str):
        arr = np.fromstring(str(pixel_str), sep=" ", dtype=np.float32)
        if arr.size != self.image_size * self.image_size:
            raise ValueError("Pixel count mismatch")
        return arr.reshape(self.image_size, self.image_size)
```

### A2. Shared Edge Index Minimal

```python
def build_edge_index(H=48, W=48, connectivity=8):
    offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        offsets += [(-1,-1),(-1,1),(1,-1),(1,1)]
    rows, cols = [], []
    for y in range(H):
        for x in range(W):
            u = y * W + x
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    rows.append(u)
                    cols.append(ny * W + nx)
    return torch.tensor([rows, cols], dtype=torch.long)
```

### A3. Resolver Minimal

```python
class GraphResolver:
    def __init__(self, shared):
        self.shared = shared

    def resolve(self, sample):
        if sample.edge_attr_dynamic.shape[0] != self.shared.edge_index.shape[1]:
            raise ValueError("edge count mismatch")
        edge_attr = torch.cat(
            [self.shared.edge_attr_static, sample.edge_attr_dynamic],
            dim=1,
        )
        return ResolvedPixelGraph(
            graph_id=sample.graph_id,
            label=sample.label,
            split=sample.split,
            node_features=sample.node_features,
            edge_index=self.shared.edge_index,
            edge_attr=edge_attr,
            node_feature_names=sample.node_feature_names,
            edge_feature_names=self.shared.static_feature_names + sample.dynamic_feature_names,
        )
```

### A4. Full Graph Collate Minimal

```python
def collate_fn_full_graph(batch):
    x = torch.stack([b["node_features"] for b in batch])       # [B,N,7]
    edge_attr = torch.stack([b["edge_attr"] for b in batch])   # [B,E,5]
    y = torch.stack([b["y"] for b in batch])                   # [B]
    node_mask = torch.stack([b["node_mask"] for b in batch])   # [B,N]
    return {
        "graph_id": torch.tensor([int(b["graph_id"]) for b in batch]),
        "x": x,
        "node_features": x,
        "edge_index": batch[0]["edge_index"],                  # [2,E]
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "y": y,
        "label": y,
    }
```

### A5. Edge-Aware GNN Layer Minimal

```python
class EdgeAwarePixelGNNLayer(nn.Module):
    def forward(self, h, edge_index, edge_attr, node_mask):
        B, N, H = h.shape
        src = edge_index[0].long()
        dst = edge_index[1].long()
        E = src.numel()
        out = torch.zeros_like(h)
        deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(B, -1, -1)
        for b in range(B):
            e = self.edge_encoder(edge_attr[b])
            msg = self.message_mlp(torch.cat([h[b, src], e], dim=-1))
            out[b].index_add_(0, dst, msg)
            deg[b].index_add_(0, dst, torch.ones(E, 1, device=h.device, dtype=h.dtype))
        agg = out / deg.clamp_min(1.0)
        h = self.msg_norm(h + self.dropout(agg))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        return h * node_mask.unsqueeze(-1).to(h.dtype)
```

### A6. Class Weights

```python
def compute_class_weights(class_counts, normalize_mean=True, power=1.0):
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights.pow(float(power))
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-8)
    return weights
```

---

## Appendix B - Formula Summary

Image normalization:

```text
I = clip(raw / 255, 0, 1)
```

Coordinates:

```text
x_norm = x / (W - 1)
y_norm = y / (H - 1)
```

Gradients:

```text
gx[y,x] = (I[y,x+1] - I[y,x-1]) / 2      for interior x
gy[y,x] = (I[y+1,x] - I[y-1,x]) / 2      for interior y
grad_mag = sqrt(gx^2 + gy^2)
```

Local contrast:

```text
local_contrast[y,x] = abs(I[y,x] - mean(edge_padded_3x3_patch(y,x)))
```

Static edge features:

```text
dx = dst_x - src_x
dy = dst_y - src_y
dist = sqrt(dx^2 + dy^2)
```

Dynamic edge features:

```text
delta_intensity = abs(I_src - I_dst)
intensity_similarity = exp(-alpha * delta_intensity)
```

Edge-aware message passing:

```text
e_uv = EdgeEncoder(a_uv)
m_uv = MessageMLP([h_u, e_uv])
agg_v = mean_{u -> v}(m_uv)
h'_v = LayerNorm(h_v + Dropout(agg_v))
h''_v = LayerNorm(h'_v + Dropout(FFN(h'_v)))
```

Class weights:

```text
w_c = ((sum_j n_j) / (C * n_c))^power
w_c = w_c / mean(w)
```

Attention entropy:

```text
H(A) = -sum_k A_k log(A_k)
```

Graph smoothness:

```text
L_smooth = mean_{(u,v) in E, c} (A_c,u - A_c,v)^2
```

Edge-node closure:

```text
L_closure = mean ReLU(edge_attn_c,uv - min(node_attn_c,u, node_attn_c,v))^2
```

Area control:

```text
L_area = mean_{b,c} (mean_n node_attn_b,c,n - target_area)^2
```

Prototype diversity:

```text
L_div = mean_{c != c'} cos(proto_c, proto_c')^2
```

---

## Appendix C - Shape Contract Summary

Single graph from repository:

| Tensor | Shape | dtype | Source |
|---|---:|---|---|
| `node_features` | `[2304, 7]` | float32 | `PixelGraphSample` |
| `edge_attr_dynamic` | `[17860, 2]` | float32 | `PixelGraphSample` |
| `edge_index` | `[2, 17860]` | int64 | `SharedGraphStructure` |
| `edge_attr_static` | `[17860, 3]` | float32 | `SharedGraphStructure` |
| `edge_attr` resolved | `[17860, 5]` | float32 | `GraphResolver` |

Batch from full graph dataloader:

| Key | Shape | dtype | Notes |
|---|---:|---|---|
| `x` | `[B, 2304, 7]` | float32 | alias of `node_features` |
| `node_features` | `[B, 2304, 7]` | float32 | canonical |
| `edge_index` | `[2, 17860]` | int64 | shared, unbatched |
| `edge_attr` | `[B, 17860, 5]` | float32 | per-image dynamic included |
| `node_mask` | `[B, 2304]` | bool | currently all true |
| `y` | `[B]` | int64 | label |
| `label` | `[B]` | int64 | alias |
| `graph_id` | `[B]` | int64 | row index within split |

Expected D5 model output:

| Key | Shape | Notes |
|---|---:|---|
| `logits` | `[B, 7]` | classification scores |
| `node_attn` | `[B, 7, 2304]` | class-specific soft subgraph node membership |
| `edge_attn` | `[B, 7, 17860]` | class-specific soft subgraph edge membership |
| `class_scores` | `[B, 7]` | can be same as logits before calibration |
| `prototype_scores` | `[B, 7, K]` | optional if multiple prototypes per class |
| `aux_loss` | scalar | optional, or handled in loss module |

