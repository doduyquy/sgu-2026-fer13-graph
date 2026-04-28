"""Raw FER-2013 CSV dataset.

The raw layer intentionally keeps images in the original FER-2013 pixel range
``[0, 255]``. Normalization belongs to the graph builder so all graph features
share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class RawSample:
    """One parsed FER-2013 row.

    Attributes:
        sample_id: Row index inside the split CSV.
        label: Integer FER-2013 class id in ``[0, 6]``.
        split: Logical split name, usually ``train``, ``val``, or ``test``.
        usage: Original Usage column if present, otherwise the split name.
        image: ``float32`` array with shape ``[48, 48]`` and raw range
            ``[0, 255]``.
    """

    sample_id: int
    label: int
    split: str
    usage: str
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class RawFERDataset(Dataset):
    """Read a FER-2013 split CSV with columns ``emotion`` and ``pixels``."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_size: int = 48,
        max_samples: Optional[int] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = str(split)
        self.image_size = int(image_size)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self._df = pd.read_csv(self.csv_path)
        self._validate_columns()
        if max_samples is not None:
            self._df = self._df.iloc[: int(max_samples)].reset_index(drop=True)

    def _validate_columns(self) -> None:
        required = {"emotion", "pixels"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {sorted(missing)}")
        self._df["emotion"] = self._df["emotion"].astype(int)
        if "Usage" not in self._df.columns:
            self._df["Usage"] = self.split

    def _parse_pixels(self, pixel_str: str) -> np.ndarray:
        expected = self.image_size * self.image_size
        arr = np.fromstring(str(pixel_str), sep=" ", dtype=np.float32)
        if arr.size != expected:
            raise ValueError(f"Pixel count mismatch: expected {expected}, got {arr.size}")
        return arr.reshape(self.image_size, self.image_size)

    def __len__(self) -> int:
        return int(len(self._df))

    def __getitem__(self, idx: int) -> RawSample:
        row = self._df.iloc[int(idx)]
        image = self._parse_pixels(row["pixels"])
        return RawSample(
            sample_id=int(idx),
            label=int(row["emotion"]),
            split=self.split,
            usage=str(row["Usage"]),
            image=image,
            metadata={"source_csv": str(self.csv_path)},
        )

    def class_counts(self, num_classes: int = 7) -> np.ndarray:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        values = self._df["emotion"].astype(int).to_numpy()
        for label in values:
            counts[int(label)] += 1
        return counts
