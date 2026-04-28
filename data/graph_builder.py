"""Canonical FER-2013 pixel graph builder."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from data.graph_config import GraphConfig
from data.graph_types import PixelGraphSample, SharedGraphStructure
from data.raw_dataset import RawSample


def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    raise ValueError(f"Unsupported connectivity: {connectivity}")


class SharedGraphBuilder:
    """Build the directed shared 4/8-neighbor image grid topology."""

    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.height = int(config.height)
        self.width = int(config.width)

    def _node_id(self, y: int, x: int) -> int:
        return y * self.width + x

    def build(self) -> SharedGraphStructure:
        rows: List[int] = []
        cols: List[int] = []
        for y in range(self.height):
            for x in range(self.width):
                u = self._node_id(y, x)
                for dy, dx in _neighbor_offsets(self.config.connectivity):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        rows.append(u)
                        cols.append(self._node_id(ny, nx))

        edge_index_np = np.stack([rows, cols], axis=0).astype(np.int64)
        edge_index = torch.from_numpy(edge_index_np)
        edge_attr_static = torch.from_numpy(self._build_static_attrs(edge_index_np))
        expected = self.config.expected_edge_count()
        if edge_index.shape[1] != expected:
            raise ValueError(f"Unexpected edge count: expected {expected}, got {edge_index.shape[1]}")

        return SharedGraphStructure(
            height=self.height,
            width=self.width,
            connectivity=int(self.config.connectivity),
            edge_index=edge_index,
            edge_attr_static=edge_attr_static,
            static_feature_names=list(self.config.edge_static_feature_names),
            config_dict=self.config.to_dict(),
        )

    def _build_static_attrs(self, edge_index: np.ndarray) -> np.ndarray:
        src = edge_index[0]
        dst = edge_index[1]
        src_y = src // self.width
        src_x = src % self.width
        dst_y = dst // self.width
        dst_x = dst % self.width

        dx = (dst_x - src_x).astype(np.float32)
        dy = (dst_y - src_y).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        feature_map = {"dx": dx, "dy": dy, "dist": dist}
        return np.stack(
            [feature_map[name] for name in self.config.edge_static_feature_names],
            axis=1,
        ).astype(np.float32)


class PixelGraphBuilder:
    """Build per-image node features and dynamic edge attributes."""

    def __init__(self, config: GraphConfig, shared: SharedGraphStructure) -> None:
        self.config = config
        self.shared = shared
        self.height = int(config.height)
        self.width = int(config.width)
        self._src_ids = shared.edge_index[0].cpu().numpy().astype(np.int64)
        self._dst_ids = shared.edge_index[1].cpu().numpy().astype(np.int64)
        self._x_norm, self._y_norm = self._build_coord_grids()

    def build(self, raw_sample: RawSample) -> PixelGraphSample:
        image = self._normalize(raw_sample.image)
        node_features = torch.from_numpy(self._build_node_features(image))
        edge_attr_dynamic = torch.from_numpy(self._build_dynamic_edge_attrs(image))
        return PixelGraphSample(
            graph_id=int(raw_sample.sample_id),
            label=int(raw_sample.label),
            split=raw_sample.split,
            usage=raw_sample.usage,
            height=self.height,
            width=self.width,
            node_features=node_features,
            edge_attr_dynamic=edge_attr_dynamic,
            node_feature_names=list(self.config.node_feature_names),
            dynamic_feature_names=list(self.config.edge_dynamic_feature_names),
            metadata=dict(raw_sample.metadata),
        )

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.shape != (self.height, self.width):
            raise ValueError(f"Expected image shape {(self.height, self.width)}, got {img.shape}")
        if self.config.normalize_pixels:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.isfinite(img).all():
            raise ValueError("Non-finite values detected after image normalization")
        return img

    def _build_coord_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        yy, xx = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing="ij",
        )
        x_norm = xx / max(float(self.width - 1), 1.0)
        y_norm = yy / max(float(self.height - 1), 1.0)
        return x_norm.reshape(-1), y_norm.reshape(-1)

    def _build_node_features(self, image: np.ndarray) -> np.ndarray:
        gx, gy, grad_mag = self.compute_gradients(image)
        local_contrast = self.compute_local_contrast(image)
        feature_map = {
            "intensity": image.reshape(-1),
            "x_norm": self._x_norm,
            "y_norm": self._y_norm,
            "gx": gx.reshape(-1),
            "gy": gy.reshape(-1),
            "grad_mag": grad_mag.reshape(-1),
            "local_contrast": local_contrast.reshape(-1),
        }
        features = np.stack(
            [feature_map[name] for name in self.config.node_feature_names],
            axis=1,
        ).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite node features")
        return features

    @staticmethod
    def compute_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = np.zeros_like(img, dtype=np.float32)
        gy = np.zeros_like(img, dtype=np.float32)
        gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
        gx[:, 0] = img[:, 1] - img[:, 0]
        gx[:, -1] = img[:, -1] - img[:, -2]

        gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
        gy[0, :] = img[1, :] - img[0, :]
        gy[-1, :] = img[-1, :] - img[-2, :]

        gx = np.clip(gx, -1.0, 1.0).astype(np.float32)
        gy = np.clip(gy, -1.0, 1.0).astype(np.float32)
        grad_mag = np.sqrt(np.clip(gx * gx + gy * gy, a_min=0.0, a_max=None))
        grad_mag = np.clip(grad_mag, 0.0, 1.0).astype(np.float32)
        return gx, gy, grad_mag

    @staticmethod
    def compute_local_contrast(img: np.ndarray, window_size: int = 3) -> np.ndarray:
        pad = window_size // 2
        padded = np.pad(img, pad_width=pad, mode="edge")
        local_sum = np.zeros_like(img, dtype=np.float32)
        for dy in range(window_size):
            for dx in range(window_size):
                local_sum += padded[dy : dy + img.shape[0], dx : dx + img.shape[1]]
        local_mean = local_sum / float(window_size * window_size)
        return np.abs(img - local_mean).astype(np.float32)

    def _build_dynamic_edge_attrs(self, image: np.ndarray) -> np.ndarray:
        flat = image.reshape(-1)
        i_src = flat[self._src_ids]
        i_dst = flat[self._dst_ids]
        delta = np.abs(i_src - i_dst).astype(np.float32)
        alpha = float(self.config.intensity_similarity_alpha)
        feature_map = {
            "delta_intensity": delta,
            "intensity_similarity": np.exp(-alpha * delta).astype(np.float32),
        }
        attrs = np.stack(
            [feature_map[name] for name in self.config.edge_dynamic_feature_names],
            axis=1,
        ).astype(np.float32)
        if not np.isfinite(attrs).all():
            raise ValueError("Non-finite dynamic edge attributes")
        return attrs
