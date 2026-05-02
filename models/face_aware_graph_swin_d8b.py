"""D8B face-aware Graph-Swin model.

This module keeps the D7 Graph-Swin region-transformer backbone, then adds
soft gates at pixel, window, and region levels. It intentionally does not use
D6/D8A part slots.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.dual_branch_graph_swin_motif import SharedPixelEncoder, WindowAttentionPooling


class _GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class FaceAwareGraphSwinBranchD8B(nn.Module):
    """D7 Graph-Swin branch with pixel/window/region gates."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 64,
        height: int = 48,
        width: int = 48,
        window_size: int = 6,
        shift_size: int = 3,
        window_heads: int = 4,
        use_window_mha: bool = False,
        region_merge: bool = True,
        class_head_type: str = "attn_only",
        use_region_transformer: bool = True,
        region_layers: int = 1,
        region_heads: int = 4,
        dropout: float = 0.2,
        face_gate: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        gate_cfg = dict(face_gate or {})
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.height = int(height)
        self.width = int(width)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.region_merge = bool(region_merge)
        self.class_head_type = str(class_head_type or "attn_only")
        self.use_region_transformer = bool(use_region_transformer)
        self.use_pixel_gate = bool(gate_cfg.get("use_pixel_gate", True))
        self.use_window_gate = bool(gate_cfg.get("use_window_gate", True))
        self.use_region_gate = bool(gate_cfg.get("use_region_gate", True))
        self.window_gate_residual = bool(gate_cfg.get("window_gate_residual", True))
        self.region_gate_residual = bool(gate_cfg.get("region_gate_residual", True))
        if self.class_head_type not in {"attn_only", "attn_plus_mean"}:
            raise ValueError(f"Unsupported graph_swin.class_head_type: {self.class_head_type}")
        if self.height % self.window_size != 0 or self.width % self.window_size != 0:
            raise ValueError("FaceAwareGraphSwinBranchD8B requires height/width divisible by window_size")

        self.num_win_h = self.height // self.window_size
        self.num_win_w = self.width // self.window_size
        self.num_windows = self.num_win_h * self.num_win_w
        self.eps = float(gate_cfg.get("eps", 1e-6))

        self.fallback_window_encoder = WindowAttentionPooling(
            hidden_dim=self.hidden_dim,
            num_heads=int(window_heads),
            dropout=dropout,
            use_window_mha=bool(use_window_mha),
        )
        self.window_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.window_gate = _GateMLP(
            in_dim=self.hidden_dim,
            hidden_dim=int(gate_cfg.get("window_gate_hidden", max(1, self.hidden_dim // 2))),
            dropout=dropout,
        )
        self.region_merge_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.region_gate = _GateMLP(
            in_dim=self.hidden_dim,
            hidden_dim=int(gate_cfg.get("region_gate_hidden", max(1, self.hidden_dim // 2))),
            dropout=dropout,
        )
        beta_window = torch.tensor(float(gate_cfg.get("beta_window", 0.5)), dtype=torch.float32)
        beta_region = torch.tensor(float(gate_cfg.get("beta_region", 0.5)), dtype=torch.float32)
        if bool(gate_cfg.get("beta_window_learnable", False)):
            self.beta_window = nn.Parameter(beta_window)
        else:
            self.register_buffer("beta_window", beta_window, persistent=True)
        if bool(gate_cfg.get("beta_region_learnable", False)):
            self.beta_region = nn.Parameter(beta_region)
        else:
            self.register_buffer("beta_region", beta_region, persistent=True)

        if self.use_region_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(region_heads),
                dim_feedforward=self.hidden_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
            )
            self.region_transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=int(region_layers),
            )
        else:
            self.region_transformer = None

        self.class_queries = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.region_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.region_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.region_mean_proj = (
            nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.class_head_type == "attn_plus_mean"
            else None
        )
        self.class_logit_head = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_queries, mean=0.0, std=0.02)

    def forward(
        self,
        h_pixel: torch.Tensor,
        pixel_gate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, hidden_dim = h_pixel.shape
        expected_nodes = self.height * self.width
        if num_nodes != expected_nodes:
            raise ValueError(f"Expected {expected_nodes} nodes, got {num_nodes}")
        h_grid = h_pixel.reshape(bsz, self.height, self.width, hidden_dim)

        if self.use_pixel_gate and torch.is_tensor(pixel_gate):
            gate_grid = pixel_gate.reshape(bsz, self.height, self.width, 1)
            regular_windows = self._partition_windows(h_grid)
            regular_gates = self._partition_gate_windows(gate_grid)
            reg_tokens = self._weighted_pool(regular_windows, regular_gates)
            reg_attn = self._normalized_gate_attn(regular_gates)

            h_shift = torch.roll(h_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            gate_shift = torch.roll(gate_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_windows = self._partition_windows(h_shift)
            shifted_gates = self._partition_gate_windows(gate_shift)
            shift_tokens = self._weighted_pool(shifted_windows, shifted_gates)
            shift_attn = self._normalized_gate_attn(shifted_gates)
        else:
            regular_windows = self._partition_windows(h_grid)
            reg_tokens, reg_attn = self.fallback_window_encoder(regular_windows)
            h_shift = torch.roll(h_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_windows = self._partition_windows(h_shift)
            shift_tokens, shift_attn = self.fallback_window_encoder(shifted_windows)

        window_tokens = self.window_fuse(torch.cat([reg_tokens, shift_tokens], dim=-1))
        window_gate = self.window_gate(window_tokens) if self.use_window_gate else torch.ones_like(window_tokens[..., :1])
        gated_window_tokens = self._apply_gate(
            window_tokens,
            window_gate,
            beta=self.beta_window.to(device=window_tokens.device, dtype=window_tokens.dtype),
            residual=self.window_gate_residual,
        )

        merged_region_tokens = self._merge_regions(gated_window_tokens)
        region_gate = self.region_gate(merged_region_tokens) if self.use_region_gate else torch.ones_like(merged_region_tokens[..., :1])
        gated_region_tokens = self._apply_gate(
            merged_region_tokens,
            region_gate,
            beta=self.beta_region.to(device=merged_region_tokens.device, dtype=merged_region_tokens.dtype),
            residual=self.region_gate_residual,
        )
        region_context = gated_region_tokens
        if self.region_transformer is not None:
            region_context = self.region_transformer(region_context)

        class_region_attn, class_repr_swin, logits_swin = self._class_region_attention(region_context)
        return {
            "logits_swin": logits_swin,
            "regular_window_tokens": reg_tokens,
            "shifted_window_tokens": shift_tokens,
            "window_tokens": gated_window_tokens,
            "window_tokens_raw": window_tokens,
            "regular_window_attn": reg_attn,
            "shifted_window_attn": shift_attn,
            "region_tokens": region_context,
            "region_tokens_raw": merged_region_tokens,
            "region_context": region_context,
            "gated_region_tokens": gated_region_tokens,
            "class_region_attn": class_region_attn,
            "class_repr_swin": class_repr_swin,
            "window_gate": window_gate,
            "region_gate": region_gate,
        }

    @staticmethod
    def _apply_gate(
        tokens: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        residual: bool,
    ) -> torch.Tensor:
        if residual:
            return tokens * (1.0 + beta * gate)
        return tokens * gate

    def _partition_windows(self, h_grid: torch.Tensor) -> torch.Tensor:
        bsz, height, width, hidden_dim = h_grid.shape
        ws = self.window_size
        windows = h_grid.view(bsz, height // ws, ws, width // ws, ws, hidden_dim)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(bsz, self.num_windows, ws * ws, hidden_dim)

    def _partition_gate_windows(self, gate_grid: torch.Tensor) -> torch.Tensor:
        bsz, height, width, channels = gate_grid.shape
        ws = self.window_size
        windows = gate_grid.view(bsz, height // ws, ws, width // ws, ws, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(bsz, self.num_windows, ws * ws, channels)

    def _weighted_pool(self, windows: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        weights = gates.to(dtype=windows.dtype)
        denom = weights.sum(dim=2).clamp_min(self.eps)
        return (windows * weights).sum(dim=2) / denom

    def _normalized_gate_attn(self, gates: torch.Tensor) -> torch.Tensor:
        weights = gates.squeeze(-1)
        return weights / weights.sum(dim=2, keepdim=True).clamp_min(self.eps)

    def _merge_regions(self, window_tokens: torch.Tensor) -> torch.Tensor:
        bsz = window_tokens.shape[0]
        grid = window_tokens.view(bsz, self.num_win_h, self.num_win_w, self.hidden_dim)
        if not self.region_merge:
            return grid.reshape(bsz, self.num_windows, self.hidden_dim)
        if self.num_win_h % 2 != 0 or self.num_win_w % 2 != 0:
            raise ValueError("2x2 region merge requires an even window grid")
        merged = grid.view(
            bsz,
            self.num_win_h // 2,
            2,
            self.num_win_w // 2,
            2,
            self.hidden_dim,
        )
        merged = merged.permute(0, 1, 3, 2, 4, 5).contiguous()
        merged = merged.view(bsz, (self.num_win_h // 2) * (self.num_win_w // 2), self.hidden_dim * 4)
        return self.region_merge_proj(merged)

    def _class_region_attention(
        self,
        region_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_region = self.region_key(region_tokens)
        v_region = self.region_value(region_tokens)
        scores = torch.einsum("ch,brh->bcr", self.class_queries, k_region)
        scores = scores / math.sqrt(float(self.hidden_dim))
        class_region_attn = torch.softmax(scores, dim=2)
        class_repr = torch.einsum("bcr,brh->bch", class_region_attn, v_region)
        if self.region_mean_proj is not None:
            mean_repr = self.region_mean_proj(region_tokens.mean(dim=1))
            class_repr = class_repr + mean_repr.unsqueeze(1)
        logits = self.class_logit_head(class_repr).squeeze(-1)
        return class_region_attn, class_repr, logits


class FaceAwareGraphSwinD8B(nn.Module):
    """D8B: pixel graph encoder plus face-aware Graph-Swin."""

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        pixel_gnn_layers: int = 1,
        dropout: float = 0.2,
        height: int = 48,
        width: int = 48,
        connectivity: int = 8,
        mode: str = "swin_only",
        graph_swin: Optional[Dict[str, Any]] = None,
        face_gate: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del connectivity
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.height = int(height)
        self.width = int(width)
        self.mode = str(mode)
        if self.mode != "swin_only":
            raise ValueError("D8B only supports mode='swin_only'; it does not use D6B part slots")
        if self.num_nodes != self.height * self.width:
            raise ValueError(
                f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}"
            )

        gate_cfg = dict(face_gate or {})
        self.face_gate_enabled = bool(gate_cfg.get("enabled", True))
        self.pixel_gate_use_xy = bool(gate_cfg.get("pixel_gate_use_xy", True))
        self.pixel_gate_use_low_level = bool(gate_cfg.get("pixel_gate_use_low_level", True))

        self.encoder = SharedPixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            pixel_gnn_layers=int(pixel_gnn_layers),
            dropout=dropout,
        )
        self.register_buffer("pixel_positions", self._make_positions(), persistent=False)
        self.register_buffer("border_mask", self._make_border_mask(border_width=3), persistent=False)
        pixel_gate_in = self.hidden_dim
        if self.pixel_gate_use_xy:
            pixel_gate_in += 2
        if self.pixel_gate_use_low_level:
            pixel_gate_in += self.node_dim
        self.pixel_gate = _GateMLP(
            in_dim=pixel_gate_in,
            hidden_dim=int(gate_cfg.get("pixel_gate_hidden", max(1, self.hidden_dim // 2))),
            dropout=dropout,
        )

        swin_cfg = dict(graph_swin or {})
        swin_cfg.setdefault("use_region_transformer", True)
        swin_cfg.setdefault("region_layers", 1)
        swin_cfg.setdefault("region_heads", 4)
        swin_cfg.setdefault("class_head_type", "attn_only")
        self.swin_branch = FaceAwareGraphSwinBranchD8B(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            height=self.height,
            width=self.width,
            dropout=dropout,
            face_gate=gate_cfg,
            **swin_cfg,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FaceAwareGraphSwinD8B":
        cfg = dict(config)
        for legacy_key in (
            "edge_hidden_dim",
            "gnn_layers",
            "use_edge_gnn",
            "temperature",
            "edge_score_weight",
            "num_edges",
            "motif_prior_path",
            "init_node_gate_from_prior",
            "prior_init_clamp_min",
            "prior_init_clamp_max",
        ):
            cfg.pop(legacy_key, None)
        return cls(**cfg)

    def forward(
        self,
        batch_or_x,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del y
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            node_mask = batch.get("node_mask")
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("FaceAwareGraphSwinD8B needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("FaceAwareGraphSwinD8B requires edge_index and edge_attr")
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        h_pixel = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        pixel_gate = self._compute_pixel_gate(h_pixel, x, node_mask=node_mask)
        out = self.swin_branch(h_pixel, pixel_gate=pixel_gate)
        out["logits"] = out["logits_swin"]
        out["logits_fused"] = out["logits_swin"]
        out["h_pixel"] = h_pixel
        out["pixel_embeddings"] = h_pixel
        out["pixel_gate"] = pixel_gate
        out["diagnostics"] = self._diagnostics(out)
        return out

    def _compute_pixel_gate(
        self,
        h_pixel: torch.Tensor,
        x: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.face_gate_enabled:
            gate = torch.ones_like(h_pixel[..., :1])
        else:
            pieces = [h_pixel]
            if self.pixel_gate_use_xy:
                pos = self.pixel_positions.to(device=h_pixel.device, dtype=h_pixel.dtype)
                pieces.append(pos.unsqueeze(0).expand(h_pixel.shape[0], -1, -1))
            if self.pixel_gate_use_low_level:
                pieces.append(x.to(device=h_pixel.device, dtype=h_pixel.dtype))
            gate = self.pixel_gate(torch.cat(pieces, dim=-1))
        if node_mask is not None:
            gate = gate * node_mask.to(device=gate.device, dtype=gate.dtype).unsqueeze(-1)
        return gate

    def _make_positions(self) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, self.height)
        xs = torch.linspace(0.0, 1.0, self.width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()

    def _make_border_mask(self, border_width: int) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)

    def _diagnostics(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        pixel_gate = out.get("pixel_gate")
        if torch.is_tensor(pixel_gate):
            gate = pixel_gate.detach().float()
            diagnostics["pixel_gate_mean"] = gate.mean()
            diagnostics["pixel_gate_std"] = gate.std(unbiased=False)
            diagnostics["pixel_gate_min"] = gate.min()
            diagnostics["pixel_gate_max"] = gate.max()
            diagnostics["pixel_gate_area"] = gate.mean()
            entropy = -(
                gate.clamp_min(1e-6).log() * gate
                + (1.0 - gate).clamp_min(1e-6).log() * (1.0 - gate)
            )
            diagnostics["pixel_gate_entropy"] = entropy.mean()
            border = self.border_mask.to(device=gate.device, dtype=torch.bool)
            diagnostics["pixel_gate_border_mean"] = gate[:, border, :].mean()
            diagnostics["pixel_gate_center_mean"] = gate[:, ~border, :].mean()
        window_gate = out.get("window_gate")
        if torch.is_tensor(window_gate):
            gate = window_gate.detach().float()
            diagnostics["window_gate_mean"] = gate.mean()
            diagnostics["window_gate_std"] = gate.std(unbiased=False)
            diagnostics["window_gate_min"] = gate.min()
            diagnostics["window_gate_max"] = gate.max()
        region_gate = out.get("region_gate")
        if torch.is_tensor(region_gate):
            gate = region_gate.detach().float()
            diagnostics["region_gate_mean"] = gate.mean()
            diagnostics["region_gate_std"] = gate.std(unbiased=False)
            diagnostics["region_gate_min"] = gate.min()
            diagnostics["region_gate_max"] = gate.max()
        region_tokens = out.get("region_tokens")
        if torch.is_tensor(region_tokens):
            diagnostics["region_token_norm"] = region_tokens.detach().float().norm(dim=-1).mean()
        class_region_attn = out.get("class_region_attn")
        if torch.is_tensor(class_region_attn):
            attn = class_region_attn.detach().float()
            entropy = -(attn * attn.clamp_min(1e-6).log()).sum(dim=2)
            diagnostics["class_region_entropy_mean"] = entropy.mean()
        return diagnostics
