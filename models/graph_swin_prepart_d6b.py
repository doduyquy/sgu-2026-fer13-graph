"""D8A Graph-Swin pre-part context encoder for the D6B motif branch."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.dual_branch_graph_swin_motif import D6BPixelMotifBranch, SharedPixelEncoder


class GraphSwinWindowContext(nn.Module):
    """Broadcast regular and shifted window summaries back to pixel resolution."""

    def __init__(
        self,
        hidden_dim: int = 64,
        height: int = 48,
        width: int = 48,
        window_size: int = 6,
        shift_size: int = 3,
        window_heads: int = 4,
        use_window_mha: bool = False,
        context_merge: str = "regular_shifted_sum",
        context_proj: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.height = int(height)
        self.width = int(width)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.use_window_mha = bool(use_window_mha)
        self.context_merge = str(context_merge or "regular_shifted_sum")
        if self.height % self.window_size != 0 or self.width % self.window_size != 0:
            raise ValueError("GraphSwinWindowContext requires height/width divisible by window_size")
        self.num_win_h = self.height // self.window_size
        self.num_win_w = self.width // self.window_size
        self.num_windows = self.num_win_h * self.num_win_w

        if self.use_window_mha:
            self.window_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=int(window_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.window_attn_norm = nn.LayerNorm(self.hidden_dim)
            self.window_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(float(dropout)),
            )
            self.window_ffn_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.window_attn = None

        self.token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        if self.context_merge in {"regular_shifted_sum", "sum"}:
            fuse_in = self.hidden_dim
        elif self.context_merge in {"regular_shifted_concat", "concat"}:
            fuse_in = self.hidden_dim * 2
        else:
            raise ValueError(f"Unsupported context_merge: {self.context_merge}")
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.out_proj = (
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            if bool(context_proj)
            else nn.Identity()
        )

    def forward(self, h_pixel: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, num_nodes, hidden_dim = h_pixel.shape
        expected_nodes = self.height * self.width
        if num_nodes != expected_nodes:
            raise ValueError(f"Expected {expected_nodes} nodes, got {num_nodes}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}")

        h_grid = h_pixel.reshape(bsz, self.height, self.width, hidden_dim)
        reg_context = self._window_context_grid(h_grid)

        if self.shift_size > 0:
            shifted_grid = torch.roll(h_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_context = self._window_context_grid(shifted_grid)
            shifted_context = torch.roll(shifted_context, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_context = reg_context

        if self.context_merge in {"regular_shifted_sum", "sum"}:
            context = self.fuse(reg_context + shifted_context)
        else:
            context = self.fuse(torch.cat([reg_context, shifted_context], dim=-1))
        context = self.out_proj(context).reshape(bsz, expected_nodes, hidden_dim)
        if node_mask is not None:
            context = context * node_mask.to(dtype=context.dtype).unsqueeze(-1)
        return context

    def _window_context_grid(self, h_grid: torch.Tensor) -> torch.Tensor:
        windows = self._partition_windows(h_grid)
        bsz, num_windows, tokens_per_window, hidden_dim = windows.shape
        tokens = windows.reshape(bsz * num_windows, tokens_per_window, hidden_dim)
        if self.window_attn is not None:
            attn_out, _ = self.window_attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.window_attn_norm(tokens + attn_out)
            tokens = self.window_ffn_norm(tokens + self.window_ffn(tokens))
        window_tokens = tokens.mean(dim=1).reshape(bsz, num_windows, hidden_dim)
        window_tokens = self.token_proj(window_tokens)
        broadcast = window_tokens.unsqueeze(2).expand(-1, -1, tokens_per_window, -1)
        return self._unpartition_windows(broadcast)

    def _partition_windows(self, h_grid: torch.Tensor) -> torch.Tensor:
        bsz, height, width, hidden_dim = h_grid.shape
        ws = self.window_size
        windows = h_grid.contiguous().view(bsz, height // ws, ws, width // ws, ws, hidden_dim)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(bsz, self.num_windows, ws * ws, hidden_dim)

    def _unpartition_windows(self, windows: torch.Tensor) -> torch.Tensor:
        bsz, _, _, hidden_dim = windows.shape
        ws = self.window_size
        grid = windows.view(bsz, self.num_win_h, self.num_win_w, ws, ws, hidden_dim)
        grid = grid.permute(0, 1, 3, 2, 4, 5).contiguous()
        return grid.view(bsz, self.height, self.width, hidden_dim)


class GraphSwinPrePartD6BD8A(nn.Module):
    """Pixel graph -> Graph-Swin context -> enhanced pixels -> D6B motif branch."""

    def __init__(
        self,
        num_classes: int = 7,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        pixel_gnn_layers: int = 1,
        num_part_slots: int = 16,
        part_layers: int = 1,
        part_heads: int = 4,
        dropout: float = 0.2,
        use_part_position: bool = True,
        assignment_temperature: float = 1.0,
        return_attention: bool = True,
        height: int = 48,
        width: int = 48,
        connectivity: int = 8,
        prepart_context: Optional[Dict[str, Any]] = None,
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
        if self.num_nodes != self.height * self.width:
            raise ValueError(
                f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}"
            )

        ctx_cfg = dict(prepart_context or {})
        enabled = bool(ctx_cfg.pop("enabled", True))
        ctx_cfg.pop("type", None)
        ctx_cfg.pop("image_size", None)
        ctx_cfg.pop("region_merge", None)
        ctx_cfg.pop("use_region_transformer", None)
        ctx_cfg.pop("region_layers", None)
        ctx_cfg.pop("region_heads", None)
        alpha_init = float(ctx_cfg.pop("alpha_init", 0.1))
        alpha_learnable = bool(ctx_cfg.pop("alpha_learnable", True))
        ctx_cfg.setdefault("hidden_dim", self.hidden_dim)
        ctx_cfg.setdefault("height", self.height)
        ctx_cfg.setdefault("width", self.width)
        ctx_cfg.setdefault("dropout", min(float(dropout), 0.1))

        self.encoder = SharedPixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            pixel_gnn_layers=int(pixel_gnn_layers),
            dropout=dropout,
        )
        self.context_enabled = enabled
        self.context_encoder = GraphSwinWindowContext(**ctx_cfg) if enabled else nn.Identity()
        if alpha_learnable:
            self.context_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer("context_alpha", torch.tensor(alpha_init, dtype=torch.float32))
        self.d6_branch = D6BPixelMotifBranch(
            num_classes=self.num_classes,
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            num_part_slots=int(num_part_slots),
            part_layers=int(part_layers),
            part_heads=int(part_heads),
            dropout=dropout,
            use_part_position=bool(use_part_position),
            assignment_temperature=float(assignment_temperature),
            return_attention=bool(return_attention),
            height=self.height,
            width=self.width,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSwinPrePartD6BD8A":
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
            "use_class_part_attention",
        ):
            cfg.pop(legacy_key, None)

        pixel_cfg = dict(cfg.pop("pixel_encoder", {}) or {})
        if "num_layers" in pixel_cfg:
            cfg.setdefault("pixel_gnn_layers", int(pixel_cfg["num_layers"]))
        if "pixel_gnn_layers" in pixel_cfg:
            cfg.setdefault("pixel_gnn_layers", int(pixel_cfg["pixel_gnn_layers"]))
        if "dropout" in pixel_cfg:
            cfg.setdefault("dropout", float(pixel_cfg["dropout"]))

        part_cfg = dict(cfg.pop("part_branch", {}) or {})
        aliases = {
            "num_parts": "num_part_slots",
            "num_slots": "num_part_slots",
            "part_self_attention_heads": "part_heads",
            "self_attention_heads": "part_heads",
        }
        for old_key, new_key in aliases.items():
            if old_key in part_cfg:
                cfg.setdefault(new_key, part_cfg[old_key])
        for key in (
            "num_part_slots",
            "part_layers",
            "part_heads",
            "use_part_position",
            "assignment_temperature",
            "return_attention",
        ):
            if key in part_cfg:
                cfg.setdefault(key, part_cfg[key])
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
            raise KeyError("GraphSwinPrePartD6BD8A needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("GraphSwinPrePartD6BD8A requires edge_index and edge_attr")
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")

        h_pixel = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        if self.context_enabled:
            h_context = self.context_encoder(h_pixel, node_mask=node_mask)
        else:
            h_context = torch.zeros_like(h_pixel)
        enhanced_h_pixel = h_pixel + self.context_alpha.to(dtype=h_pixel.dtype, device=h_pixel.device) * h_context
        if node_mask is not None:
            enhanced_h_pixel = enhanced_h_pixel * node_mask.to(dtype=enhanced_h_pixel.dtype).unsqueeze(-1)

        branch_out = self.d6_branch(enhanced_h_pixel, node_mask=node_mask)
        logits = branch_out["logits_d6"]
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "logits_d6": logits,
            "h_pixel": h_pixel,
            "pixel_embeddings": h_pixel,
            "h_context": h_context,
            "enhanced_h_pixel": enhanced_h_pixel,
            "context_alpha": self.context_alpha,
        }
        out.update(branch_out)
        out["logits"] = logits
        out["class_pixel_motif"] = self._class_pixel_motif(out)
        out["diagnostics"] = self._diagnostics(out)
        return out

    @staticmethod
    def _class_pixel_motif(out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        class_part_attn = out.get("class_part_attn")
        part_masks = out.get("part_masks")
        if not torch.is_tensor(class_part_attn) or not torch.is_tensor(part_masks):
            return None
        return torch.einsum("bck,bkn->bcn", class_part_attn, part_masks)

    @staticmethod
    def _diagnostics(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        h_pixel = out["h_pixel"]
        h_context = out["h_context"]
        enhanced = out["enhanced_h_pixel"]
        pixel_norm = h_pixel.float().norm(dim=-1).mean()
        context_norm = h_context.float().norm(dim=-1).mean()
        enhanced_norm = enhanced.float().norm(dim=-1).mean()
        alpha = out["context_alpha"].float()
        diagnostics["context_alpha"] = alpha.detach()
        diagnostics["context_norm"] = context_norm.detach()
        diagnostics["h_pixel_norm"] = pixel_norm.detach()
        diagnostics["enhanced_norm"] = enhanced_norm.detach()
        diagnostics["context_to_pixel_ratio"] = (context_norm / pixel_norm.clamp_min(1e-6)).detach()

        part_masks = out.get("part_masks")
        if torch.is_tensor(part_masks):
            m = F.normalize(part_masks.float(), dim=2, eps=1e-6)
            sim = torch.bmm(m, m.transpose(1, 2))
            k = sim.shape[1]
            off = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0))
            diagnostics["slot_div"] = off.mean().detach()
            slot_area = out.get("slot_area")
            if torch.is_tensor(slot_area):
                area_norm = slot_area.float() / slot_area.float().sum(dim=1, keepdim=True).clamp_min(1e-6)
                diagnostics["slot_area_entropy"] = (
                    -(area_norm * area_norm.clamp_min(1e-6).log()).sum(dim=1).mean().detach()
                )
                diagnostics["slot_area_mean"] = slot_area.float().mean().detach()
        border_mass = out.get("border_mass_per_slot", out.get("border_mass"))
        if torch.is_tensor(border_mass):
            diagnostics["border_mass_mean"] = border_mass.float().mean().detach()
        class_part_attn = out.get("class_part_attn")
        if torch.is_tensor(class_part_attn):
            entropy = -(class_part_attn.float() * class_part_attn.float().clamp_min(1e-6).log()).sum(dim=2)
            diagnostics["class_part_entropy"] = entropy.mean().detach()
        part_attn = out.get("part_attn")
        if torch.is_tensor(part_attn):
            diagnostics["part_attn_std"] = part_attn.float().std(unbiased=False).detach()
        return diagnostics
