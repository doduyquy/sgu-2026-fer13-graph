"""Motif discovery module for D8M Stage-1 debug/audit."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from models.dual_branch_graph_swin_motif import SharedPixelEncoder
from utils.motif_audit import (
    audit_motif_outputs,
    compute_border_center_mass,
    compute_motif_area,
    compute_motif_centers,
)


class MotifDiscoveryModule(nn.Module):
    """This module is for motif discovery/audit pretraining; not used as main classifier yet."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_motifs: int = 16,
        image_hw: Tuple[int, int] = (48, 48),
        dropout: float = 0.1,
        attention_temperature: float = 1.0,
        use_learnable_logit_scale: bool = False,
        init_logit_scale: float = 1.0,
        max_logit_scale: float = 10.0,
        use_spatial_query_bias: bool = False,
        spatial_bias_type: str = "gaussian",
        spatial_bias_sigma: float = 0.25,
        spatial_bias_strength: float = 0.0,
        spatial_bias_decay: bool = False,
        spatial_bias_decay_epochs: int = 20,
        spatial_bias_min_strength: float = 0.0,
        spatial_grid_min: float = 0.0,
        spatial_grid_max: float = 1.0,
        border_width: int = 4,
        map_sim_threshold: float = 0.80,
        emb_sim_threshold: float = 0.80,
        center_distance_threshold: float = 0.15,
        min_effective_area: float = 40.0,
        max_effective_area: float = 400.0,
        max_border_mass: float = 0.45,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_motifs = int(num_motifs)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.attention_temperature = float(attention_temperature)
        if self.attention_temperature <= 0.0:
            raise ValueError(f"attention_temperature must be > 0, got {attention_temperature}")
        self.use_learnable_logit_scale = bool(use_learnable_logit_scale)
        self.max_logit_scale = float(max_logit_scale)
        if self.max_logit_scale <= 0.0:
            raise ValueError(f"max_logit_scale must be > 0, got {max_logit_scale}")
        init_scale = max(float(init_logit_scale), 1e-6)
        if self.use_learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(init_scale), dtype=torch.float32), persistent=True)
        self.use_spatial_query_bias = bool(use_spatial_query_bias)
        self.spatial_bias_type = str(spatial_bias_type or "gaussian").lower()
        if self.spatial_bias_type != "gaussian":
            raise ValueError(f"Unsupported spatial_bias_type: {self.spatial_bias_type}")
        self.spatial_bias_sigma = float(spatial_bias_sigma)
        if self.spatial_bias_sigma <= 0.0:
            raise ValueError(f"spatial_bias_sigma must be > 0, got {spatial_bias_sigma}")
        self.spatial_bias_decay = bool(spatial_bias_decay)
        self.spatial_bias_decay_epochs = int(spatial_bias_decay_epochs)
        self.spatial_bias_min_strength = float(spatial_bias_min_strength)
        self.spatial_grid_min = float(spatial_grid_min)
        self.spatial_grid_max = float(spatial_grid_max)
        if not (0.0 <= self.spatial_grid_min < self.spatial_grid_max <= 1.0):
            raise ValueError(
                "spatial_grid_min/max must satisfy 0 <= min < max <= 1, "
                f"got {self.spatial_grid_min}, {self.spatial_grid_max}"
            )
        self.border_width = int(border_width)
        self.map_sim_threshold = float(map_sim_threshold)
        self.emb_sim_threshold = float(emb_sim_threshold)
        self.center_distance_threshold = float(center_distance_threshold)
        self.min_effective_area = float(min_effective_area)
        self.max_effective_area = float(max_effective_area)
        self.max_border_mass = float(max_border_mass)

        self.motif_queries = nn.Parameter(torch.empty(self.num_motifs, self.hidden_dim))
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("pixel_positions", self._make_pixel_positions(), persistent=False)
        self.register_buffer("query_init_centers", self._make_query_centers(), persistent=True)
        self.register_buffer(
            "spatial_bias_strength_current",
            torch.tensor(float(spatial_bias_strength), dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.motif_queries)

    def set_spatial_bias_strength(self, strength: float) -> None:
        value = max(float(strength), 0.0)
        self.spatial_bias_strength_current.fill_(value)

    def _make_pixel_positions(self) -> torch.Tensor:
        height, width = self.image_hw
        ys = torch.linspace(0.0, 1.0, height)
        xs = torch.linspace(0.0, 1.0, width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()

    def _make_query_centers(self) -> torch.Tensor:
        cols = int(math.ceil(math.sqrt(float(self.num_motifs))))
        rows = int(math.ceil(float(self.num_motifs) / float(cols)))
        ys = torch.linspace(self.spatial_grid_min, self.spatial_grid_max, rows)
        xs = torch.linspace(self.spatial_grid_min, self.spatial_grid_max, cols)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return centers[: self.num_motifs].float()

    def _spatial_query_bias(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (height, width) == self.image_hw:
            positions = self.pixel_positions.to(device=device, dtype=dtype)
        else:
            ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            positions = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        centers = self.query_init_centers.to(device=device, dtype=dtype)
        dist2 = (centers[:, None, :] - positions[None, :, :]).square().sum(dim=-1)
        bias = -dist2 / (2.0 * self.spatial_bias_sigma * self.spatial_bias_sigma)
        return bias.unsqueeze(0)

    def forward(
        self,
        h_pixel: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
        node_mask: Optional[torch.Tensor] = None,
        foreground_prior: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del node_features
        if h_pixel.ndim != 3:
            raise ValueError(f"h_pixel must be [B, N, D], got {tuple(h_pixel.shape)}")
        bsz, num_nodes, hidden_dim = h_pixel.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}")
        height, width = image_hw or self.image_hw
        height, width = int(height), int(width)
        if num_nodes != height * width:
            raise ValueError(f"Expected N={height * width} for image_hw={(height, width)}, got {num_nodes}")

        q = self.query_proj(self.motif_queries).unsqueeze(0).expand(bsz, -1, -1)
        k = self.key_proj(h_pixel)
        v = self.value_proj(h_pixel)
        logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(float(self.hidden_dim))
        logits = logits / self.attention_temperature
        logit_scale_value = torch.exp(
            torch.clamp(
                self.logit_scale.to(device=logits.device, dtype=logits.dtype),
                max=math.log(self.max_logit_scale),
            )
        )
        if self.use_learnable_logit_scale:
            logits = logits * logit_scale_value
        if self.use_spatial_query_bias:
            strength = self.spatial_bias_strength_current.to(device=logits.device, dtype=logits.dtype)
            logits = logits + strength * self._spatial_query_bias(height, width, dtype=logits.dtype, device=logits.device)
        if foreground_prior is not None:
            prior = foreground_prior
            if prior.ndim == 3 and prior.shape[1:] == (height, width):
                prior = prior.flatten(1)
            if prior.ndim == 3 and prior.shape[-1] == 1:
                prior = prior.squeeze(-1)
            if prior.shape != (bsz, num_nodes):
                raise ValueError(f"foreground_prior must broadcast to [B, N], got {tuple(foreground_prior.shape)}")
            logits = logits + prior.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6).log().unsqueeze(1)
        if node_mask is not None:
            logits = logits.masked_fill(~node_mask.bool().unsqueeze(1), -1e4)

        attn = torch.softmax(logits, dim=2)
        if node_mask is not None:
            attn = attn * node_mask.to(device=attn.device, dtype=attn.dtype).unsqueeze(1)
            attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-8)

        motif_embeddings = self.out_norm(torch.bmm(attn, v))
        motif_assignment_maps = attn.reshape(bsz, self.num_motifs, height, width)
        assignment_flat = motif_assignment_maps.flatten(2)
        assignment_entropy = -(
            assignment_flat * assignment_flat.clamp_min(1e-8).log()
        ).sum(dim=2)
        motif_scores = self.score_head(motif_embeddings).squeeze(-1)
        motif_centers = compute_motif_centers(motif_assignment_maps, normalized=True)
        motif_area = compute_motif_area(motif_assignment_maps)
        motif_border_mass, motif_center_mass = compute_border_center_mass(
            motif_assignment_maps,
            border_width=self.border_width,
        )
        audit = audit_motif_outputs(
            motif_assignment_maps,
            motif_embeddings,
            motif_scores=motif_scores,
            motif_centers=motif_centers,
            border_width=self.border_width,
            map_sim_threshold=self.map_sim_threshold,
            emb_sim_threshold=self.emb_sim_threshold,
            center_distance_threshold=self.center_distance_threshold,
            min_effective_area=self.min_effective_area,
            max_effective_area=self.max_effective_area,
            max_border_mass=self.max_border_mass,
        )
        audit.update(
            {
                "attention_temperature": torch.tensor(self.attention_temperature, device=h_pixel.device, dtype=h_pixel.dtype),
                "logit_scale_value": logit_scale_value.detach().to(dtype=h_pixel.dtype),
                "logits_mean": logits.detach().float().mean().to(dtype=h_pixel.dtype),
                "logits_std": logits.detach().float().std(unbiased=False).to(dtype=h_pixel.dtype),
                "logits_min": logits.detach().float().min().to(dtype=h_pixel.dtype),
                "logits_max": logits.detach().float().max().to(dtype=h_pixel.dtype),
                "assignment_entropy_mean": assignment_entropy.detach().float().mean().to(dtype=h_pixel.dtype),
                "spatial_bias_strength_current": self.spatial_bias_strength_current.to(device=h_pixel.device, dtype=h_pixel.dtype),
                "motif_initial_center_min": self.query_init_centers.min().to(device=h_pixel.device, dtype=h_pixel.dtype),
                "motif_initial_center_max": self.query_init_centers.max().to(device=h_pixel.device, dtype=h_pixel.dtype),
            }
        )
        return {
            "motif_embeddings": motif_embeddings,
            "motif_assignment_maps": motif_assignment_maps,
            "motif_scores": motif_scores,
            "motif_centers": motif_centers,
            "motif_area": motif_area,
            "motif_border_mass": motif_border_mass,
            "motif_center_mass": motif_center_mass,
            "motif_effective_count": audit["effective_motif_count"],
            "motif_audit": audit,
        }


class MotifDiscoveryDebugModel(nn.Module):
    """This module is for motif discovery/audit pretraining; not used as main classifier yet."""

    def __init__(
        self,
        num_nodes: int = 2304,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        pixel_gnn_layers: int = 1,
        num_motifs: int = 16,
        image_size: int = 48,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dropout: float = 0.1,
        **motif_kwargs: Any,
    ) -> None:
        super().__init__()
        self.freeze_pixel_encoder = bool(motif_kwargs.pop("freeze_pixel_encoder", False))
        self.use_aux_motif_classifier = bool(motif_kwargs.pop("use_aux_motif_classifier", False))
        self.aux_pooling = str(motif_kwargs.pop("aux_pooling", "mean")).lower()
        self.selection_temperature = float(motif_kwargs.pop("selection_temperature", 1.0))
        self.selection_top_m = int(motif_kwargs.pop("selection_top_m", num_motifs))
        self.use_soft_top_m_mask = bool(motif_kwargs.pop("use_soft_top_m_mask", True))
        self.aux_dropout = float(motif_kwargs.pop("aux_dropout", 0.2))
        self.num_classes = int(motif_kwargs.pop("num_classes", 7))
        if self.aux_pooling not in {"mean", "score_weighted", "selected_score_weighted", "selected_topm_mean"}:
            raise ValueError(f"Unsupported aux_pooling={self.aux_pooling!r}")
        if self.selection_temperature <= 0.0:
            raise ValueError(f"selection_temperature must be > 0, got {self.selection_temperature}")
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.height = int(height or image_size)
        self.width = int(width or image_size)
        if self.num_nodes != self.height * self.width:
            raise ValueError(f"num_nodes={self.num_nodes} must match height*width={self.height * self.width}")
        motif_kwargs.pop("use_pixel_encoder_from", None)
        self.encoder = SharedPixelEncoder(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            pixel_gnn_layers=int(pixel_gnn_layers),
            dropout=float(dropout),
        )
        self.motif_discovery = MotifDiscoveryModule(
            hidden_dim=self.hidden_dim,
            num_motifs=int(num_motifs),
            image_hw=(self.height, self.width),
            dropout=float(dropout),
            **motif_kwargs,
        )
        if self.use_aux_motif_classifier:
            self.aux_classifier = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(self.aux_dropout),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            self.aux_classifier = None
        if self.freeze_pixel_encoder:
            self.set_pixel_encoder_trainable(False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MotifDiscoveryDebugModel":
        cfg = dict(config)
        return cls(**cfg)

    def set_pixel_encoder_trainable(self, trainable: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = bool(trainable)

    def set_spatial_bias_strength(self, strength: float) -> None:
        if hasattr(self.motif_discovery, "set_spatial_bias_strength"):
            self.motif_discovery.set_spatial_bias_strength(strength)

    def _selected_score_weights(self, motif_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = motif_scores / float(self.selection_temperature)
        num_motifs = logits.shape[1]
        top_m = max(1, min(int(self.selection_top_m), int(num_motifs)))
        top_indices = torch.topk(logits, k=top_m, dim=1).indices
        selected_mask = logits.new_zeros(logits.shape)
        selected_mask.scatter_(dim=1, index=top_indices, value=1.0)
        if self.use_soft_top_m_mask and top_m < num_motifs:
            masked_logits = logits.new_full(logits.shape, -1e4)
            masked_logits = masked_logits.masked_fill(selected_mask.bool(), 0.0)
            masked_logits = torch.where(selected_mask.bool(), logits, masked_logits)
            weights = torch.softmax(masked_logits, dim=1)
        else:
            weights = torch.softmax(logits, dim=1)
        return weights, top_indices, selected_mask

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
            raise KeyError("MotifDiscoveryDebugModel needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("MotifDiscoveryDebugModel requires edge_index and edge_attr")
        if self.freeze_pixel_encoder:
            self.encoder.eval()
            with torch.no_grad():
                h_pixel = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        else:
            h_pixel = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        out = self.motif_discovery(h_pixel, image_hw=(self.height, self.width), node_mask=node_mask)
        if self.aux_pooling in {"selected_score_weighted", "selected_topm_mean"}:
            selection_weights, selected_indices, selected_mask = self._selected_score_weights(out["motif_scores"])
            out["selection_weights"] = selection_weights
            out["selected_indices"] = selected_indices
            out["selected_mask"] = selected_mask
            out["selection_entropy"] = -(
                selection_weights * selection_weights.clamp_min(1e-8).log()
            ).sum(dim=1).detach()
            out["selection_effective_count"] = out["selection_entropy"].exp()
            out["selected_top_m"] = out["motif_scores"].new_tensor(float(selected_indices.shape[1]))
            if self.aux_pooling == "selected_topm_mean":
                out["selected_repr"] = (
                    selected_mask.unsqueeze(-1) * out["motif_embeddings"]
                ).sum(dim=1) / float(selected_indices.shape[1])
            else:
                out["selected_repr"] = (selection_weights.unsqueeze(-1) * out["motif_embeddings"]).sum(dim=1)
            if "motif_border_mass" in out:
                out["selected_border_mass"] = (selection_weights * out["motif_border_mass"]).sum(dim=1).detach()
        if self.aux_classifier is not None:
            motif_embeddings = out["motif_embeddings"]
            if self.aux_pooling in {"selected_score_weighted", "selected_topm_mean"}:
                motif_repr = out["selected_repr"]
            elif self.aux_pooling == "score_weighted":
                weights = torch.softmax(out["motif_scores"], dim=1).unsqueeze(-1)
                motif_repr = (weights * motif_embeddings).sum(dim=1)
            else:
                motif_repr = motif_embeddings.mean(dim=1)
            aux_logits = self.aux_classifier(motif_repr)
            out["aux_logits"] = aux_logits
            out["aux_probs"] = torch.softmax(aux_logits, dim=1)
        out["h_pixel"] = h_pixel
        out["pixel_embeddings"] = h_pixel
        return out
