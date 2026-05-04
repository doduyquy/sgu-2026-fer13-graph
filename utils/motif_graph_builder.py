"""Build small motif graphs from frozen Stage 1 motif discovery outputs."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from utils.motif_audit import compute_soft_region_masses


def _normalize_spatial_sum(maps: torch.Tensor) -> torch.Tensor:
    return maps / maps.flatten(2).sum(dim=2).clamp_min(1e-8).view(maps.shape[0], maps.shape[1], 1, 1)


def compute_motif_centers(assignment_maps: torch.Tensor) -> torch.Tensor:
    maps = _normalize_spatial_sum(assignment_maps)
    bsz, num_motifs, height, width = maps.shape
    ys = torch.linspace(0.0, 1.0, height, device=maps.device, dtype=maps.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=maps.device, dtype=maps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    flat = maps.flatten(2)
    cx = (flat * xx.reshape(1, 1, -1)).sum(dim=2)
    cy = (flat * yy.reshape(1, 1, -1)).sum(dim=2)
    return torch.stack([cx, cy], dim=-1)


def compute_effective_area(assignment_maps: torch.Tensor) -> torch.Tensor:
    flat = _normalize_spatial_sum(assignment_maps).flatten(2)
    entropy = -(flat * flat.clamp_min(1e-8).log()).sum(dim=2)
    return entropy.exp()


def _gather_by_index(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expand_shape = list(indices.shape) + list(values.shape[2:])
    gather_index = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (values.ndim - 2))).expand(expand_shape)
    return values.gather(dim=1, index=gather_index)


def _selection_weights(outputs: Dict[str, torch.Tensor], source: str) -> torch.Tensor:
    if source == "selection_weights" and "selection_weights" in outputs:
        return outputs["selection_weights"]
    if "motif_scores" not in outputs:
        raise KeyError("Stage 1 outputs need selection_weights or motif_scores for top-M selection")
    return torch.softmax(outputs["motif_scores"], dim=1)


def _maybe_detach(value: torch.Tensor, detach: bool) -> torch.Tensor:
    return value.detach() if detach else value


def build_motif_graph(stage1_outputs: Dict[str, torch.Tensor], stage2_cfg: Dict[str, Any], detach: bool = True) -> Dict[str, torch.Tensor]:
    embeddings = _maybe_detach(stage1_outputs["motif_embeddings"], detach)
    maps = _maybe_detach(stage1_outputs["motif_assignment_maps"], detach)
    source = str(stage2_cfg.get("selection_source", "selection_weights"))
    weights = _maybe_detach(_selection_weights(stage1_outputs, source), detach)
    top_m = max(1, min(int(stage2_cfg.get("num_selected_motifs", 6)), int(weights.shape[1])))
    top_indices = torch.topk(weights, k=top_m, dim=1).indices
    selected_embeddings = _gather_by_index(embeddings, top_indices)
    selected_weights = _gather_by_index(weights.unsqueeze(-1), top_indices).squeeze(-1)
    selected_weights_norm = selected_weights / selected_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    centers = stage1_outputs.get("motif_centers")
    if centers is None:
        centers = stage1_outputs.get("centers")
    if centers is None:
        centers = compute_motif_centers(maps)
    centers = _maybe_detach(centers, detach).to(device=embeddings.device, dtype=embeddings.dtype)
    selected_centers = _gather_by_index(centers, top_indices).clamp(0.0, 1.0)

    area = stage1_outputs.get("motif_area")
    if area is None:
        area = stage1_outputs.get("effective_area")
    if area is None:
        area = compute_effective_area(maps)
    area = _maybe_detach(area, detach).to(device=embeddings.device, dtype=embeddings.dtype)
    selected_area = _gather_by_index(area.unsqueeze(-1), top_indices).squeeze(-1)

    features = [selected_embeddings]
    if bool(stage2_cfg.get("use_selection_weight_feature", True)):
        features.append(selected_weights_norm.unsqueeze(-1))
    if bool(stage2_cfg.get("use_spatial_features", True)):
        features.append(selected_centers)
    if bool(stage2_cfg.get("use_area_feature", True)):
        denom = float(stage2_cfg.get("area_norm", maps.shape[-1] * maps.shape[-2]))
        features.append((selected_area / max(denom, 1.0)).clamp_min(0.0).unsqueeze(-1))
    if bool(stage2_cfg.get("use_region_features", False)):
        if "motif_assignment_maps" not in stage1_outputs:
            region_missing = str(stage2_cfg.get("region_feature_missing", "error")).lower()
            if region_missing not in {"zeros", "zero"}:
                raise KeyError(
                    "stage2.use_region_features=true requires Stage 1 output 'motif_assignment_maps' "
                    "to compute upper/middle/lower soft region masses"
                )
            features.append(selected_embeddings.new_zeros((*selected_embeddings.shape[:2], 3)))
        else:
            # Match Stage 1G's soft y-band region masks instead of hard thirds.
            selected_maps = _gather_by_index(maps, top_indices)
            region = compute_soft_region_masses(
                selected_maps,
                region_tau=float(stage2_cfg.get("region_tau", 0.05)),
            )
            region_features = torch.stack(
                [region["upper_mass"], region["middle_mass"], region["lower_mass"]],
                dim=-1,
            ).to(device=embeddings.device, dtype=embeddings.dtype)
            features.append(region_features)
    node_features = torch.cat(features, dim=-1)

    ci = selected_centers.unsqueeze(2)
    cj = selected_centers.unsqueeze(1)
    delta = cj - ci
    dx = delta[..., 0]
    dy = delta[..., 1]
    dist = torch.sqrt(dx.square() + dy.square()).clamp_min(0.0)
    edge_features = torch.stack([dx, dy, dist, dx.abs(), dy.abs()], dim=-1)
    if not bool(stage2_cfg.get("self_loops", False)):
        eye = torch.eye(top_m, device=edge_features.device, dtype=torch.bool).view(1, top_m, top_m, 1)
        edge_features = edge_features.masked_fill(eye, 0.0)
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "selected_embeddings": selected_embeddings,
        "selected_weights": selected_weights_norm,
        "selected_centers": selected_centers,
        "selected_area": selected_area,
        "selected_indices": top_indices,
    }
