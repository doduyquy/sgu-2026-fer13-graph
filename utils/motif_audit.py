"""Audit helpers for motif discovery redundancy and spatial quality."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _pair_mask(num_motifs: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(num_motifs, num_motifs, dtype=torch.bool, device=device), diagonal=1)


def compute_motif_centers(assignment_maps: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Compute x/y centers from assignment maps [B, K, H, W]."""

    if assignment_maps.ndim != 4:
        raise ValueError(f"assignment_maps must be [B, K, H, W], got {tuple(assignment_maps.shape)}")
    _, _, height, width = assignment_maps.shape
    dtype = assignment_maps.dtype
    device = assignment_maps.device
    if normalized:
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    else:
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mass = assignment_maps.sum(dim=(2, 3)).clamp_min(1e-8)
    cx = (assignment_maps * xx.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    cy = (assignment_maps * yy.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    return torch.stack([cx, cy], dim=-1)


def compute_motif_area(assignment_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Effective motif area in pixels, exp(entropy(A_k))."""

    if assignment_maps.ndim != 4:
        raise ValueError(f"assignment_maps must be [B, K, H, W], got {tuple(assignment_maps.shape)}")
    flat = assignment_maps.flatten(2)
    probs = flat / flat.sum(dim=2, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=2)
    return entropy.exp()


def compute_border_center_mass(
    assignment_maps: torch.Tensor,
    border_width: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return border and non-border mass for maps normalized over H*W."""

    if assignment_maps.ndim != 4:
        raise ValueError(f"assignment_maps must be [B, K, H, W], got {tuple(assignment_maps.shape)}")
    _, _, height, width = assignment_maps.shape
    mask = torch.zeros(height, width, device=assignment_maps.device, dtype=assignment_maps.dtype)
    bw = max(0, int(border_width))
    if bw > 0:
        mask[:bw, :] = 1.0
        mask[-bw:, :] = 1.0
        mask[:, :bw] = 1.0
        mask[:, -bw:] = 1.0
    border_mass = (assignment_maps * mask.view(1, 1, height, width)).sum(dim=(2, 3))
    total_mass = assignment_maps.sum(dim=(2, 3)).clamp_min(1e-8)
    center_mass = total_mass - border_mass
    return border_mass, center_mass


def compute_outer_border_mass(
    assignment_maps: torch.Tensor,
    outer_border_width: int = 6,
) -> torch.Tensor:
    """Return assignment mass inside the explicit outer crop-edge band."""

    border_mass, _ = compute_border_center_mass(assignment_maps, border_width=outer_border_width)
    return border_mass


def compute_clean_candidate_scores(
    border_mass: torch.Tensor,
    outer_border_mass: torch.Tensor,
    foreground_mass: torch.Tensor,
    effective_area: torch.Tensor,
    clean_border_threshold: float = 0.30,
    clean_outer_threshold: float = 0.40,
    clean_foreground_threshold: float = 0.25,
    clean_tau: float = 0.05,
    clean_area_tau: float = 50.0,
    clean_min_effective_area: float = 40.0,
    clean_max_effective_area: float = 512.0,
) -> Dict[str, torch.Tensor]:
    """Soft per-motif cleanliness score and hard clean candidate count."""

    tau = max(float(clean_tau), 1e-8)
    area_tau = max(float(clean_area_tau), 1e-8)
    clean_border_score = torch.sigmoid((float(clean_border_threshold) - border_mass) / tau)
    clean_outer_score = torch.sigmoid((float(clean_outer_threshold) - outer_border_mass) / tau)
    clean_foreground_score = torch.sigmoid((foreground_mass - float(clean_foreground_threshold)) / tau)
    clean_area_low_score = torch.sigmoid((effective_area - float(clean_min_effective_area)) / area_tau)
    clean_area_high_score = torch.sigmoid((float(clean_max_effective_area) - effective_area) / area_tau)
    clean_score = (
        clean_border_score
        * clean_outer_score
        * clean_foreground_score
        * clean_area_low_score
        * clean_area_high_score
    )
    hard_clean = (
        (border_mass < float(clean_border_threshold))
        & (outer_border_mass < float(clean_outer_threshold))
        & (foreground_mass > float(clean_foreground_threshold))
        & (effective_area >= float(clean_min_effective_area))
        & (effective_area <= float(clean_max_effective_area))
    )
    return {
        "clean_score": clean_score,
        "clean_border_score": clean_border_score,
        "clean_outer_score": clean_outer_score,
        "clean_foreground_score": clean_foreground_score,
        "clean_area_low_score": clean_area_low_score,
        "clean_area_high_score": clean_area_high_score,
        "clean_candidate_count_per_sample": clean_score.sum(dim=1),
        "clean_candidate_count": clean_score.sum(dim=1).mean(),
        "hard_clean_candidate_count_per_sample": hard_clean.float().sum(dim=1),
        "hard_clean_candidate_count": hard_clean.float().sum(dim=1).mean(),
    }


def compute_soft_region_masses(
    assignment_maps: torch.Tensor,
    region_tau: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Soft upper/middle/lower y-band masses for normalized assignment maps."""

    if assignment_maps.ndim != 4:
        raise ValueError(f"assignment_maps must be [B, K, H, W], got {tuple(assignment_maps.shape)}")
    _, _, height, width = assignment_maps.shape
    tau = max(float(region_tau), 1e-8)
    y = torch.linspace(0.0, 1.0, height, device=assignment_maps.device, dtype=assignment_maps.dtype).view(height, 1)
    upper_mask = torch.sigmoid((0.42 - y) / tau).expand(height, width)
    middle_mask = (torch.sigmoid((y - 0.30) / tau) * torch.sigmoid((0.70 - y) / tau)).expand(height, width)
    lower_mask = torch.sigmoid((y - 0.55) / tau).expand(height, width)
    upper_mass = (assignment_maps * upper_mask.view(1, 1, height, width)).sum(dim=(2, 3))
    middle_mass = (assignment_maps * middle_mask.view(1, 1, height, width)).sum(dim=(2, 3))
    lower_mass = (assignment_maps * lower_mask.view(1, 1, height, width)).sum(dim=(2, 3))
    return {
        "upper_mask": upper_mask,
        "middle_mask": middle_mask,
        "lower_mask": lower_mask,
        "upper_mass": upper_mass,
        "middle_mass": middle_mass,
        "lower_mass": lower_mass,
    }


def pairwise_assignment_similarity(
    assignment_maps: torch.Tensor,
    redundant_threshold: float = 0.80,
) -> Dict[str, torch.Tensor]:
    """Cosine similarity between motif assignment maps within each sample."""

    if assignment_maps.ndim != 4:
        raise ValueError(f"assignment_maps must be [B, K, H, W], got {tuple(assignment_maps.shape)}")
    bsz, num_motifs, _, _ = assignment_maps.shape
    flat = F.normalize(assignment_maps.flatten(2).float(), dim=2, eps=1e-8)
    sim = torch.bmm(flat, flat.transpose(1, 2))
    mask = _pair_mask(num_motifs, sim.device).unsqueeze(0).expand(bsz, -1, -1)
    values = sim[mask]
    if values.numel() == 0:
        values = sim.new_zeros(1)
    redundant = values > float(redundant_threshold)
    return {
        "pairwise_map_sim": sim,
        "mean_pairwise_map_sim": values.mean(),
        "max_pairwise_map_sim": values.max(),
        "redundant_map_pair_count": redundant.sum().to(dtype=torch.float32),
    }


def pairwise_embedding_similarity(
    motif_embeddings: torch.Tensor,
    redundant_threshold: float = 0.80,
) -> Dict[str, torch.Tensor]:
    """Cosine similarity between motif embeddings within each sample."""

    if motif_embeddings.ndim != 3:
        raise ValueError(f"motif_embeddings must be [B, K, D], got {tuple(motif_embeddings.shape)}")
    bsz, num_motifs, _ = motif_embeddings.shape
    emb = F.normalize(motif_embeddings.float(), dim=2, eps=1e-8)
    sim = torch.bmm(emb, emb.transpose(1, 2))
    mask = _pair_mask(num_motifs, sim.device).unsqueeze(0).expand(bsz, -1, -1)
    values = sim[mask]
    if values.numel() == 0:
        values = sim.new_zeros(1)
    redundant = values > float(redundant_threshold)
    return {
        "pairwise_emb_sim": sim,
        "mean_pairwise_emb_sim": values.mean(),
        "max_pairwise_emb_sim": values.max(),
        "redundant_emb_pair_count": redundant.sum().to(dtype=torch.float32),
    }


def pairwise_center_distance(
    motif_centers: torch.Tensor,
    close_threshold: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Pairwise motif center distances for centers [B, K, 2]."""

    if motif_centers.ndim != 3 or motif_centers.shape[-1] != 2:
        raise ValueError(f"motif_centers must be [B, K, 2], got {tuple(motif_centers.shape)}")
    bsz, num_motifs, _ = motif_centers.shape
    dist = torch.cdist(motif_centers.float(), motif_centers.float(), p=2)
    mask = _pair_mask(num_motifs, dist.device).unsqueeze(0).expand(bsz, -1, -1)
    values = dist[mask]
    if values.numel() == 0:
        values = dist.new_zeros(1)
    close = values < float(close_threshold)
    return {
        "pairwise_center_dist": dist,
        "mean_center_dist": values.mean(),
        "min_center_dist": values.min(),
        "close_center_pair_count": close.sum().to(dtype=torch.float32),
    }


def compute_effective_motif_count(
    assignment_maps: torch.Tensor,
    motif_embeddings: torch.Tensor,
    motif_scores: torch.Tensor | None = None,
    motif_centers: torch.Tensor | None = None,
    border_width: int = 4,
    min_effective_area: float = 40.0,
    max_effective_area: float = 400.0,
    max_border_mass: float = 0.45,
    map_sim_threshold: float = 0.80,
    emb_sim_threshold: float = 0.80,
    center_distance_threshold: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Greedy active motif count using area, border, and redundancy checks."""

    if motif_centers is None:
        motif_centers = compute_motif_centers(assignment_maps, normalized=True)
    bsz, num_motifs, _, _ = assignment_maps.shape
    area = compute_motif_area(assignment_maps)
    border_mass, _ = compute_border_center_mass(assignment_maps, border_width=border_width)
    map_sim = pairwise_assignment_similarity(assignment_maps, redundant_threshold=map_sim_threshold)["pairwise_map_sim"]
    emb_sim = pairwise_embedding_similarity(motif_embeddings, redundant_threshold=emb_sim_threshold)["pairwise_emb_sim"]
    center_dist = pairwise_center_distance(motif_centers, close_threshold=center_distance_threshold)["pairwise_center_dist"]

    area_ok = (area >= float(min_effective_area)) & (area <= float(max_effective_area))
    border_ok = border_mass < float(max_border_mass)
    candidate_ok = area_ok & border_ok
    if motif_scores is None:
        area_mid = 0.5 * (float(min_effective_area) + float(max_effective_area))
        order_score = -torch.abs(area.float() - area_mid)
    else:
        order_score = motif_scores.detach().float()

    counts = []
    active_masks = []
    for bidx in range(bsz):
        active: list[int] = []
        active_mask = torch.zeros(num_motifs, device=assignment_maps.device, dtype=torch.float32)
        order = torch.argsort(order_score[bidx], descending=True)
        for item in order.tolist():
            if not bool(candidate_ok[bidx, item]):
                continue
            is_redundant = False
            for chosen in active:
                same_map = bool(map_sim[bidx, item, chosen] > float(map_sim_threshold))
                same_emb = bool(emb_sim[bidx, item, chosen] > float(emb_sim_threshold))
                close_center = bool(center_dist[bidx, item, chosen] < float(center_distance_threshold))
                if same_map and same_emb and close_center:
                    is_redundant = True
                    break
            if not is_redundant:
                active.append(int(item))
                active_mask[item] = 1.0
        counts.append(float(len(active)))
        active_masks.append(active_mask)

    count = torch.tensor(counts, device=assignment_maps.device, dtype=torch.float32)
    active_mask_tensor = torch.stack(active_masks, dim=0)
    return {
        "effective_motif_count_per_sample": count,
        "effective_motif_count": count.mean(),
        "effective_motif_ratio": count.mean() / float(max(1, num_motifs)),
        "active_motif_mask": active_mask_tensor,
    }


def audit_motif_outputs(
    assignment_maps: torch.Tensor,
    motif_embeddings: torch.Tensor,
    motif_scores: torch.Tensor | None = None,
    motif_centers: torch.Tensor | None = None,
    border_width: int = 4,
    map_sim_threshold: float = 0.80,
    emb_sim_threshold: float = 0.80,
    center_distance_threshold: float = 0.15,
    min_effective_area: float = 40.0,
    max_effective_area: float = 400.0,
    max_border_mass: float = 0.45,
    outer_border_width: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Summarize motif quality and collapse/redundancy indicators."""

    if motif_centers is None:
        motif_centers = compute_motif_centers(assignment_maps, normalized=True)
    map_stats = pairwise_assignment_similarity(assignment_maps, redundant_threshold=map_sim_threshold)
    emb_stats = pairwise_embedding_similarity(motif_embeddings, redundant_threshold=emb_sim_threshold)
    center_stats = pairwise_center_distance(motif_centers, close_threshold=center_distance_threshold)
    border_mass, center_mass = compute_border_center_mass(assignment_maps, border_width=border_width)
    outer_border_mass = compute_outer_border_mass(
        assignment_maps,
        outer_border_width=border_width if outer_border_width is None else int(outer_border_width),
    )
    area = compute_motif_area(assignment_maps)

    map_sim = map_stats["pairwise_map_sim"]
    emb_sim = emb_stats["pairwise_emb_sim"]
    center_dist = center_stats["pairwise_center_dist"]
    bsz, num_motifs, _ = map_sim.shape
    pair_mask = _pair_mask(num_motifs, map_sim.device).unsqueeze(0).expand(bsz, -1, -1)
    redundant_mask = (
        (map_sim > float(map_sim_threshold))
        & (emb_sim > float(emb_sim_threshold))
        & (center_dist < float(center_distance_threshold))
        & pair_mask
    )
    redundant_pair_count = redundant_mask.sum().to(dtype=torch.float32)
    total_pairs = max(1, bsz * num_motifs * (num_motifs - 1) // 2)
    redundant_pair_ratio = redundant_pair_count / float(total_pairs)
    effective = compute_effective_motif_count(
        assignment_maps,
        motif_embeddings,
        motif_scores=motif_scores,
        motif_centers=motif_centers,
        border_width=border_width,
        min_effective_area=min_effective_area,
        max_effective_area=max_effective_area,
        max_border_mass=max_border_mass,
        map_sim_threshold=map_sim_threshold,
        emb_sim_threshold=emb_sim_threshold,
        center_distance_threshold=center_distance_threshold,
    )

    out: Dict[str, torch.Tensor] = {}
    out.update(map_stats)
    out.update(emb_stats)
    out.update(center_stats)
    out.update(
        {
            "motif_area_mean": area.mean(),
            "motif_area_min": area.min(),
            "motif_area_max": area.max(),
            "effective_area_mean": area.mean(),
            "effective_area_min": area.min(),
            "effective_area_max": area.max(),
            "area_in_range_ratio": ((area >= float(min_effective_area)) & (area <= float(max_effective_area))).float().mean(),
            "area_over_max_ratio": (area > float(max_effective_area)).float().mean(),
            "area_under_min_ratio": (area < float(min_effective_area)).float().mean(),
            "effective_area_over_max_ratio": (area > float(max_effective_area)).float().mean(),
            "effective_area_under_min_ratio": (area < float(min_effective_area)).float().mean(),
            "border_mass_mean": border_mass.mean(),
            "center_mass_mean": center_mass.mean(),
            "center_minus_border": center_mass.mean() - border_mass.mean(),
            "border_dominance": torch.relu(border_mass.mean() - center_mass.mean()),
            "outer_border_mass_mean": outer_border_mass.mean(),
            "redundant_pair_count": redundant_pair_count,
            "redundant_pair_ratio": redundant_pair_ratio,
        }
    )
    out.update(effective)
    return out
