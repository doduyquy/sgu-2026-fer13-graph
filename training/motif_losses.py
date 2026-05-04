"""Light losses for motif discovery debug/pretraining."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from utils.motif_audit import compute_clean_candidate_scores, compute_soft_region_masses


class MotifDiscoveryStage1Loss(nn.Module):
    """This module is for motif discovery/audit pretraining; not used as main classifier yet."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.lambda_div_map = float(cfg.get("lambda_div_map", 0.02))
        self.lambda_div_emb = float(cfg.get("lambda_div_emb", 0.01))
        self.lambda_coverage = float(cfg.get("lambda_coverage", 0.0))
        self.lambda_anchor = float(cfg.get("lambda_anchor", 0.0))
        self.lambda_border = float(cfg.get("lambda_border", 0.02))
        self.lambda_soft_border = float(cfg.get("lambda_soft_border", 0.0))
        self.lambda_outer_border = float(cfg.get("lambda_outer_border", 0.0))
        self.lambda_selected_border = float(cfg.get("lambda_selected_border", 0.0))
        self.lambda_selected_outer_border = float(cfg.get("lambda_selected_outer_border", 0.0))
        self.lambda_selected_foreground = float(cfg.get("lambda_selected_foreground", 0.0))
        self.lambda_selected_diversity = float(cfg.get("lambda_selected_diversity", 0.0))
        self.lambda_selection_entropy = float(cfg.get("lambda_selection_entropy", 0.0))
        self.target_selection_entropy_min = float(cfg.get("target_selection_entropy_min", 1.2))
        self.use_clean_candidate_loss = bool(cfg.get("use_clean_candidate_loss", False))
        self.lambda_clean_count = float(cfg.get("lambda_clean_count", 0.0))
        self.lambda_clean_mean = float(cfg.get("lambda_clean_mean", 0.0))
        self.use_region_clean_loss = bool(cfg.get("use_region_clean_loss", False))
        self.lambda_region_clean = float(cfg.get("lambda_region_clean", 0.0))
        self.region_clean_target = float(cfg.get("region_clean_target", 0.7))
        self.region_clean_upper_weight = float(cfg.get("region_clean_upper_weight", 1.0))
        self.region_clean_middle_weight = float(cfg.get("region_clean_middle_weight", 1.0))
        self.region_clean_lower_weight = float(cfg.get("region_clean_lower_weight", 1.0))
        self.region_tau = float(cfg.get("region_tau", 0.05))
        self.clean_border_threshold = float(cfg.get("clean_border_threshold", 0.30))
        self.clean_outer_threshold = float(cfg.get("clean_outer_threshold", 0.40))
        self.clean_foreground_threshold = float(cfg.get("clean_foreground_threshold", 0.25))
        self.clean_tau = float(cfg.get("clean_tau", 0.05))
        self.clean_area_tau = float(cfg.get("clean_area_tau", 50.0))
        self.clean_count_target = float(cfg.get("clean_count_target", 3.0))
        self.lambda_area = float(cfg.get("lambda_area", 0.02))
        self.lambda_entropy = float(cfg.get("lambda_entropy", 0.01))
        self.area_min = float(cfg.get("min_effective_area", cfg.get("area_min", 40.0)))
        self.area_max = float(cfg.get("max_effective_area", cfg.get("area_max", 400.0)))
        self.clean_min_effective_area = float(cfg.get("clean_min_effective_area", self.area_min))
        self.clean_max_effective_area = float(cfg.get("clean_max_effective_area", cfg.get("audit_max_effective_area", 512.0)))
        self.height = int(cfg.get("height", cfg.get("image_size", 48)))
        self.width = int(cfg.get("width", cfg.get("image_size", 48)))
        self.entropy_min_ratio = float(cfg.get("entropy_min_ratio", 0.18))
        self.entropy_max_ratio = float(cfg.get("entropy_max_ratio", 0.88))
        self.use_soft_border_penalty = bool(cfg.get("use_soft_border_penalty", False))
        self.soft_border_tau = float(cfg.get("soft_border_tau", 0.12))
        if self.soft_border_tau <= 0.0:
            raise ValueError(f"soft_border_tau must be > 0, got {self.soft_border_tau}")
        self.use_outer_border_penalty = bool(cfg.get("use_outer_border_penalty", False))
        self.outer_border_width = int(cfg.get("outer_border_width", 6))
        if self.outer_border_width < 0:
            raise ValueError(f"outer_border_width must be >= 0, got {self.outer_border_width}")
        self.use_center_weighted_foreground = bool(cfg.get("use_center_weighted_foreground", False))
        self.use_face_safe_foreground = bool(cfg.get("use_face_safe_foreground", False))
        self.use_foreground_anchor = bool(cfg.get("use_foreground_anchor", False))
        self.face_safe_margin = float(cfg.get("face_safe_margin", 0.12))
        self.face_safe_tau = float(cfg.get("face_safe_tau", 0.04))
        self.center_prior_sigma = float(cfg.get("center_prior_sigma", 0.35))
        self.center_prior_power = float(cfg.get("center_prior_power", 1.0))
        self.coverage_loss_type = str(cfg.get("coverage_loss_type", "mse")).lower()
        if self.coverage_loss_type not in {"mse", "cosine", "mse_plus_cosine"}:
            raise ValueError(f"Unsupported coverage_loss_type: {self.coverage_loss_type}")
        if self.center_prior_sigma <= 0.0:
            raise ValueError(f"center_prior_sigma must be > 0, got {self.center_prior_sigma}")
        if self.face_safe_tau <= 0.0:
            raise ValueError(f"face_safe_tau must be > 0, got {self.face_safe_tau}")
        if self.clean_tau <= 0.0:
            raise ValueError(f"clean_tau must be > 0, got {self.clean_tau}")
        if self.clean_area_tau <= 0.0:
            raise ValueError(f"clean_area_tau must be > 0, got {self.clean_area_tau}")
        if self.clean_count_target <= 0.0:
            raise ValueError(f"clean_count_target must be > 0, got {self.clean_count_target}")
        if self.region_clean_target <= 0.0:
            raise ValueError(f"region_clean_target must be > 0, got {self.region_clean_target}")
        if self.region_clean_upper_weight < 0.0:
            raise ValueError(f"region_clean_upper_weight must be >= 0, got {self.region_clean_upper_weight}")
        if self.region_clean_middle_weight < 0.0:
            raise ValueError(f"region_clean_middle_weight must be >= 0, got {self.region_clean_middle_weight}")
        if self.region_clean_lower_weight < 0.0:
            raise ValueError(f"region_clean_lower_weight must be >= 0, got {self.region_clean_lower_weight}")
        region_clean_weight_sum = (
            self.region_clean_upper_weight + self.region_clean_middle_weight + self.region_clean_lower_weight
        )
        if region_clean_weight_sum <= 0.0:
            raise ValueError("region clean weights must sum to > 0")
        if self.region_tau <= 0.0:
            raise ValueError(f"region_tau must be > 0, got {self.region_tau}")
        self.eps = float(cfg.get("eps", 1e-8))

    def set_loss_weights(self, weights: Dict[str, float]) -> None:
        for name, value in (weights or {}).items():
            if hasattr(self, name):
                setattr(self, name, float(value))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        maps = model_out["motif_assignment_maps"]
        emb = model_out["motif_embeddings"]
        area = model_out.get("motif_area")
        border_mass = model_out.get("motif_border_mass")
        center_mass = model_out.get("motif_center_mass")
        if area is None:
            area = self._effective_area(maps)
        if border_mass is None:
            border_mass = maps.new_zeros(maps.shape[:2])
        if center_mass is None:
            center_mass = maps.sum(dim=(2, 3)) - border_mass

        loss_div_map, mean_map_sim, max_map_sim = self._pairwise_cosine_loss(maps.flatten(2))
        loss_div_emb, mean_emb_sim, max_emb_sim = self._pairwise_cosine_loss(emb)
        (
            loss_coverage,
            coverage_cosine,
            foreground_mass_mean,
            foreground_center_mass,
            foreground_border_mass,
            foreground_prior_sum,
            foreground_safe_mass,
            foreground,
        ) = self._coverage_loss(maps, batch)
        loss_anchor, motif_fg_mean, motif_fg_min, motif_fg_max = self._foreground_anchor_loss(maps, foreground)
        loss_border = border_mass.mean()
        loss_soft_border, soft_border_mass_mean = self._soft_border_loss(maps)
        loss_outer_border, outer_border_mass_mean = self._outer_border_loss(maps)
        (
            loss_selected_border,
            loss_selected_outer_border,
            loss_selected_foreground,
            loss_selected_diversity,
            selected_border_mass_mean,
            selected_outer_border_mass_mean,
            selected_foreground_mass_mean,
            selection_entropy,
            selection_effective_count,
            loss_selection_entropy,
            selected_pairwise_map_sim,
            selected_pairwise_emb_sim,
            motif_foreground_mass,
        ) = self._selected_utility_losses(model_out, maps, emb, border_mass, foreground)
        clean_metrics = self._clean_candidate_losses(
            maps=maps,
            border_mass=border_mass,
            motif_foreground_mass=motif_foreground_mass,
            area=area,
        )
        region_metrics = self._region_clean_losses(maps=maps, clean_score=clean_metrics["clean_score"])
        loss_entropy, loss_area_high, loss_area_low = self._entropy_control_loss(area)
        loss_area = self._area_balance_loss(area)
        over_max = (area > float(self.area_max)).float().mean()
        under_min = (area < float(self.area_min)).float().mean()
        total = (
            self.lambda_div_map * loss_div_map
            + self.lambda_div_emb * loss_div_emb
            + self.lambda_coverage * loss_coverage
            + self.lambda_anchor * loss_anchor
            + self.lambda_border * loss_border
            + self.lambda_soft_border * loss_soft_border
            + self.lambda_outer_border * loss_outer_border
            + self.lambda_selected_border * loss_selected_border
            + self.lambda_selected_outer_border * loss_selected_outer_border
            + self.lambda_selected_foreground * loss_selected_foreground
            + self.lambda_selected_diversity * loss_selected_diversity
            + self.lambda_selection_entropy * loss_selection_entropy
            + self.lambda_clean_count * clean_metrics["loss_clean_count"]
            + self.lambda_clean_mean * clean_metrics["loss_clean_mean"]
            + self.lambda_region_clean * region_metrics["loss_region_clean"]
            + self.lambda_area * loss_area
            + self.lambda_entropy * loss_entropy
        )
        return {
            "loss": total,
            "total_loss": total,
            "loss_map_diversity": loss_div_map,
            "loss_embedding_diversity": loss_div_emb,
            "loss_coverage": loss_coverage,
            "loss_anchor": loss_anchor,
            "loss_border": loss_border,
            "loss_soft_border": loss_soft_border,
            "loss_outer_border": loss_outer_border,
            "loss_selected_border": loss_selected_border,
            "loss_selected_outer_border": loss_selected_outer_border,
            "loss_selected_foreground": loss_selected_foreground,
            "loss_selected_diversity": loss_selected_diversity,
            "loss_selection_entropy": loss_selection_entropy,
            "loss_clean_count": clean_metrics["loss_clean_count"],
            "loss_clean_mean": clean_metrics["loss_clean_mean"],
            "loss_region_clean": region_metrics["loss_region_clean"],
            "loss_region_clean_upper_component": region_metrics["loss_region_clean_upper_component"],
            "loss_region_clean_middle_component": region_metrics["loss_region_clean_middle_component"],
            "loss_region_clean_lower_component": region_metrics["loss_region_clean_lower_component"],
            "loss_entropy": loss_entropy,
            "loss_entropy_control": loss_entropy,
            "loss_area_high": loss_area_high,
            "loss_area_low": loss_area_low,
            "loss_area_balance": loss_area,
            "mean_pairwise_map_sim": mean_map_sim.detach(),
            "max_pairwise_map_sim": max_map_sim.detach(),
            "mean_pairwise_emb_sim": mean_emb_sim.detach(),
            "max_pairwise_emb_sim": max_emb_sim.detach(),
            "coverage_cosine": coverage_cosine.detach(),
            "foreground_mass_mean": foreground_mass_mean.detach(),
            "foreground_center_mass": foreground_center_mass.detach(),
            "foreground_border_mass": foreground_border_mass.detach(),
            "foreground_prior_sum": foreground_prior_sum.detach(),
            "foreground_safe_mass": foreground_safe_mass.detach(),
            "foreground_face_safe_margin": maps.new_tensor(float(self.face_safe_margin)),
            "foreground_face_safe_tau": maps.new_tensor(float(self.face_safe_tau)),
            "motif_foreground_mass_mean": motif_fg_mean.detach(),
            "motif_foreground_mass_min": motif_fg_min.detach(),
            "motif_foreground_mass_max": motif_fg_max.detach(),
            "center_prior_sigma": maps.new_tensor(float(self.center_prior_sigma)),
            "center_prior_power": maps.new_tensor(float(self.center_prior_power)),
            "soft_border_mass_mean": soft_border_mass_mean.detach(),
            "outer_border_mass_mean": outer_border_mass_mean.detach(),
            "selected_border_mass_mean": selected_border_mass_mean.detach(),
            "selected_outer_border_mass_mean": selected_outer_border_mass_mean.detach(),
            "selected_foreground_mass_mean": selected_foreground_mass_mean.detach(),
            "selection_entropy": selection_entropy.detach(),
            "selection_entropy_soft": selection_entropy.detach(),
            "selection_effective_count": selection_effective_count.detach(),
            "selected_effective_count": selection_effective_count.detach(),
            "selected_pairwise_map_sim": selected_pairwise_map_sim.detach(),
            "selected_pairwise_emb_sim": selected_pairwise_emb_sim.detach(),
            "target_selection_entropy_min": maps.new_tensor(float(self.target_selection_entropy_min)),
            "selected_top_m": model_out.get("selected_top_m", maps.new_tensor(float(maps.shape[1]))).detach(),
            "clean_score_mean": clean_metrics["clean_score_mean"].detach(),
            "clean_score_max": clean_metrics["clean_score_max"].detach(),
            "clean_candidate_count": clean_metrics["clean_candidate_count"].detach(),
            "hard_clean_candidate_count": clean_metrics["hard_clean_candidate_count"].detach(),
            "clean_border_score_mean": clean_metrics["clean_border_score_mean"].detach(),
            "clean_outer_score_mean": clean_metrics["clean_outer_score_mean"].detach(),
            "clean_foreground_score_mean": clean_metrics["clean_foreground_score_mean"].detach(),
            "clean_area_low_score_mean": clean_metrics["clean_area_low_score_mean"].detach(),
            "clean_area_high_score_mean": clean_metrics["clean_area_high_score_mean"].detach(),
            "clean_count_target": maps.new_tensor(float(self.clean_count_target)),
            "motif_upper_mass_mean": region_metrics["motif_upper_mass_mean"].detach(),
            "motif_middle_mass_mean": region_metrics["motif_middle_mass_mean"].detach(),
            "motif_lower_mass_mean": region_metrics["motif_lower_mass_mean"].detach(),
            "upper_clean_count": region_metrics["upper_clean_count"].detach(),
            "middle_clean_count": region_metrics["middle_clean_count"].detach(),
            "lower_clean_count": region_metrics["lower_clean_count"].detach(),
            "min_region_clean_count": region_metrics["min_region_clean_count"].detach(),
            "mean_region_clean_count": region_metrics["mean_region_clean_count"].detach(),
            "region_clean_target": maps.new_tensor(float(self.region_clean_target)),
            "region_clean_upper_weight": maps.new_tensor(float(self.region_clean_upper_weight)),
            "region_clean_middle_weight": maps.new_tensor(float(self.region_clean_middle_weight)),
            "region_clean_lower_weight": maps.new_tensor(float(self.region_clean_lower_weight)),
            "border_mass_mean": border_mass.detach().mean(),
            "center_mass_mean": center_mass.detach().mean(),
            "center_minus_border": center_mass.detach().mean() - border_mass.detach().mean(),
            "border_dominance": torch.relu(border_mass.detach().mean() - center_mass.detach().mean()),
            "effective_area_mean": area.detach().mean(),
            "effective_area_min": area.detach().min(),
            "effective_area_max": area.detach().max(),
            "effective_area_over_max_ratio": over_max.detach(),
            "effective_area_under_min_ratio": under_min.detach(),
            "current_lambda_div_map": maps.new_tensor(float(self.lambda_div_map)),
            "current_lambda_div_emb": maps.new_tensor(float(self.lambda_div_emb)),
            "current_lambda_coverage": maps.new_tensor(float(self.lambda_coverage)),
            "current_lambda_anchor": maps.new_tensor(float(self.lambda_anchor)),
            "current_lambda_border": maps.new_tensor(float(self.lambda_border)),
            "current_lambda_soft_border": maps.new_tensor(float(self.lambda_soft_border)),
            "current_lambda_outer_border": maps.new_tensor(float(self.lambda_outer_border)),
            "current_lambda_selected_border": maps.new_tensor(float(self.lambda_selected_border)),
            "current_lambda_selected_outer_border": maps.new_tensor(float(self.lambda_selected_outer_border)),
            "current_lambda_selected_foreground": maps.new_tensor(float(self.lambda_selected_foreground)),
            "current_lambda_selected_diversity": maps.new_tensor(float(self.lambda_selected_diversity)),
            "current_lambda_selection_entropy": maps.new_tensor(float(self.lambda_selection_entropy)),
            "current_lambda_clean_count": maps.new_tensor(float(self.lambda_clean_count)),
            "current_lambda_clean_mean": maps.new_tensor(float(self.lambda_clean_mean)),
            "current_lambda_region_clean": maps.new_tensor(float(self.lambda_region_clean)),
            "current_lambda_entropy": maps.new_tensor(float(self.lambda_entropy)),
            "current_lambda_area": maps.new_tensor(float(self.lambda_area)),
        }

    def _pairwise_cosine_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1] <= 1:
            zero = x.new_zeros(())
            return zero, zero, zero
        z = F.normalize(x.float(), dim=2, eps=self.eps)
        sim = torch.bmm(z, z.transpose(1, 2))
        k = sim.shape[1]
        mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=sim.device), diagonal=1)
        off = sim[:, mask]
        loss = off.clamp_min(0.0).pow(2).mean().to(dtype=x.dtype)
        return loss, off.mean().to(dtype=x.dtype), off.max().to(dtype=x.dtype)

    def _coverage_loss(
        self,
        maps: torch.Tensor,
        batch: Dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, height, width = maps.shape
        a_total = maps.sum(dim=1).flatten(1)
        a_total = a_total / a_total.sum(dim=1, keepdim=True).clamp_min(self.eps)
        foreground = self._foreground_prior(batch, bsz=bsz, height=height, width=width, device=maps.device, dtype=maps.dtype)
        coverage_cosine = F.cosine_similarity(a_total.float(), foreground.float(), dim=1, eps=self.eps).mean().to(dtype=maps.dtype)
        loss_mse = F.mse_loss(a_total, foreground)
        loss_cosine = 1.0 - coverage_cosine
        if self.coverage_loss_type == "cosine":
            loss = loss_cosine
        elif self.coverage_loss_type == "mse_plus_cosine":
            loss = loss_mse + loss_cosine
        else:
            loss = loss_mse
        foreground_mass_mean = foreground.sum(dim=1).mean().to(dtype=maps.dtype)
        border_mask = self._hard_border_mask(height, width, device=maps.device, dtype=maps.dtype).flatten()
        foreground_border_mass = (foreground * border_mask.view(1, -1)).sum(dim=1).mean().to(dtype=maps.dtype)
        foreground_center_mass = (foreground * (1.0 - border_mask).view(1, -1)).sum(dim=1).mean().to(dtype=maps.dtype)
        safe_mask = self._face_safe_mask(height, width, device=maps.device, dtype=maps.dtype).flatten()
        foreground_prior_sum = foreground.sum(dim=1).mean().to(dtype=maps.dtype)
        foreground_safe_mass = (foreground * safe_mask.view(1, -1)).sum(dim=1).mean().to(dtype=maps.dtype)
        return (
            loss,
            coverage_cosine,
            foreground_mass_mean,
            foreground_center_mass,
            foreground_border_mass,
            foreground_prior_sum,
            foreground_safe_mass,
            foreground,
        )

    def _foreground_anchor_loss(
        self,
        maps: torch.Tensor,
        foreground: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = maps.flatten(2)
        anchor_prior = foreground / foreground.max(dim=1, keepdim=True).values.clamp_min(self.eps)
        motif_mass = (flat * anchor_prior.unsqueeze(1)).sum(dim=2)
        loss = 1.0 - motif_mass.mean()
        if not self.use_foreground_anchor:
            loss = maps.new_zeros(())
        return loss, motif_mass.mean(), motif_mass.min(), motif_mass.max()

    def _selected_utility_losses(
        self,
        model_out: Dict[str, torch.Tensor],
        maps: torch.Tensor,
        emb: torch.Tensor,
        border_mass: torch.Tensor,
        foreground: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        weights = model_out.get("selection_weights")
        if weights is None:
            scores = model_out.get("motif_scores")
            if scores is None:
                weights = maps.new_full(maps.shape[:2], 1.0 / float(max(1, maps.shape[1])))
            else:
                weights = torch.softmax(scores, dim=1)
        weights = weights.to(device=maps.device, dtype=maps.dtype)
        _, _, height, width = maps.shape
        outer_mask = self._hard_border_mask(
            height,
            width,
            device=maps.device,
            dtype=maps.dtype,
            border_width=self.outer_border_width,
        )
        outer_border_mass = (maps * outer_mask.view(1, 1, height, width)).sum(dim=(2, 3))

        foreground_anchor = foreground / foreground.max(dim=1, keepdim=True).values.clamp_min(self.eps)
        motif_foreground_mass = (maps.flatten(2) * foreground_anchor.unsqueeze(1)).sum(dim=2)

        selected_border = (weights * border_mass).sum(dim=1)
        selected_outer_border = (weights * outer_border_mass).sum(dim=1)
        selected_foreground = (weights * motif_foreground_mass).sum(dim=1)
        loss_selected_border = selected_border.mean()
        loss_selected_outer_border = selected_outer_border.mean()
        loss_selected_foreground = (1.0 - selected_foreground).mean()
        loss_selected_diversity, selected_pairwise_map_sim, selected_pairwise_emb_sim = self._selected_diversity_loss(
            maps,
            emb,
            weights,
        )
        selection_entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1).mean()
        selection_effective_count = selection_entropy.exp()
        loss_selection_entropy = F.relu(weights.new_tensor(float(self.target_selection_entropy_min)) - selection_entropy)
        return (
            loss_selected_border,
            loss_selected_outer_border,
            loss_selected_foreground,
            loss_selected_diversity,
            selected_border.mean(),
            selected_outer_border.mean(),
            selected_foreground.mean(),
            selection_entropy,
            selection_effective_count,
            loss_selection_entropy,
            selected_pairwise_map_sim,
            selected_pairwise_emb_sim,
            motif_foreground_mass,
        )

    def _clean_candidate_losses(
        self,
        maps: torch.Tensor,
        border_mass: torch.Tensor,
        motif_foreground_mass: torch.Tensor,
        area: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _, _, height, width = maps.shape
        outer_mask = self._hard_border_mask(
            height,
            width,
            device=maps.device,
            dtype=maps.dtype,
            border_width=self.outer_border_width,
        )
        outer_border_mass = (maps * outer_mask.view(1, 1, height, width)).sum(dim=(2, 3))
        clean = compute_clean_candidate_scores(
            border_mass=border_mass,
            outer_border_mass=outer_border_mass,
            foreground_mass=motif_foreground_mass,
            effective_area=area,
            clean_border_threshold=self.clean_border_threshold,
            clean_outer_threshold=self.clean_outer_threshold,
            clean_foreground_threshold=self.clean_foreground_threshold,
            clean_tau=self.clean_tau,
            clean_area_tau=self.clean_area_tau,
            clean_min_effective_area=self.clean_min_effective_area,
            clean_max_effective_area=self.clean_max_effective_area,
        )
        clean_score = clean["clean_score"]
        target = maps.new_tensor(float(self.clean_count_target))
        loss_clean_count = (F.relu(target - clean["clean_candidate_count_per_sample"]) / target).mean()
        loss_clean_mean = 1.0 - clean_score.mean()
        if not self.use_clean_candidate_loss:
            loss_clean_count = maps.new_zeros(())
            loss_clean_mean = maps.new_zeros(())
        return {
            "clean_score": clean_score,
            "loss_clean_count": loss_clean_count,
            "loss_clean_mean": loss_clean_mean,
            "clean_score_mean": clean_score.mean(),
            "clean_score_max": clean_score.max(),
            "clean_candidate_count": clean["clean_candidate_count"],
            "hard_clean_candidate_count": clean["hard_clean_candidate_count"],
            "clean_border_score_mean": clean["clean_border_score"].mean(),
            "clean_outer_score_mean": clean["clean_outer_score"].mean(),
            "clean_foreground_score_mean": clean["clean_foreground_score"].mean(),
            "clean_area_low_score_mean": clean["clean_area_low_score"].mean(),
            "clean_area_high_score_mean": clean["clean_area_high_score"].mean(),
        }

    def _region_clean_losses(
        self,
        maps: torch.Tensor,
        clean_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        region = compute_soft_region_masses(maps, region_tau=self.region_tau)
        upper_mass = region["upper_mass"]
        middle_mass = region["middle_mass"]
        lower_mass = region["lower_mass"]
        upper_clean = (clean_score * upper_mass).sum(dim=1)
        middle_clean = (clean_score * middle_mass).sum(dim=1)
        lower_clean = (clean_score * lower_mass).sum(dim=1)
        target = maps.new_tensor(float(self.region_clean_target))
        upper_weight = maps.new_tensor(float(self.region_clean_upper_weight))
        middle_weight = maps.new_tensor(float(self.region_clean_middle_weight))
        lower_weight = maps.new_tensor(float(self.region_clean_lower_weight))
        weight_sum = (upper_weight + middle_weight + lower_weight).clamp_min(self.eps)
        denom = weight_sum * target
        upper_component = (upper_weight * F.relu(target - upper_clean) / denom).mean()
        middle_component = (middle_weight * F.relu(target - middle_clean) / denom).mean()
        lower_component = (lower_weight * F.relu(target - lower_clean) / denom).mean()
        loss_region_clean = upper_component + middle_component + lower_component
        if not self.use_region_clean_loss:
            loss_region_clean = maps.new_zeros(())
            upper_component = maps.new_zeros(())
            middle_component = maps.new_zeros(())
            lower_component = maps.new_zeros(())
        region_counts = torch.stack([upper_clean, middle_clean, lower_clean], dim=1)
        return {
            "loss_region_clean": loss_region_clean,
            "loss_region_clean_upper_component": upper_component,
            "loss_region_clean_middle_component": middle_component,
            "loss_region_clean_lower_component": lower_component,
            "motif_upper_mass_mean": upper_mass.mean(),
            "motif_middle_mass_mean": middle_mass.mean(),
            "motif_lower_mass_mean": lower_mass.mean(),
            "upper_clean_count": upper_clean.mean(),
            "middle_clean_count": middle_clean.mean(),
            "lower_clean_count": lower_clean.mean(),
            "min_region_clean_count": region_counts.min(dim=1).values.mean(),
            "mean_region_clean_count": region_counts.mean(dim=1).mean(),
        }

    def _selected_diversity_loss(
        self,
        maps: torch.Tensor,
        emb: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if maps.shape[1] <= 1:
            zero = maps.new_zeros(())
            return zero, zero, zero
        map_flat = F.normalize(maps.flatten(2).float(), dim=2, eps=self.eps)
        map_sim = torch.bmm(map_flat, map_flat.transpose(1, 2)).clamp_min(0.0).to(dtype=maps.dtype)
        emb_norm = F.normalize(emb.float(), dim=2, eps=self.eps)
        emb_sim = torch.bmm(emb_norm, emb_norm.transpose(1, 2)).clamp_min(0.0).to(dtype=maps.dtype)
        k = map_sim.shape[1]
        pair_mask = torch.triu(torch.ones(k, k, device=maps.device, dtype=maps.dtype), diagonal=1)
        pair_weights = weights.unsqueeze(2) * weights.unsqueeze(1) * pair_mask.view(1, k, k)
        denom = pair_weights.sum().clamp_min(self.eps)
        selected_map_sim = (map_sim * pair_weights).sum() / denom
        selected_emb_sim = (emb_sim * pair_weights).sum() / denom
        return selected_map_sim, selected_map_sim, selected_emb_sim

    def _foreground_prior(
        self,
        batch: Dict[str, torch.Tensor] | None,
        bsz: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        node_features = None if batch is None else batch.get("node_features", batch.get("x"))
        if torch.is_tensor(node_features) and node_features.ndim == 3 and node_features.shape[-1] >= 7:
            feats = node_features.to(device=device, dtype=dtype)
            foreground = torch.relu(feats[..., 5]) + torch.relu(feats[..., 6])
        else:
            foreground = torch.zeros(bsz, height * width, device=device, dtype=dtype)
        raw_mass = foreground.sum(dim=1, keepdim=True)
        center_prior = self._center_prior(bsz, height, width, device=device, dtype=dtype)
        if self.use_face_safe_foreground:
            face_safe = self._face_safe_mask(height, width, device=device, dtype=dtype).flatten().view(1, -1)
            center_weight = center_prior.clamp_min(self.eps).pow(self.center_prior_power)
            foreground = foreground * face_safe * center_weight
            fallback = face_safe * center_prior
            fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(self.eps)
        else:
            fallback = center_prior
            if self.use_center_weighted_foreground:
                foreground = foreground * center_prior.clamp_min(self.eps).pow(self.center_prior_power)
        mass = foreground.sum(dim=1, keepdim=True)
        use_raw = raw_mass > self.eps
        foreground = torch.where(use_raw, foreground / mass.clamp_min(self.eps), fallback)
        return foreground

    def _center_prior(
        self,
        bsz: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = max(float(self.center_prior_sigma), self.eps)
        prior = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma)).flatten()
        prior = prior / prior.sum().clamp_min(self.eps)
        return prior.view(1, -1).expand(bsz, -1)

    def _face_safe_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        margin = float(self.face_safe_margin)
        tau = max(float(self.face_safe_tau), self.eps)
        safe_x = torch.sigmoid((xx - margin) / tau) * torch.sigmoid((1.0 - margin - xx) / tau)
        safe_y = torch.sigmoid((yy - margin) / tau) * torch.sigmoid((1.0 - margin - yy) / tau)
        return (safe_x * safe_y).clamp(0.0, 1.0)

    def _soft_border_loss(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_soft_border_penalty:
            zero = maps.new_zeros(())
            return zero, zero
        _, _, height, width = maps.shape
        penalty = self._soft_border_penalty_map(height, width, device=maps.device, dtype=maps.dtype)
        mass = (maps * penalty.view(1, 1, height, width)).sum(dim=(2, 3))
        return mass.mean(), mass.mean()

    def _outer_border_loss(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_outer_border_penalty:
            zero = maps.new_zeros(())
            return zero, zero
        _, _, height, width = maps.shape
        mask = self._hard_border_mask(
            height,
            width,
            device=maps.device,
            dtype=maps.dtype,
            border_width=self.outer_border_width,
        )
        mass = (maps * mask.view(1, 1, height, width)).sum(dim=(2, 3))
        return mass.mean(), mass.mean()

    def _soft_border_penalty_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dist_to_edge = torch.minimum(
            torch.minimum(xx, 1.0 - xx),
            torch.minimum(yy, 1.0 - yy),
        )
        penalty = torch.exp(-dist_to_edge / self.soft_border_tau)
        return penalty.clamp_min(0.0)

    def _hard_border_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        border_width: int = 4,
    ) -> torch.Tensor:
        mask = torch.zeros(height, width, device=device, dtype=dtype)
        bw = max(0, int(border_width))
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask

    def _entropy_control_loss(self, area: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = F.relu(float(self.area_min) - area) / max(self.area_min, self.eps)
        high = F.relu(area - float(self.area_max)) / max(self.area_max, self.eps)
        loss_area_high = high.mean()
        loss_area_low = low.mean()
        return loss_area_high + loss_area_low, loss_area_high, loss_area_low

    def _area_balance_loss(self, area: torch.Tensor) -> torch.Tensor:
        scaled = area / max(self.area_max, self.eps)
        mean = scaled.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return (scaled / mean - 1.0).pow(2).mean()

    def _effective_area(self, maps: torch.Tensor) -> torch.Tensor:
        flat = maps.flatten(2)
        probs = flat / flat.sum(dim=2, keepdim=True).clamp_min(self.eps)
        entropy = -(probs * probs.clamp_min(self.eps).log()).sum(dim=2)
        return entropy.exp()
