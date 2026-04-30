"""Losses for D5 class-level graph retrieval."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights(
    class_counts,
    normalize_mean: bool = True,
    power: float = 1.0,
) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights.pow(float(power))
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-8)
    return weights


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, y.long(), weight=weight)


class FixedMotifClassificationLoss(nn.Module):
    """Cross-entropy loss for the D5B fixed motif classifier."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_cls = self.ce(model_out["logits"], y.long())
        return {
            "loss": loss_cls,
            "loss_cls": loss_cls,
        }


class D5RetrievalLoss(nn.Module):
    """CE plus soft-subgraph regularizers for D5A retrieval."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.lambda_cls = float(cfg.get("lambda_cls", 1.0))
        self.lambda_contrast = float(cfg.get("lambda_contrast", 0.1))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 0.01))
        self.lambda_closure = float(cfg.get("lambda_closure", 0.01))
        self.lambda_area = float(cfg.get("lambda_area", 0.01))
        self.lambda_div = float(cfg.get("lambda_div", 0.0))
        self.lambda_prior = float(cfg.get("lambda_prior", 0.0))
        self.prior_loss_type = str(cfg.get("prior_loss_type", "mse")).lower()
        self.margin = float(cfg.get("margin", 0.2))
        self.target_area = float(cfg.get("target_area", 0.15))
        if self.prior_loss_type != "mse":
            raise ValueError(f"Unsupported prior_loss_type: {self.prior_loss_type}")

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        node_attn = model_out["node_attn"]
        edge_attn = model_out["edge_attn"]
        edge_index = batch["edge_index"].long()
        y = y.long()

        loss_cls = self.ce(logits, y)
        loss_contrast = self._contrast_loss(logits, y)
        loss_smooth = self._smoothness_loss(node_attn, edge_index, y)
        loss_closure = self._closure_loss(node_attn, edge_attn, edge_index, y)
        loss_area = self._area_loss(node_attn)
        loss_div = self._diversity_loss(model_out)
        loss_prior = self._prior_loss(model_out)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_contrast * loss_contrast
            + self.lambda_smooth * loss_smooth
            + self.lambda_closure * loss_closure
            + self.lambda_area * loss_area
            + self.lambda_div * loss_div
            + self.lambda_prior * loss_prior
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_contrast": loss_contrast,
            "loss_smooth": loss_smooth,
            "loss_closure": loss_closure,
            "loss_area": loss_area,
            "loss_div": loss_div,
            "loss_prior": loss_prior,
        }

    def _contrast_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz, num_classes = logits.shape
        true_score = logits[torch.arange(bsz, device=logits.device), y]
        wrong = logits.masked_fill(F.one_hot(y, num_classes=num_classes).bool(), -1e9)
        max_wrong = wrong.max(dim=1).values
        return F.relu(self.margin - true_score + max_wrong).mean()

    @staticmethod
    def _smoothness_loss(
        node_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        true_attn = node_attn[torch.arange(bsz, device=node_attn.device), y]
        src = edge_index[0]
        dst = edge_index[1]
        return (true_attn[:, src] - true_attn[:, dst]).pow(2).mean()

    @staticmethod
    def _closure_loss(
        node_attn: torch.Tensor,
        edge_attn: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        bsz = node_attn.shape[0]
        rows = torch.arange(bsz, device=node_attn.device)
        node_true = node_attn[rows, y]
        edge_true = edge_attn[rows, y]
        src = edge_index[0]
        dst = edge_index[1]
        endpoint_min = torch.minimum(node_true[:, src], node_true[:, dst])
        return F.relu(edge_true - endpoint_min).pow(2).mean()

    def _area_loss(self, node_attn: torch.Tensor) -> torch.Tensor:
        mass = node_attn.mean(dim=-1)
        return (mass - self.target_area).pow(2).mean()

    @staticmethod
    def _diversity_loss(model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        gate = model_out.get("class_node_gate")
        if gate is None or gate.shape[0] <= 1:
            logits = model_out["logits"]
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        flat = F.normalize(gate.float(), dim=1, eps=1e-6)
        sim = flat @ flat.t()
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    def _prior_loss(self, model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model_out["logits"]
        if self.lambda_prior <= 0.0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        gate = model_out.get("class_node_gate")
        prior = model_out.get("motif_node_prior")
        if gate is None:
            raise KeyError("lambda_prior > 0 requires model_out['class_node_gate']")
        if prior is None:
            raise KeyError("lambda_prior > 0 requires a loaded motif_node_prior")
        prior = prior.to(device=gate.device, dtype=gate.dtype)
        if tuple(prior.shape) != tuple(gate.shape):
            raise ValueError(f"Prior shape {tuple(prior.shape)} does not match gate shape {tuple(gate.shape)}")
        return F.mse_loss(gate, prior)


class D6HierarchicalMotifLoss(nn.Module):
    """CE plus soft-slot regularizers for D6A hierarchical motifs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        self.lambda_cls = float(cfg.get("lambda_cls", 1.0))
        self.lambda_slot_div = float(cfg.get("lambda_slot_div", 0.01))
        self.lambda_border = float(cfg.get("lambda_border", 0.005))
        self.lambda_slot_balance = float(cfg.get("lambda_slot_balance", 0.0))
        self.lambda_slot_smooth = float(cfg.get("lambda_slot_smooth", 0.0))
        self.border_width = int(cfg.get("border_width", 3))
        self.border_loss_type = str(cfg.get("border_loss_type", "pixel_mean")).lower()
        self.slot_balance_type = str(cfg.get("slot_balance_type", "kl_uniform")).lower()
        self.height = int(cfg.get("height", 48))
        self.width = int(cfg.get("width", 48))
        self.eps = float(cfg.get("eps", 1e-6))

        class_weights = None
        if cfg.get("use_class_weights", True):
            counts = cfg.get("class_counts")
            if counts is None:
                raise ValueError("loss.use_class_weights=true requires loss.class_counts")
            class_weights = compute_class_weights(
                counts,
                normalize_mean=True,
                power=float(cfg.get("class_weight_power", 1.0)),
            )
        self.ce = WeightedCrossEntropy(class_weights)
        self.register_buffer("border_mask", self._make_border_mask(), persistent=False)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        part_masks = model_out["part_masks"]
        loss_ce = self.ce(logits, y.long())
        loss_slot_div = self._slot_diversity_loss(part_masks)
        loss_border = self._border_loss(part_masks)
        loss_slot_balance = self._slot_balance_loss(part_masks)
        loss_slot_smooth = self._slot_smoothness_loss(part_masks, batch.get("edge_index"))
        total = (
            self.lambda_cls * loss_ce
            + self.lambda_slot_div * loss_slot_div
            + self.lambda_border * loss_border
            + self.lambda_slot_balance * loss_slot_balance
            + self.lambda_slot_smooth * loss_slot_smooth
        )
        return {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_slot_div": loss_slot_div,
            "loss_border": loss_border,
            "loss_slot_balance": loss_slot_balance,
            "loss_slot_smooth": loss_slot_smooth,
        }

    def _slot_diversity_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        m = part_masks / part_masks.norm(dim=2, keepdim=True).clamp_min(self.eps)
        sim = torch.bmm(m, m.transpose(1, 2))
        k = sim.shape[1]
        off_diag = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device).unsqueeze(0))
        return off_diag.mean()

    def _border_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        border_mask = self.border_mask.to(device=part_masks.device, dtype=part_masks.dtype)
        if self.border_loss_type in ("pixel_mean", "d6a_pixel_mean"):
            return (part_masks * border_mask.view(1, 1, -1)).mean()
        if self.border_loss_type in ("slot_ratio", "slot_border_ratio"):
            border_mass = (part_masks * border_mask.view(1, 1, -1)).sum(dim=2)
            slot_mass = part_masks.sum(dim=2).clamp_min(self.eps)
            return (border_mass / slot_mass).mean()
        if self.border_loss_type in ("dominant", "dominant_border"):
            dominant = part_masks.max(dim=1).values
            border_pixels = border_mask.bool()
            if not bool(border_pixels.any()):
                return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
            return dominant[:, border_pixels].mean()
        raise ValueError(f"Unsupported border_loss_type: {self.border_loss_type}")

    def _slot_balance_loss(self, part_masks: torch.Tensor) -> torch.Tensor:
        if self.lambda_slot_balance <= 0.0:
            return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
        if self.slot_balance_type != "kl_uniform":
            raise ValueError(f"Unsupported slot_balance_type: {self.slot_balance_type}")
        area = part_masks.mean(dim=2)
        area_norm = area / area.sum(dim=1, keepdim=True).clamp_min(self.eps)
        k = area_norm.shape[1]
        return (area_norm * (area_norm.clamp_min(self.eps).log() + math.log(float(k)))).sum(dim=1).mean()

    def _slot_smoothness_loss(
        self,
        part_masks: torch.Tensor,
        edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.lambda_slot_smooth <= 0.0 or edge_index is None:
            return torch.zeros((), device=part_masks.device, dtype=part_masks.dtype)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        return (part_masks[:, :, src] - part_masks[:, :, dst]).abs().mean()

    def _make_border_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.float32)
        bw = int(self.border_width)
        if bw > 0:
            mask[:bw, :] = 1.0
            mask[-bw:, :] = 1.0
            mask[:, :bw] = 1.0
            mask[:, -bw:] = 1.0
        return mask.reshape(-1)


class D6ClassAttendedMotifLoss(D6HierarchicalMotifLoss):
    """D6B loss plus class-conditioned objectives for D6C-light."""

    DEFAULT_CONFUSION_PAIRS = ((2, 4), (2, 6), (2, 5), (4, 6), (0, 1))
    PAIR_DIAG_NAMES = {
        (2, 4): "fear_sad",
        (2, 6): "fear_neutral",
        (2, 5): "fear_surprise",
        (4, 6): "sad_neutral",
        (0, 1): "angry_disgust",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        super().__init__(cfg)
        self.lambda_class_border = float(cfg.get("lambda_class_border", 0.0025))
        self.lambda_class_attn_sep = float(cfg.get("lambda_class_attn_sep", 0.005))
        self.lambda_supcon = float(cfg.get("lambda_supcon", 0.03))
        self.class_attn_sep_margin = float(cfg.get("class_attn_sep_margin", 0.90))
        self.supcon_temperature = float(cfg.get("supcon_temperature", 0.2))
        self.confusion_pairs = self._parse_confusion_pairs(cfg.get("confusion_pairs", self.DEFAULT_CONFUSION_PAIRS))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        part_masks = model_out["part_masks"]
        y = y.long()

        loss_ce = self.ce(logits, y)
        loss_slot_div = self._slot_diversity_loss(part_masks)
        loss_border = self._border_loss(part_masks)
        loss_slot_balance = self._slot_balance_loss(part_masks)
        loss_slot_smooth = self._slot_smoothness_loss(part_masks, batch.get("edge_index"))
        loss_class_border, class_border_diag = self._class_attended_border_loss(model_out, y)
        loss_class_attn_sep, class_attn_diag = self._class_attention_separation_loss(model_out)
        loss_supcon, supcon_diag = self._supervised_contrastive_loss(model_out, y)

        total = (
            self.lambda_cls * loss_ce
            + self.lambda_slot_div * loss_slot_div
            + self.lambda_border * loss_border
            + self.lambda_slot_balance * loss_slot_balance
            + self.lambda_slot_smooth * loss_slot_smooth
            + self.lambda_class_border * loss_class_border
            + self.lambda_class_attn_sep * loss_class_attn_sep
            + self.lambda_supcon * loss_supcon
        )

        out = {
            "loss": total,
            "total_loss": total,
            "loss_ce": loss_ce,
            "loss_slot_div": loss_slot_div,
            "loss_border": loss_border,
            "loss_slot_balance": loss_slot_balance,
            "loss_slot_smooth": loss_slot_smooth,
            "loss_class_border": loss_class_border,
            "loss_class_attn_sep": loss_class_attn_sep,
            "loss_supcon": loss_supcon,
        }
        out.update(class_border_diag)
        out.update(class_attn_diag)
        out.update(supcon_diag)
        return out

    def _class_attended_border_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        part_masks = model_out.get("part_masks")
        class_part_attn = model_out.get("class_part_attn")
        zero = logits.new_zeros(())
        if not torch.is_tensor(part_masks) or not torch.is_tensor(class_part_attn):
            return zero, {
                "diag_true_class_border_mass_mean": zero,
                "diag_true_class_border_mass_max": zero,
            }

        border_mask = self.border_mask.to(device=part_masks.device, dtype=part_masks.dtype)
        if border_mask.numel() != part_masks.shape[-1]:
            raise ValueError(
                f"Border mask has {border_mask.numel()} pixels, but part_masks has {part_masks.shape[-1]} nodes"
            )
        class_pixel_attn = torch.einsum("bck,bkn->bcn", class_part_attn, part_masks)
        rows = torch.arange(y.shape[0], device=y.device)
        true_motif = class_pixel_attn[rows, y, :]
        border_mass = (true_motif * border_mask.view(1, -1)).sum(dim=1)
        motif_mass = true_motif.sum(dim=1).clamp_min(self.eps)
        border_ratio = border_mass / motif_mass
        return border_ratio.mean(), {
            "diag_true_class_border_mass_mean": border_ratio.detach().mean(),
            "diag_true_class_border_mass_max": border_ratio.detach().max(),
        }

    def _class_attention_separation_loss(
        self,
        model_out: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        class_part_attn = model_out.get("class_part_attn")
        zero = logits.new_zeros(())
        diag: Dict[str, torch.Tensor] = {
            "diag_class_attn_sim_fear_sad": zero,
            "diag_class_attn_sim_fear_neutral": zero,
            "diag_class_attn_sim_fear_surprise": zero,
            "diag_class_attn_sim_sad_neutral": zero,
            "diag_class_attn_sim_angry_disgust": zero,
            "diag_class_part_entropy_mean": zero,
        }
        if not torch.is_tensor(class_part_attn):
            return zero, diag

        attn = class_part_attn.float()
        entropy = -(attn * attn.clamp_min(self.eps).log()).sum(dim=2)
        diag["diag_class_part_entropy_mean"] = entropy.detach().mean()
        avg_attn = attn.mean(dim=0)
        penalties = []
        for left, right in self.confusion_pairs:
            if avg_attn.shape[0] <= max(left, right):
                continue
            sim = F.cosine_similarity(avg_attn[left], avg_attn[right], dim=0, eps=self.eps)
            penalties.append(F.relu(sim - self.class_attn_sep_margin))
            pair_name = self.PAIR_DIAG_NAMES.get((left, right), self.PAIR_DIAG_NAMES.get((right, left)))
            if pair_name is not None:
                diag[f"diag_class_attn_sim_{pair_name}"] = sim.detach()

        if not penalties:
            return zero, diag
        return torch.stack(penalties).mean().to(dtype=logits.dtype), diag

    def _supervised_contrastive_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model_out["logits"]
        class_repr = model_out.get("class_repr")
        zero = logits.new_zeros(())
        diag = {
            "diag_supcon_valid_anchors": zero,
            "diag_supcon_positive_pairs": zero,
        }
        if not torch.is_tensor(class_repr) or class_repr.shape[0] <= 1:
            return zero, diag

        rows = torch.arange(y.shape[0], device=y.device)
        z = class_repr[rows, y, :].float()
        z = F.normalize(z, dim=-1, eps=self.eps)
        temperature = max(self.supcon_temperature, self.eps)
        sim = (z @ z.transpose(0, 1)) / temperature
        bsz = sim.shape[0]
        self_mask = torch.eye(bsz, dtype=torch.bool, device=sim.device)
        positive_mask = y.view(-1, 1).eq(y.view(1, -1)) & ~self_mask
        valid_anchor = positive_mask.any(dim=1)
        if not bool(valid_anchor.any()):
            return zero, diag

        non_self_logits = sim.masked_fill(self_mask, -float("inf"))
        max_logits = non_self_logits.max(dim=1, keepdim=True).values
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        stable_logits = sim - max_logits.detach()
        exp_logits = stable_logits.exp().masked_fill(self_mask, 0.0)
        log_prob = stable_logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps).log()
        pos_float = positive_mask.to(dtype=log_prob.dtype)
        pos_count = pos_float.sum(dim=1).clamp_min(1.0)
        per_anchor = -(log_prob * pos_float).sum(dim=1) / pos_count
        loss = per_anchor[valid_anchor].mean()
        diag["diag_supcon_valid_anchors"] = valid_anchor.float().sum().detach()
        diag["diag_supcon_positive_pairs"] = positive_mask.float().sum().detach()
        return loss.to(dtype=logits.dtype), diag

    @staticmethod
    def _parse_confusion_pairs(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
        pairs = []
        for item in value:
            if len(item) != 2:
                raise ValueError(f"Each confusion pair must have two class indices, got {item}")
            pairs.append((int(item[0]), int(item[1])))
        return tuple(pairs)


def build_loss(config: Dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    name = str(cfg.get("name", "d5_retrieval")).lower()
    if name in ("d5_retrieval", "class_pixel_motif_retrieval"):
        return D5RetrievalLoss(cfg)
    if name in ("d6_hierarchical_motif", "d6a_hierarchical_motif"):
        return D6HierarchicalMotifLoss(cfg)
    if name in ("d6b_class_part_motif", "d6b_class_part_attention_motif"):
        cfg.setdefault("border_loss_type", "slot_ratio")
        cfg.setdefault("slot_balance_type", "kl_uniform")
        return D6HierarchicalMotifLoss(cfg)
    if name in ("d6c_class_attended_motif", "d6c_class_attended_objectives"):
        return D6ClassAttendedMotifLoss(cfg)
    if name in ("fixed_motif_classification", "d5b_fixed_motif"):
        return FixedMotifClassificationLoss(cfg)
    raise ValueError(f"Unknown loss: {name}")
