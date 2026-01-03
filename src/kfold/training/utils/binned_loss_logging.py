from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torchmetrics import MeanMetric

from kfold.training.utils.entity_binning import entity_bin_index_from_asym_id
from kfold.training.utils.time_binning import binned_sum_and_count, compute_bin_index


@dataclass(frozen=True)
class TimeBinConfig:
    enabled: bool
    width: float = 0.1


@dataclass(frozen=True)
class EntityBinConfig:
    enabled: bool
    nbins: int = 10  # interval1..9, >=10


def _timebin_labels(width: float) -> list[str]:
    nbins = int(round(1.0 / float(width)))
    return [f"u{i:02d}_{i + 1:02d}" for i in range(nbins)]


def _entitybin_labels(nbins: int) -> list[str]:
    return [f"interval{i}" for i in range(1, nbins + 1)]


def _get_time_bounds(structure_module: Any) -> tuple[float, float]:
    """Return (t_min, t_max) bounds used for u-normalization.

    - ECSI: sigma_min/max are already in [0, 1].
    - EDM/AF3/Boltz-style: sigma_min/max are relative and typically scaled by sigma_data.
    """
    t_min = float(structure_module.sigma_min)
    t_max = float(structure_module.sigma_max)
    if hasattr(structure_module, "sigma_data") and t_max > 1.0:
        sigma_data = float(structure_module.sigma_data)
        t_min *= sigma_data
        t_max *= sigma_data
    return t_min, t_max


def _per_bin_update(
    metrics: torch.nn.ModuleDict,
    labels: list[str],
    name: str,
    values: torch.Tensor,
    bin_index: torch.Tensor,
    nbins: int,
) -> None:
    """Update MeanMetric objects by (bin_mean, bin_count) for each bin."""
    bin_sum, bin_count = binned_sum_and_count(values, bin_index, nbins)
    for b_idx, label in enumerate(labels):
        c = bin_count[b_idx]
        if float(c.item()) <= 0.0:
            continue
        m = bin_sum[b_idx] / c
        metrics[f"{label}__{name}"].update(m, c)


class TimeBinnedLossLogger(torch.nn.Module):
    """
    Epoch-level aggregation of diffusion losses by normalized diffusion time u∈[0,1].
    """

    def __init__(self, cfg: TimeBinConfig):
        super().__init__()
        self.enabled = bool(cfg.enabled)
        self.width = float(cfg.width)
        self.labels = _timebin_labels(self.width)
        self.nbins = len(self.labels)

        md: dict[str, MeanMetric] = {}
        for label in self.labels:
            for name in [
                "loss",
                "mse_loss",
                "bond_loss",
                "smooth_lddt_loss",
                "diffusion_loss",
            ]:
                md[f"{label}__{name}"] = MeanMetric()
        self.metrics = torch.nn.ModuleDict(md)

    def update(
        self,
        *,
        t_hat: torch.Tensor,
        structure_module: Any,
        diffusion_per_sample: dict[str, torch.Tensor],
        distogram_loss_per_batch: torch.Tensor,
        loss_weights: dict[str, float],
    ) -> None:
        if not self.enabled:
            return
        if self.nbins <= 0:
            return

        t_min, t_max = _get_time_bounds(structure_module)
        denom = max(t_max - t_min, 1e-12)
        u = ((t_hat - t_min) / denom).clamp_(0.0, 1.0)

        bin_index = compute_bin_index(u, self.nbins)

        for name, values in diffusion_per_sample.items():
            _per_bin_update(
                self.metrics, self.labels, name, values, bin_index, self.nbins
            )

        diffusion_weight = float(loss_weights["diffusion"])
        distogram_weight = float(loss_weights["distogram"])
        disto_term = distogram_loss_per_batch * distogram_weight  # [B]
        diffusion_term = (
            diffusion_per_sample["diffusion_loss"] * diffusion_weight
        )  # [B, N]
        total_per_sample = diffusion_term + disto_term[:, None]
        _per_bin_update(
            self.metrics,
            self.labels,
            "loss",
            total_per_sample,
            bin_index,
            self.nbins,
        )

    def flush(self) -> dict[str, torch.Tensor]:
        """Return metrics to log and reset internal state."""
        if not self.enabled:
            return {}

        out: dict[str, torch.Tensor] = {}
        for key, m in self.metrics.items():
            v = m.compute()
            if not v.isfinite().all():
                continue
            label, name = key.split("__", 1)
            try:
                interval_idx = self.labels.index(label) + 1
            except ValueError:
                interval_idx = None
            if interval_idx is not None:
                # Flat namespace for W&B panels
                out[f"train_time_bin/{name}_interval{interval_idx}"] = v
            m.reset()
        return out


class EntityBinnedLossLogger(torch.nn.Module):
    """Epoch-level aggregation of diffusion losses by entity count (unique asym_id)."""

    def __init__(self, cfg: EntityBinConfig):
        super().__init__()
        self.enabled = bool(cfg.enabled)
        self.nbins = int(cfg.nbins)
        self.labels = _entitybin_labels(self.nbins)

        md: dict[str, MeanMetric] = {}
        for label in self.labels:
            for name in [
                "loss",
                "mse_loss",
                "bond_loss",
                "smooth_lddt_loss",
                "diffusion_loss",
            ]:
                md[f"{label}__{name}"] = MeanMetric()
        self.metrics = torch.nn.ModuleDict(md)

    def update(
        self,
        *,
        f_input: Any,
        diffusion_per_sample: dict[str, torch.Tensor],
        distogram_loss_per_batch: torch.Tensor,
        loss_weights: dict[str, float],
    ) -> None:
        if not self.enabled:
            return
        if self.nbins <= 0:
            return

        entity_bin = entity_bin_index_from_asym_id(
            asym_id=f_input.token.asym_id, pad_mask=f_input.token.pad_mask
        )
        entity_bin = torch.clamp(entity_bin, 0, self.nbins - 1)

        # Expand to [B, N] for binned aggregation
        # (all noise samples share the same entity bin)
        any_val = next(iter(diffusion_per_sample.values()))
        B, N = any_val.shape[:2]
        bin_index = entity_bin[:, None].expand(B, N)

        for name, values in diffusion_per_sample.items():
            _per_bin_update(
                self.metrics, self.labels, name, values, bin_index, self.nbins
            )

        diffusion_weight = float(loss_weights["diffusion"])
        distogram_weight = float(loss_weights["distogram"])
        disto_term = distogram_loss_per_batch * distogram_weight  # [B]
        diffusion_term = (
            diffusion_per_sample["diffusion_loss"] * diffusion_weight
        )  # [B, N]
        total_per_sample = diffusion_term + disto_term[:, None]
        _per_bin_update(
            self.metrics,
            self.labels,
            "loss",
            total_per_sample,
            bin_index,
            self.nbins,
        )

    def flush(self) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}

        out: dict[str, torch.Tensor] = {}
        for key, m in self.metrics.items():
            v = m.compute()
            if not v.isfinite().all():
                continue
            label, name = key.split("__", 1)
            interval_idx = int(label.replace("interval", ""))
            out[f"train_entity_bin/{name}_interval{interval_idx}"] = v
            m.reset()
        return out
