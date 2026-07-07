import math
from typing import TypedDict

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LayerNorm, Linear
from kfold.utils.registry import BaseConfig

CROP_UNKNOWN = 0
CROP_CONTIGUOUS = 1
CROP_SPATIAL = 2
CROP_SPATIAL_INTERFACE = 3


class _Patch(TypedDict):
    idx: torch.Tensor
    chain_id: int
    is_interface: bool
    is_background: bool


class _PatchPair(TypedDict):
    patch_i: int
    patch_j: int
    target: torch.Tensor
    weight: float
    hard_negative: bool
    priority: float


class PatchPairGeometryHead(nn.Module):
    class Config(BaseConfig):
        enabled: bool = False
        channel_z: int = 256
        num_bins: int = 64
        min_dist: float = 2.0
        max_dist: float = 22.0
        patch_size: int = 16
        max_patches_per_chain: int = 8
        max_patch_pairs: int = 256
        interface_cutoff: float = 12.0
        positive_cutoff: float = 12.0
        hard_negative_cutoff: float = 22.0

    def __init__(self, cfg: Config, channel_z: int | None = None):
        super().__init__()
        self.enabled: bool = bool(cfg.enabled)
        self.channel_z: int = int(cfg.channel_z if channel_z is None else channel_z)
        self.num_bins: int = int(cfg.num_bins)
        self.min_dist: float = float(cfg.min_dist)
        self.max_dist: float = float(cfg.max_dist)
        self.patch_size: int = int(cfg.patch_size)
        self.max_patches_per_chain: int = int(cfg.max_patches_per_chain)
        self.max_patch_pairs: int = int(cfg.max_patch_pairs)
        self.interface_cutoff: float = float(cfg.interface_cutoff)
        self.positive_cutoff: float = float(cfg.positive_cutoff)
        self.hard_negative_cutoff: float = float(cfg.hard_negative_cutoff)

        bin_size = (self.max_dist - self.min_dist) / self.num_bins
        first_bin = self.min_dist + bin_size
        last_bin = self.max_dist - bin_size
        boundaries = torch.linspace(first_bin, last_bin, self.num_bins - 1)
        self.register_buffer("boundaries", boundaries, persistent=False)

        self.norm_z = LayerNorm(self.channel_z)
        self.pool_score = Linear(self.channel_z, 1)
        self.pool_value = Linear(self.channel_z, self.channel_z)
        self.transition = nn.Sequential(
            LayerNorm(self.channel_z),
            Linear(self.channel_z, 4 * self.channel_z),
            nn.GELU(),
            Linear(4 * self.channel_z, self.channel_z),
        )
        self.out = Linear(self.channel_z, self.num_bins, init="final")
        if not self.enabled:
            for param in self.parameters():
                param.requires_grad_(False)

    def _zero_active_param_sum(self, ref: torch.Tensor) -> torch.Tensor:
        zero = ref.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                zero = zero + param.to(dtype=ref.dtype).sum() * 0.0
        return zero

    def forward(self, f_input: FoldingInput, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if not f_input.is_batched:
            raise ValueError("PatchPairGeometryHead expects batched FoldingInput.")

        per_batch_logits: list[torch.Tensor] = []
        per_batch_targets: list[torch.Tensor] = []
        per_batch_weights: list[torch.Tensor] = []
        per_batch_hard_negative: list[torch.Tensor] = []

        for b in range(z.shape[0]):
            patches, patch_pairs = self._build_patch_pairs(f_input, b)
            logits = [
                self._pool_patch_pair(z[b], patches[p["patch_i"]], patches[p["patch_j"]])
                for p in patch_pairs
            ]
            if logits:
                logits_b = torch.stack(logits, dim=0)
                targets_b = torch.stack([p["target"] for p in patch_pairs], dim=0)
                weights_b = z.new_tensor([p["weight"] for p in patch_pairs])
                hard_b = torch.tensor(
                    [p["hard_negative"] for p in patch_pairs],
                    device=z.device,
                    dtype=torch.bool,
                )
            else:
                logits_b = z.new_zeros((0, self.num_bins))
                targets_b = torch.zeros((0,), device=z.device, dtype=torch.long)
                weights_b = z.new_zeros((0,))
                hard_b = torch.zeros((0,), device=z.device, dtype=torch.bool)
            per_batch_logits.append(logits_b)
            per_batch_targets.append(targets_b)
            per_batch_weights.append(weights_b)
            per_batch_hard_negative.append(hard_b)

        max_pairs = max((x.shape[0] for x in per_batch_logits), default=0)
        if max_pairs == 0:
            max_pairs = 1
        logits_out = z.new_zeros((z.shape[0], max_pairs, self.num_bins))
        targets_out = torch.zeros(
            (z.shape[0], max_pairs),
            device=z.device,
            dtype=torch.long,
        )
        weights_out = z.new_zeros((z.shape[0], max_pairs))
        hard_out = torch.zeros(
            (z.shape[0], max_pairs),
            device=z.device,
            dtype=torch.bool,
        )
        valid_out = torch.zeros(
            (z.shape[0], max_pairs),
            device=z.device,
            dtype=torch.bool,
        )
        logits_out = logits_out + self._zero_active_param_sum(z)

        for b, logits_b in enumerate(per_batch_logits):
            n = logits_b.shape[0]
            if n == 0:
                continue
            logits_out[b, :n] = logits_b
            targets_out[b, :n] = per_batch_targets[b]
            weights_out[b, :n] = per_batch_weights[b]
            hard_out[b, :n] = per_batch_hard_negative[b]
            valid_out[b, :n] = True

        return {
            "logits": logits_out,
            "target": targets_out,
            "weight": weights_out,
            "hard_negative": hard_out,
            "valid_mask": valid_out,
        }

    def _build_patch_pairs(
        self,
        f_input: FoldingInput,
        batch_idx: int,
    ) -> tuple[list[_Patch], list[_PatchPair]]:
        with torch.no_grad():
            coords = f_input.token.repr_coords[batch_idx]
            repr_mask = f_input.token.repr_mask[batch_idx]
            token_mask = f_input.token.pad_mask[batch_idx]
            finite_mask = torch.isfinite(coords).all(dim=-1)
            valid = token_mask & repr_mask & finite_mask
            asym_id = f_input.token.asym_id[batch_idx]
            crop_mode = self._crop_mode(f_input, batch_idx)

            chain_ids = sorted(int(x.item()) for x in asym_id[valid].unique())
            if len(chain_ids) < 2:
                return [], []

            interface_mask = self._interface_token_mask(coords, asym_id, valid)
            patches: list[_Patch] = []
            for chain_id in chain_ids:
                idx = torch.where(valid & (asym_id == chain_id))[0]
                if idx.numel() == 0:
                    continue
                patches.extend(
                    self._make_chain_patches(
                        f_input=f_input,
                        batch_idx=batch_idx,
                        chain_id=chain_id,
                        idx=idx,
                        crop_mode=crop_mode,
                        interface_mask=interface_mask,
                    )
                )

            patch_pairs = self._select_patch_pairs(patches, coords, crop_mode)
        return patches, patch_pairs

    def _crop_mode(self, f_input: FoldingInput, batch_idx: int) -> int:
        crop_mode = getattr(f_input, "crop_mode", None)
        if crop_mode is None:
            return CROP_UNKNOWN
        if crop_mode.ndim == 0:
            return int(crop_mode.item())
        return int(crop_mode[batch_idx].item())

    def _interface_token_mask(
        self,
        coords: torch.Tensor,
        asym_id: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        interface = torch.zeros_like(valid)
        idx = torch.where(valid)[0]
        if idx.numel() < 2:
            return interface
        d = torch.cdist(coords[idx], coords[idx])
        inter_chain = asym_id[idx, None] != asym_id[idx][None, :]
        near = (d < self.interface_cutoff) & inter_chain
        interface[idx] = near.any(dim=-1)
        return interface

    def _make_chain_patches(
        self,
        f_input: FoldingInput,
        batch_idx: int,
        chain_id: int,
        idx: torch.Tensor,
        crop_mode: int,
        interface_mask: torch.Tensor,
    ) -> list[_Patch]:
        if crop_mode == CROP_CONTIGUOUS:
            return self._contiguous_patches(f_input, batch_idx, chain_id, idx)
        if crop_mode == CROP_SPATIAL_INTERFACE:
            return self._interface_patches(
                f_input,
                batch_idx,
                chain_id,
                idx,
                interface_mask,
            )
        return self._spatial_patches(
            f_input.token.repr_coords[batch_idx],
            chain_id,
            idx,
            is_interface=False,
            is_background=False,
        )

    def _contiguous_patches(
        self,
        f_input: FoldingInput,
        batch_idx: int,
        chain_id: int,
        idx: torch.Tensor,
    ) -> list[_Patch]:
        org_index = f_input.token.org_token_index[batch_idx, idx]
        idx = idx[torch.argsort(org_index)]
        n_patches = min(
            self.max_patches_per_chain,
            max(1, math.ceil(idx.numel() / self.patch_size)),
        )
        return [
            {
                "idx": chunk,
                "chain_id": chain_id,
                "is_interface": False,
                "is_background": False,
            }
            for chunk in torch.tensor_split(idx, n_patches)
            if chunk.numel() > 0
        ]

    def _spatial_patches(
        self,
        coords: torch.Tensor,
        chain_id: int,
        idx: torch.Tensor,
        is_interface: bool,
        is_background: bool,
        max_patches: int | None = None,
    ) -> list[_Patch]:
        max_patches = self.max_patches_per_chain if max_patches is None else max_patches
        n_patches = min(
            max_patches,
            max(1, math.ceil(idx.numel() / self.patch_size)),
        )
        clusters = self._fps_clusters(coords[idx], idx, n_patches)
        return [
            {
                "idx": cluster,
                "chain_id": chain_id,
                "is_interface": is_interface,
                "is_background": is_background,
            }
            for cluster in clusters
            if cluster.numel() > 0
        ]

    def _interface_patches(
        self,
        f_input: FoldingInput,
        batch_idx: int,
        chain_id: int,
        idx: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> list[_Patch]:
        coords = f_input.token.repr_coords[batch_idx]
        iface_idx = idx[interface_mask[idx]]
        bg_idx = idx[~interface_mask[idx]]
        if iface_idx.numel() == 0:
            return self._spatial_patches(
                coords,
                chain_id,
                idx,
                is_interface=False,
                is_background=False,
            )

        iface_budget = max(1, self.max_patches_per_chain - int(bg_idx.numel() > 0))
        patches = self._spatial_patches(
            coords,
            chain_id,
            iface_idx,
            is_interface=True,
            is_background=False,
            max_patches=iface_budget,
        )

        if bg_idx.numel() > 0 and len(patches) < self.max_patches_per_chain:
            patches.append(
                {
                    "idx": bg_idx,
                    "chain_id": chain_id,
                    "is_interface": False,
                    "is_background": True,
                }
            )
        return patches

    def _fps_clusters(
        self,
        coords_local: torch.Tensor,
        idx_global: torch.Tensor,
        n_clusters: int,
    ) -> list[torch.Tensor]:
        if n_clusters <= 1 or idx_global.numel() <= 1:
            return [idx_global]

        n_clusters = min(n_clusters, idx_global.numel())
        centers = [0]
        min_dist = torch.full(
            (idx_global.numel(),),
            float("inf"),
            device=coords_local.device,
            dtype=coords_local.dtype,
        )
        for _ in range(1, n_clusters):
            last = coords_local[centers[-1]].unsqueeze(0)
            dist = torch.cdist(coords_local, last).squeeze(-1)
            min_dist = torch.minimum(min_dist, dist)
            centers.append(int(torch.argmax(min_dist).item()))

        center_coords = coords_local[centers]
        assign = torch.cdist(coords_local, center_coords).argmin(dim=-1)
        clusters = []
        for cluster_i in range(n_clusters):
            cluster = idx_global[assign == cluster_i]
            if cluster.numel() > 0:
                clusters.append(cluster)
        return clusters

    def _select_patch_pairs(
        self,
        patches: list[_Patch],
        coords: torch.Tensor,
        crop_mode: int,
    ) -> list[_PatchPair]:
        pairs: list[_PatchPair] = []
        for i, patch_i in enumerate(patches):
            for j, patch_j in enumerate(patches[i + 1 :], start=i + 1):
                if patch_i["chain_id"] == patch_j["chain_id"]:
                    continue

                coords_i = coords[patch_i["idx"]]
                coords_j = coords[patch_j["idx"]]
                com_i = coords_i.mean(dim=0)
                com_j = coords_j.mean(dim=0)
                com_dist = torch.linalg.norm(com_i - com_j)
                token_min_dist = torch.cdist(coords_i, coords_j).min()
                positive = bool((token_min_dist < self.positive_cutoff).item())
                hard_negative = bool((token_min_dist > self.hard_negative_cutoff).item())
                weight = self._patch_pair_weight(
                    crop_mode,
                    patch_i,
                    patch_j,
                    positive=positive,
                    hard_negative=hard_negative,
                )
                if weight <= 0:
                    continue
                target = (com_dist.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
                priority = self._patch_pair_priority(
                    crop_mode,
                    positive=positive,
                    hard_negative=hard_negative,
                    distance=float(com_dist.item()),
                )
                pairs.append(
                    {
                        "patch_i": i,
                        "patch_j": j,
                        "target": target,
                        "weight": weight,
                        "hard_negative": hard_negative,
                        "priority": priority,
                    }
                )

        pairs.sort(key=lambda x: x["priority"], reverse=True)
        return pairs[: self.max_patch_pairs]

    def _patch_pair_weight(
        self,
        crop_mode: int,
        patch_i: _Patch,
        patch_j: _Patch,
        positive: bool,
        hard_negative: bool,
    ) -> float:
        any_background = patch_i["is_background"] or patch_j["is_background"]
        both_interface = patch_i["is_interface"] and patch_j["is_interface"]

        if crop_mode == CROP_CONTIGUOUS:
            if hard_negative:
                return 2.0
            return 1.0 if positive else 0.5

        if crop_mode == CROP_SPATIAL_INTERFACE:
            if any_background:
                return 0.1
            if positive and both_interface:
                return 1.5
            if hard_negative and both_interface:
                return 1.0
            return 1.0 if positive else 0.25

        if any_background:
            return 0.1
        if hard_negative:
            return 0.5
        return 1.0 if positive else 0.25

    def _patch_pair_priority(
        self,
        crop_mode: int,
        positive: bool,
        hard_negative: bool,
        distance: float,
    ) -> float:
        if crop_mode == CROP_CONTIGUOUS and hard_negative:
            return 4.0 + min(distance, 100.0) / 1000.0
        if positive:
            return 3.0 - min(distance, 100.0) / 1000.0
        if hard_negative:
            return 2.0 + min(distance, 100.0) / 1000.0
        return 1.0

    def _pool_patch_pair(
        self,
        z: torch.Tensor,
        patch_i: _Patch,
        patch_j: _Patch,
    ) -> torch.Tensor:
        idx_i = patch_i["idx"]
        idx_j = patch_j["idx"]
        z_ij = z[idx_i][:, idx_j]
        z_ji = z[idx_j][:, idx_i].transpose(0, 1)
        pair_z = 0.5 * (z_ij + z_ji)
        flat_z = self.norm_z(pair_z.reshape(-1, self.channel_z))
        score = self.pool_score(flat_z).squeeze(-1)
        alpha = score.float().softmax(dim=-1).to(flat_z.dtype)
        value = self.pool_value(flat_z)
        pooled = torch.einsum("n,nc->c", alpha, value)
        pooled = pooled + self.transition(pooled)
        return self.out(pooled)
