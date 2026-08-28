import math
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LayerNorm, Linear
from kfold.utils.config import configurable


class _Patch(TypedDict):
    idx: torch.Tensor
    pool_idx: torch.Tensor
    chain_id: torch.Tensor


class _PatchPairBatch(TypedDict):
    idx_i: torch.Tensor
    idx_j: torch.Tensor
    mask_i: torch.Tensor
    mask_j: torch.Tensor
    target: torch.Tensor
    contact_strength: torch.Tensor
    rank_group: torch.Tensor


@configurable
class PatchPairGeometryHead(nn.Module):
    @dataclass(kw_only=True)
    class Config:
        enabled: bool = False
        channel_z: int = 256
        num_bins: int = 64
        min_dist: float = 2.0
        max_dist: float = 22.0
        patch_size: int = 16
        max_patches_per_chain: int = 8
        max_patch_pairs: int = 256
        pool_max_tokens_per_patch: int = 0
        ranking_contact_cutoff: float = 8.0

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
        self.pool_max_tokens_per_patch: int = int(
            getattr(cfg, "pool_max_tokens_per_patch", 0)
        )
        self.ranking_contact_cutoff: float = float(cfg.ranking_contact_cutoff)
        if self.pool_max_tokens_per_patch < 0:
            raise ValueError("pool_max_tokens_per_patch must be non-negative.")
        if self.ranking_contact_cutoff <= 0:
            raise ValueError("ranking_contact_cutoff must be positive.")

        bin_size = (self.max_dist - self.min_dist) / self.num_bins
        first_bin = self.min_dist + bin_size
        last_bin = self.max_dist - bin_size
        bin_boundaries = torch.linspace(first_bin, last_bin, self.num_bins - 1)
        self.register_buffer("bin_boundaries", bin_boundaries, persistent=False)

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
        self.rank_out = Linear(self.channel_z, 1, init="final")
        if not self.enabled:
            for param in self.parameters():
                param.requires_grad_(False)

    def _zero_active_param_sum(self, ref: torch.Tensor) -> torch.Tensor:
        zero = ref.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                zero = zero + param.reshape(-1)[0].to(dtype=ref.dtype) * 0.0
        return zero

    def forward(
        self,
        f_input: FoldingInput,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if not f_input.is_batched:
            raise ValueError("PatchPairGeometryHead expects batched FoldingInput.")

        per_batch_logits: list[torch.Tensor] = []
        per_batch_rank_scores: list[torch.Tensor] = []
        per_batch_targets: list[torch.Tensor] = []
        per_batch_contact_strengths: list[torch.Tensor] = []
        per_batch_rank_groups: list[torch.Tensor] = []
        rank_group_offset = 0

        for b in range(z.shape[0]):
            patch_pairs = self._build_patch_pairs(f_input, b)
            if patch_pairs is None:
                continue
            logits, rank_score = self._pool_patch_pairs(z[b], patch_pairs)
            per_batch_logits.append(logits)
            per_batch_rank_scores.append(rank_score)
            per_batch_targets.append(patch_pairs["target"])
            per_batch_contact_strengths.append(patch_pairs["contact_strength"])
            local_rank_group = patch_pairs["rank_group"]
            per_batch_rank_groups.append(local_rank_group + rank_group_offset)
            if local_rank_group.numel() > 0:
                rank_group_offset += int(local_rank_group.max().item()) + 1

        if per_batch_logits:
            logits = torch.cat(per_batch_logits)
            rank_score = torch.cat(per_batch_rank_scores)
            target = torch.cat(per_batch_targets)
            contact_strength = torch.cat(per_batch_contact_strengths)
            rank_group = torch.cat(per_batch_rank_groups)
        else:
            logits = z.new_zeros((0, self.num_bins)) + self._zero_active_param_sum(z)
            target = torch.zeros((0,), device=z.device, dtype=torch.long)
            contact_strength = z.new_zeros((0,))
            rank_group = torch.zeros((0,), device=z.device, dtype=torch.long)
            rank_score = z.new_zeros((0,)) + self._zero_active_param_sum(z)

        return {
            "logits": logits,
            "rank_score": rank_score,
            "bin_boundaries": self.bin_boundaries,
            "target": target,
            "contact_strength": contact_strength,
            "rank_group": rank_group,
        }

    def _build_patch_pairs(
        self,
        f_input: FoldingInput,
        batch_idx: int,
    ) -> _PatchPairBatch | None:
        with torch.no_grad():
            coords = f_input.token.repr_coords[batch_idx]
            repr_mask = f_input.token.repr_mask[batch_idx]
            token_mask = f_input.token.pad_mask[batch_idx]
            finite_mask = torch.isfinite(coords).all(dim=-1)
            valid = token_mask & repr_mask & finite_mask
            asym_id = f_input.token.asym_id[batch_idx]
            chain_ids = asym_id[valid].unique(sorted=True)
            if chain_ids.numel() < 2:
                return None

            patches: list[_Patch] = []
            for chain_id in chain_ids.unbind():
                idx = torch.where(valid & (asym_id == chain_id))[0]
                patches.extend(
                    self._spatial_patches(
                        coords=coords,
                        chain_id=chain_id,
                        idx=idx,
                    )
                )

            patch_pairs = self._select_patch_pairs(
                patches=patches,
                coords=coords,
            )
        return patch_pairs

    def _spatial_patches(
        self,
        coords: torch.Tensor,
        chain_id: torch.Tensor,
        idx: torch.Tensor,
    ) -> list[_Patch]:
        n_patches = min(
            self.max_patches_per_chain,
            max(1, math.ceil(idx.numel() / self.patch_size)),
        )
        clusters = self._fps_clusters(coords[idx], idx, n_patches)
        return [self._make_patch(idx=cluster, chain_id=chain_id) for cluster in clusters]

    def _make_patch(
        self,
        idx: torch.Tensor,
        chain_id: torch.Tensor,
    ) -> _Patch:
        return {
            "idx": idx,
            "pool_idx": self._pool_limited_idx(idx),
            "chain_id": chain_id,
        }

    def _pool_limited_idx(self, idx: torch.Tensor) -> torch.Tensor:
        max_tokens = self.pool_max_tokens_per_patch
        if max_tokens <= 0 or idx.numel() <= max_tokens:
            return idx
        take = torch.linspace(
            0,
            idx.numel() - 1,
            steps=max_tokens,
            device=idx.device,
        ).round()
        return idx[take.long()]

    def _fps_clusters(
        self,
        coords_local: torch.Tensor,
        idx_global: torch.Tensor,
        n_clusters: int,
    ) -> list[torch.Tensor]:
        if n_clusters <= 1 or idx_global.numel() <= 1:
            return [idx_global]

        n_clusters = min(n_clusters, idx_global.numel())
        centers = [torch.zeros((), device=coords_local.device, dtype=torch.long)]
        min_dist = torch.full(
            (idx_global.numel(),),
            float("inf"),
            device=coords_local.device,
            dtype=coords_local.dtype,
        )
        for _ in range(1, n_clusters):
            last = coords_local[centers[-1]].unsqueeze(0)
            diff = coords_local - last
            dist = (diff * diff).sum(dim=-1)
            min_dist = torch.minimum(min_dist, dist)
            centers.append(torch.argmax(min_dist))

        center_coords = coords_local[torch.stack(centers)]
        diff = coords_local[:, None, :] - center_coords[None, :, :]
        assign = (diff * diff).sum(dim=-1).argmin(dim=-1)
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
    ) -> _PatchPairBatch:
        n_patches = len(patches)
        device = coords.device
        chain_id = torch.stack([p["chain_id"] for p in patches])
        com = torch.stack([coords[p["idx"]].float().mean(dim=0) for p in patches])

        pool_idx = nn.utils.rnn.pad_sequence(
            [p["pool_idx"] for p in patches],
            batch_first=True,
        )
        pool_lengths = torch.tensor(
            [p["pool_idx"].numel() for p in patches],
            device=device,
        )
        pool_mask = torch.arange(pool_idx.shape[1], device=device) < pool_lengths[:, None]

        patch_i, patch_j = torch.triu_indices(
            n_patches,
            n_patches,
            offset=1,
            device=device,
        )
        inter_chain = chain_id[patch_i] != chain_id[patch_j]
        patch_i = patch_i[inter_chain]
        patch_j = patch_j[inter_chain]

        com_diff = com[patch_i] - com[patch_j]
        com_dist = (com_diff * com_diff).sum(dim=-1).sqrt()
        target = (com_dist[:, None] > self.bin_boundaries).sum(dim=-1).long()

        # Uniform supervision: every valid inter-chain patch pair is sampled
        # with equal probability. Contact density is only a ranking target;
        # it does not change pair selection or CE weighting.
        n_select = min(self.max_patch_pairs, patch_i.numel())
        selected = torch.randperm(patch_i.numel(), device=device)[:n_select]
        selected_i = patch_i[selected]
        selected_j = patch_j[selected]
        contact_strength = torch.stack(
            [
                self._contact_strength(coords[p_i["idx"]], coords[p_j["idx"]])
                for p_i, p_j in zip(
                    [patches[idx] for idx in selected_i.tolist()],
                    [patches[idx] for idx in selected_j.tolist()],
                    strict=True,
                )
            ]
        )
        selected_chain_i = chain_id[selected_i]
        selected_chain_j = chain_id[selected_j]
        chain_pair = torch.stack(
            (
                torch.minimum(selected_chain_i, selected_chain_j),
                torch.maximum(selected_chain_i, selected_chain_j),
            ),
            dim=-1,
        )
        _, rank_group = torch.unique(chain_pair, dim=0, return_inverse=True)

        return {
            "idx_i": pool_idx[selected_i],
            "idx_j": pool_idx[selected_j],
            "mask_i": pool_mask[selected_i],
            "mask_j": pool_mask[selected_j],
            "target": target[selected],
            "contact_strength": contact_strength,
            "rank_group": rank_group,
        }

    def _contact_strength(
        self,
        coords_i: torch.Tensor,
        coords_j: torch.Tensor,
    ) -> torch.Tensor:
        distance_squared = (coords_i[:, None] - coords_j[None, :]).square().sum(dim=-1)
        return (distance_squared < self.ranking_contact_cutoff**2).float().mean()

    def _pool_patch_pairs(
        self,
        z: torch.Tensor,
        patch_pairs: _PatchPairBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_pairs = patch_pairs["target"].numel()
        if n_pairs == 0:
            return z.new_zeros((0, self.num_bins)), z.new_zeros((0,))

        z_flat = z.reshape(-1, self.channel_z)
        seq_len = z.shape[0]
        idx_i = patch_pairs["idx_i"]
        idx_j = patch_pairs["idx_j"]
        flat_idx_ij = (idx_i[:, :, None] * seq_len + idx_j[:, None, :]).reshape(-1)
        flat_idx_ji = (idx_j[:, None, :] * seq_len + idx_i[:, :, None]).reshape(-1)
        directional_z = z_flat.index_select(
            0,
            torch.cat((flat_idx_ij, flat_idx_ji)),
        ).view(
            2,
            n_pairs,
            idx_i.shape[1],
            idx_j.shape[1],
            self.channel_z,
        )
        pair_z = directional_z[0] + directional_z[1]
        flat_z = self.norm_z(pair_z.reshape(n_pairs, -1, self.channel_z))

        pair_mask = patch_pairs["mask_i"][:, :, None] & patch_pairs["mask_j"][:, None, :]
        flat_mask = pair_mask.reshape(n_pairs, -1)
        score = self.pool_score(flat_z).squeeze(-1).float()
        score = score.masked_fill(~flat_mask, torch.finfo(score.dtype).min)
        alpha = score.softmax(dim=-1).to(flat_z.dtype)
        pooled_z = torch.einsum("pn,pnc->pc", alpha, flat_z)
        pooled = self.pool_value(pooled_z)
        pooled = pooled + self.transition(pooled)
        return self.out(pooled), self.rank_out(pooled).squeeze(-1)
