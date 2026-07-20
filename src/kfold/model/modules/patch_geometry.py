import math
import time
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LayerNorm, Linear
from kfold.utils.config import configurable

CROP_UNKNOWN = 0
CROP_CONTIGUOUS = 1
CROP_SPATIAL = 2
CROP_SPATIAL_INTERFACE = 3


class _Patch(TypedDict):
    idx: torch.Tensor
    pool_idx: torch.Tensor
    chain_id: int
    is_interface: bool
    is_background: bool


class _PatchPair(TypedDict):
    patch_i: int
    patch_j: int
    target: torch.Tensor
    weight: torch.Tensor
    hard_negative: torch.Tensor


class _PatchPairBatch(TypedDict):
    idx_i: torch.Tensor
    idx_j: torch.Tensor
    mask_i: torch.Tensor
    mask_j: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    hard_negative: torch.Tensor


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
        pool_chunk_size: int = 32
        pool_max_tokens_per_patch: int = 0
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
        self.pool_chunk_size: int = int(getattr(cfg, "pool_chunk_size", 32))
        self.pool_max_tokens_per_patch: int = int(
            getattr(cfg, "pool_max_tokens_per_patch", 0)
        )
        self.interface_cutoff: float = float(cfg.interface_cutoff)
        self.positive_cutoff: float = float(cfg.positive_cutoff)
        self.hard_negative_cutoff: float = float(cfg.hard_negative_cutoff)
        if self.pool_max_tokens_per_patch < 0:
            raise ValueError("pool_max_tokens_per_patch must be non-negative.")

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

    def _timing_start(self, ref: torch.Tensor) -> object:
        if ref.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _timing_stop(self, start: object, ref: torch.Tensor) -> tuple[object, object]:
        if ref.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return start, event
        return start, time.perf_counter()

    def _timing_sum_ms(self, records: list[tuple[object, object]]) -> float:
        total = 0.0
        for start, end in records:
            if isinstance(start, torch.cuda.Event) and isinstance(
                end,
                torch.cuda.Event,
            ):
                total += float(start.elapsed_time(end))
            else:
                total += 1000.0 * (float(end) - float(start))
        return total

    def forward(self, f_input: FoldingInput, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if not f_input.is_batched:
            raise ValueError("PatchPairGeometryHead expects batched FoldingInput.")

        total_timing = self._timing_start(z)
        timing_records: dict[str, list[tuple[object, object]]] = {
            "build_ms": [],
            "pool_ms": [],
            "pack_ms": [],
        }
        per_batch_logits: list[torch.Tensor] = []
        per_batch_targets: list[torch.Tensor] = []
        per_batch_weights: list[torch.Tensor] = []
        per_batch_hard_negative: list[torch.Tensor] = []

        for b in range(z.shape[0]):
            build_timing = self._timing_start(z)
            patch_pairs = self._build_patch_pairs(f_input, b)
            timing_records["build_ms"].append(self._timing_stop(build_timing, z))
            if patch_pairs is not None and patch_pairs["target"].numel() > 0:
                pool_timing = self._timing_start(z)
                logits_b = self._pool_patch_pairs(z[b], patch_pairs)
                timing_records["pool_ms"].append(self._timing_stop(pool_timing, z))
                targets_b = patch_pairs["target"]
                weights_b = patch_pairs["weight"].to(device=z.device, dtype=z.dtype)
                hard_b = patch_pairs["hard_negative"].to(
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

        pack_timing = self._timing_start(z)
        for b, logits_b in enumerate(per_batch_logits):
            n = logits_b.shape[0]
            if n == 0:
                continue
            logits_out[b, :n] = logits_b
            targets_out[b, :n] = per_batch_targets[b]
            weights_out[b, :n] = per_batch_weights[b]
            hard_out[b, :n] = per_batch_hard_negative[b]
            valid_out[b, :n] = True
        timing_records["pack_ms"].append(self._timing_stop(pack_timing, z))
        total_record = self._timing_stop(total_timing, z)
        timing_records["total_ms"] = [total_record]
        if z.is_cuda:
            total_record[1].synchronize()

        return {
            "logits": logits_out,
            "target": targets_out,
            "weight": weights_out,
            "hard_negative": hard_out,
            "valid_mask": valid_out,
            "timing": {
                name: z.new_tensor(self._timing_sum_ms(records))
                for name, records in timing_records.items()
            },
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
            crop_mode = self._crop_mode(f_input, batch_idx)

            chain_ids = sorted(
                int(x) for x in asym_id[valid].unique().detach().cpu().tolist()
            )
            if len(chain_ids) < 2:
                return None

            token_dist2 = self._token_squared_dist(coords)
            if crop_mode == CROP_SPATIAL_INTERFACE:
                interface_mask = self._interface_token_mask(
                    asym_id,
                    valid,
                    token_dist2,
                )
            else:
                interface_mask = torch.zeros_like(valid)
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

            patch_pairs = self._select_patch_pairs(
                patches,
                coords,
                crop_mode,
                token_dist2,
            )
        return patch_pairs

    def _token_squared_dist(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        diff = coords[:, None, :] - coords[None, :, :]
        return (diff * diff).sum(dim=-1)

    def _crop_mode(self, f_input: FoldingInput, batch_idx: int) -> int:
        crop_mode = getattr(f_input, "crop_mode", None)
        if crop_mode is None:
            return CROP_UNKNOWN
        if crop_mode.ndim == 0:
            return int(crop_mode.detach().cpu().tolist())
        return int(crop_mode[batch_idx].detach().cpu().tolist())

    def _interface_token_mask(
        self,
        asym_id: torch.Tensor,
        valid: torch.Tensor,
        token_dist2: torch.Tensor,
    ) -> torch.Tensor:
        interface = torch.zeros_like(valid)
        idx = torch.where(valid)[0]
        if idx.numel() < 2:
            return interface
        d2 = token_dist2[idx][:, idx]
        inter_chain = asym_id[idx, None] != asym_id[idx][None, :]
        near = (d2 < self.interface_cutoff**2) & inter_chain
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
            self._make_patch(
                idx=chunk,
                chain_id=chain_id,
                is_interface=False,
                is_background=False,
            )
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
            self._make_patch(
                idx=cluster,
                chain_id=chain_id,
                is_interface=is_interface,
                is_background=is_background,
            )
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
                self._make_patch(
                    idx=bg_idx,
                    chain_id=chain_id,
                    is_interface=False,
                    is_background=True,
                )
            )
        return patches

    def _make_patch(
        self,
        idx: torch.Tensor,
        chain_id: int,
        is_interface: bool,
        is_background: bool,
    ) -> _Patch:
        return {
            "idx": idx,
            "pool_idx": self._pool_limited_idx(idx),
            "chain_id": chain_id,
            "is_interface": is_interface,
            "is_background": is_background,
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
        crop_mode: int,
        token_dist2: torch.Tensor,
    ) -> _PatchPairBatch | None:
        n_patches = len(patches)
        if n_patches < 2:
            return None

        device = coords.device
        chain_id = torch.tensor(
            [p["chain_id"] for p in patches],
            device=device,
            dtype=torch.long,
        )
        is_interface = torch.tensor(
            [p["is_interface"] for p in patches],
            device=device,
            dtype=torch.bool,
        )
        is_background = torch.tensor(
            [p["is_background"] for p in patches],
            device=device,
            dtype=torch.bool,
        )
        com = torch.stack([coords[p["idx"]].float().mean(dim=0) for p in patches])

        token_patch = torch.full(
            (coords.shape[0],),
            -1,
            device=device,
            dtype=torch.long,
        )
        for patch_idx, patch in enumerate(patches):
            token_patch[patch["idx"]] = patch_idx

        token_patch_i = token_patch[:, None]
        token_patch_j = token_patch[None, :]
        token_pair_valid = ((token_patch_i >= 0) & (token_patch_j >= 0)).reshape(-1)
        token_pair_patch = (token_patch_i * n_patches + token_patch_j).reshape(-1)
        patch_min_dist2 = torch.full(
            (n_patches * n_patches,),
            float("inf"),
            device=device,
            dtype=token_dist2.dtype,
        )
        patch_min_dist2.scatter_reduce_(
            0,
            token_pair_patch[token_pair_valid],
            token_dist2.reshape(-1)[token_pair_valid],
            reduce="amin",
            include_self=True,
        )
        patch_min_dist2 = patch_min_dist2.view(n_patches, n_patches)

        patch_i, patch_j = torch.triu_indices(
            n_patches,
            n_patches,
            offset=1,
            device=device,
        )
        inter_chain = chain_id[patch_i] != chain_id[patch_j]
        if not inter_chain.any():
            return None

        patch_i = patch_i[inter_chain]
        patch_j = patch_j[inter_chain]
        token_min_dist2 = patch_min_dist2[patch_i, patch_j]
        positive = token_min_dist2 < self.positive_cutoff**2
        hard_negative = token_min_dist2 > self.hard_negative_cutoff**2

        com_diff = com[patch_i] - com[patch_j]
        com_dist = (com_diff * com_diff).sum(dim=-1).sqrt()
        target = (com_dist[:, None] > self.boundaries).sum(dim=-1).long()

        any_background = is_background[patch_i] | is_background[patch_j]
        both_interface = is_interface[patch_i] & is_interface[patch_j]
        weight = self._patch_pair_weight_tensor(
            crop_mode,
            positive=positive,
            hard_negative=hard_negative,
            any_background=any_background,
            both_interface=both_interface,
            dtype=coords.dtype,
        )
        keep = weight > 0
        if not keep.any():
            return None

        patch_i = patch_i[keep]
        patch_j = patch_j[keep]
        target = target[keep]
        weight = weight[keep]
        hard_negative = hard_negative[keep]
        positive = positive[keep]
        com_dist = com_dist[keep]

        priority = self._patch_pair_priority_tensor(
            crop_mode,
            positive=positive,
            hard_negative=hard_negative,
            distance=com_dist,
        )
        n_select = min(self.max_patch_pairs, priority.numel())
        selected = torch.argsort(priority, descending=True, stable=True)[:n_select]
        selected_i = patch_i[selected].detach().cpu().tolist()
        selected_j = patch_j[selected].detach().cpu().tolist()
        idx_i_list = [patches[i]["pool_idx"] for i in selected_i]
        idx_j_list = [patches[j]["pool_idx"] for j in selected_j]
        max_i = max(idx.numel() for idx in idx_i_list)
        max_j = max(idx.numel() for idx in idx_j_list)

        idx_i = torch.zeros(
            (n_select, max_i),
            device=device,
            dtype=torch.long,
        )
        idx_j = torch.zeros(
            (n_select, max_j),
            device=device,
            dtype=torch.long,
        )
        mask_i = torch.zeros(
            (n_select, max_i),
            device=device,
            dtype=torch.bool,
        )
        mask_j = torch.zeros(
            (n_select, max_j),
            device=device,
            dtype=torch.bool,
        )
        for pair_idx, (patch_i_idx, patch_j_idx) in enumerate(
            zip(idx_i_list, idx_j_list, strict=True)
        ):
            n_i = patch_i_idx.numel()
            n_j = patch_j_idx.numel()
            idx_i[pair_idx, :n_i] = patch_i_idx
            idx_j[pair_idx, :n_j] = patch_j_idx
            mask_i[pair_idx, :n_i] = True
            mask_j[pair_idx, :n_j] = True

        return {
            "idx_i": idx_i,
            "idx_j": idx_j,
            "mask_i": mask_i,
            "mask_j": mask_j,
            "target": target[selected],
            "weight": weight[selected],
            "hard_negative": hard_negative[selected],
        }

    def _patch_pair_weight_tensor(
        self,
        crop_mode: int,
        positive: torch.Tensor,
        hard_negative: torch.Tensor,
        any_background: torch.Tensor,
        both_interface: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        one = positive.new_tensor(1.0, dtype=dtype)
        if crop_mode == CROP_CONTIGUOUS:
            return torch.where(
                hard_negative,
                positive.new_tensor(2.0, dtype=dtype),
                torch.where(positive, one, positive.new_tensor(0.5, dtype=dtype)),
            )

        if crop_mode == CROP_SPATIAL_INTERFACE:
            return torch.where(
                any_background,
                positive.new_tensor(0.1, dtype=dtype),
                torch.where(
                    positive & both_interface,
                    positive.new_tensor(1.5, dtype=dtype),
                    torch.where(
                        hard_negative & both_interface,
                        one,
                        torch.where(
                            positive,
                            one,
                            positive.new_tensor(0.25, dtype=dtype),
                        ),
                    ),
                ),
            )

        return torch.where(
            any_background,
            positive.new_tensor(0.1, dtype=dtype),
            torch.where(
                hard_negative,
                positive.new_tensor(0.5, dtype=dtype),
                torch.where(positive, one, positive.new_tensor(0.25, dtype=dtype)),
            ),
        )

    def _patch_pair_priority_tensor(
        self,
        crop_mode: int,
        positive: torch.Tensor,
        hard_negative: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        dist_term = torch.minimum(distance, distance.new_tensor(100.0)) / 1000.0
        if crop_mode == CROP_CONTIGUOUS:
            return torch.where(
                hard_negative,
                4.0 + dist_term,
                torch.where(positive, 3.0 - dist_term, distance.new_ones(())),
            )
        return torch.where(
            positive,
            3.0 - dist_term,
            torch.where(hard_negative, 2.0 + dist_term, distance.new_ones(())),
        )

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

    def _pool_patch_pairs(
        self,
        z: torch.Tensor,
        patch_pairs: _PatchPairBatch,
    ) -> torch.Tensor:
        chunk_size = max(1, self.pool_chunk_size)
        n_pairs = patch_pairs["target"].numel()
        if n_pairs == 0:
            return z.new_zeros((0, self.num_bins))
        logits = []
        z_flat = z.reshape(-1, self.channel_z)
        seq_len = z.shape[0]
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            logits.append(
                self._pool_patch_pair_chunk(
                    z_flat,
                    seq_len,
                    patch_pairs["idx_i"][start:end],
                    patch_pairs["idx_j"][start:end],
                    patch_pairs["mask_i"][start:end],
                    patch_pairs["mask_j"][start:end],
                )
            )
        return torch.cat(logits, dim=0)

    def _pool_patch_pair_chunk(
        self,
        z_flat: torch.Tensor,
        seq_len: int,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
    ) -> torch.Tensor:
        n_pairs = idx_i.shape[0]
        flat_idx_ij = (idx_i[:, :, None] * seq_len + idx_j[:, None, :]).reshape(-1)
        flat_idx_ji = (idx_j[:, None, :] * seq_len + idx_i[:, :, None]).reshape(-1)
        z_ij = z_flat.index_select(0, flat_idx_ij).view(
            n_pairs,
            idx_i.shape[1],
            idx_j.shape[1],
            self.channel_z,
        )
        z_ji = z_flat.index_select(0, flat_idx_ji).view_as(z_ij)
        pair_z = 0.5 * (z_ij + z_ji)
        flat_z = self.norm_z(pair_z.reshape(n_pairs, -1, self.channel_z))

        pair_mask = mask_i[:, :, None] & mask_j[:, None, :]
        flat_mask = pair_mask.reshape(n_pairs, -1)
        score = self.pool_score(flat_z).squeeze(-1).float()
        score = score.masked_fill(~flat_mask, torch.finfo(score.dtype).min)
        alpha = score.softmax(dim=-1).to(flat_z.dtype)
        pooled_z = torch.einsum("pn,pnc->pc", alpha, flat_z)
        pooled = self.pool_value(pooled_z)
        pooled = pooled + self.transition(pooled)
        return self.out(pooled)
