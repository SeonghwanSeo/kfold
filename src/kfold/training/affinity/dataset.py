"""Cached frozen-feature dataset and fixed assay-slot collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

import kfold.constants as C

from .cache import AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1, FeatureCacheReader
from .crop import select_pocket_annotation_crop
from .data import AffinityBatchIndex
from .pair_storage import (
    AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    cache_schema_from_payload,
    crop_full_cross_payload,
    full_cross_pl_distogram_profile,
    unpack_cross_only_payload,
)
from .pocket import PocketAnnotationLookup
from .schema import (
    AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
    AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1,
    AFFINITY_CACHE_SCHEMAS,
)


@dataclass(frozen=True, kw_only=True)
class CachedAffinityItem:
    """One cropped cached system paired with its immutable assay label."""

    features: dict[str, torch.Tensor]
    label: float
    assay_key: str
    canonical_smiles: str
    shape_bucket: int | None
    origin: str
    record_id: str
    ranking_group_id: int | None = None


class CachedAffinityFeatureDataset(Dataset[CachedAffinityItem | None]):
    """Map v4 full-cross records to deterministic train-time affinity crops."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        cache_root: str,
        cache_schema: str = AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
        max_crop_tokens: int = 256,
        max_protein_crop_tokens: int = 200,
        crop_mode: str = "legacy_predicted_contact",
        pocket_annotations: PocketAnnotationLookup | None = None,
        pocket_neighborhood_size: int = 10,
        distogram_pocket_distance_cutoff: float = 15.0,
        distogram_use_entropy_tiebreak: bool = True,
        reader: FeatureCacheReader | None = None,
        required_cache_encoding: str | None = None,
    ) -> None:
        self.rows = list(rows)
        self.cache_root = cache_root
        self.cache_schema = cache_schema
        self.required_cache_encoding = required_cache_encoding
        self.max_crop_tokens = max_crop_tokens
        self.max_protein_crop_tokens = max_protein_crop_tokens
        if crop_mode not in {
            "distogram_predicted_contact",
            "distogram_query_window",
            "distogram_target_consensus",
            "distogram_target_consensus_compact",
            "pocket80k_target_compact",
            "pocket80k_query_adaptive_80_20",
            "legacy_predicted_contact",
            "boltz2_pocket",
        }:
            raise ValueError(f"Unsupported affinity crop mode: {crop_mode!r}.")
        if crop_mode == "boltz2_pocket" and pocket_annotations is None:
            raise ValueError("Boltz2 pocket crop requires a pocket annotation manifest.")
        if crop_mode == "distogram_target_consensus" and pocket_annotations is None:
            raise ValueError(
                "Distogram target consensus requires a pocket annotation manifest."
            )
        if pocket_neighborhood_size <= 0:
            raise ValueError("pocket_neighborhood_size must be positive.")
        if cache_schema not in AFFINITY_CACHE_SCHEMAS:
            raise ValueError(
                "Affinity dataset requires a supported physical cache, "
                f"got {cache_schema!r}."
            )
        if (
            crop_mode
            in {
                "distogram_predicted_contact",
                "distogram_query_window",
                "distogram_target_consensus",
                "legacy_predicted_contact",
            }
            and cache_schema != AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1
        ):
            raise ValueError("Predicted-distogram crop requires the v1 physical cache.")
        if (
            crop_mode == "distogram_target_consensus_compact"
            and cache_schema != AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1
        ):
            raise ValueError(
                "Compact target consensus requires its compact cache schema."
            )
        if (
            cache_schema == AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1
            and crop_mode != "distogram_target_consensus_compact"
        ):
            raise ValueError("Legacy compact pocket payloads are already cropped.")
        if (
            cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1
            and crop_mode != "pocket80k_target_compact"
        ):
            raise ValueError("Compact pocket payloads are already cropped.")
        if crop_mode == "pocket80k_target_compact" and cache_schema not in {
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
        }:
            raise ValueError(
                "80k target-consensus crop requires its final compact schema."
            )
        if cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2:
            if crop_mode != "pocket80k_target_compact":
                raise ValueError("Direct compact pocket payloads are already cropped.")
            if required_cache_encoding != AFFINITY_CACHE_ENCODING_ARRAYPACK_RAW_V1:
                raise ValueError(
                    "Direct compact affinity training requires a raw arraypack replica."
                )
        if (
            crop_mode == "pocket80k_query_adaptive_80_20"
            and cache_schema != AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1
        ):
            raise ValueError("80/20 ablation requires its final compact schema.")
        if (
            cache_schema == AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1
            and crop_mode != "pocket80k_query_adaptive_80_20"
        ):
            raise ValueError("80/20 ablation payload is already cropped.")
        self.crop_mode = crop_mode
        self.pocket_annotations = pocket_annotations
        self.pocket_neighborhood_size = pocket_neighborhood_size
        if distogram_pocket_distance_cutoff <= 0:
            raise ValueError("distogram_pocket_distance_cutoff must be positive.")
        self.distogram_pocket_distance_cutoff = distogram_pocket_distance_cutoff
        self.distogram_use_entropy_tiebreak = distogram_use_entropy_tiebreak
        schemas = {str(row.get("cache_schema")) for row in self.rows}
        if schemas != {cache_schema}:
            raise ValueError(
                "Affinity dataset cannot mix cache schemas: "
                f"expected {cache_schema!r}, found {sorted(schemas)!r}."
            )
        self._reader = reader

    def __len__(self) -> int:
        return len(self.rows)

    def _get_reader(self) -> FeatureCacheReader:
        if self._reader is None:
            self._reader = FeatureCacheReader(self.cache_root)
        return self._reader

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __getitem__(self, index: int | AffinityBatchIndex) -> CachedAffinityItem | None:
        # The bucketed validation sampler uses -1 for only its final padded
        # batch; fixed-valid training batches never contain these positions.
        ranking_group_id = None
        if isinstance(index, AffinityBatchIndex):
            ranking_group_id = index.logical_group_id
            index = index.record_index
        if index < 0:
            return None
        row = self.rows[index]
        shard = row.get("cache_shard", row.get("shard"))
        key = row.get("cache_key", row.get("key", row.get("system_id")))
        if shard is None or key is None:
            raise KeyError("Affinity manifest rows require cache_shard and cache_key.")
        row_encoding = row.get("cache_encoding")
        if self.required_cache_encoding is not None and (
            row_encoding is None or str(row_encoding) != self.required_cache_encoding
        ):
            raise ValueError(
                "Affinity manifest row does not use the required cache encoding: "
                f"{row_encoding!r} != {self.required_cache_encoding!r}."
            )
        arrays = self._get_reader().get(
            {
                "shard": str(shard),
                "key": str(key),
                "cache_encoding": row_encoding,
            }
        )
        payload_schema = cache_schema_from_payload(arrays)
        if payload_schema != self.cache_schema:
            raise ValueError(
                "Cache payload schema does not match its finalized manifest row: "
                f"{payload_schema!r} != {self.cache_schema!r}."
            )
        if self.cache_schema in {
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
            AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
            AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1,
        }:
            crop = encoded_target_compact_features(arrays)
        elif self.crop_mode == "distogram_target_consensus_compact":
            crop = unpack_cross_only_payload(arrays)
        elif self.crop_mode in {
            "distogram_predicted_contact",
            "legacy_predicted_contact",
        }:
            crop = crop_full_cross_payload(
                arrays,
                max_tokens=self.max_crop_tokens,
                max_protein_tokens=self.max_protein_crop_tokens,
                pocket_distance_cutoff=(
                    self.distogram_pocket_distance_cutoff
                    if self.crop_mode == "distogram_predicted_contact"
                    else None
                ),
                use_entropy_tiebreak=(
                    self.distogram_use_entropy_tiebreak
                    if self.crop_mode == "distogram_predicted_contact"
                    else False
                ),
            )
        elif self.crop_mode == "distogram_query_window":
            profile = full_cross_pl_distogram_profile(arrays)
            crop_indices = select_pocket_annotation_crop(
                token_mask=torch.from_numpy(arrays["token_mask"]).bool(),
                chain_type=torch.from_numpy(arrays["chain_type"]).long(),
                protein_min_distance=profile.protein_min_expected_distance,
                max_tokens=self.max_crop_tokens,
                max_protein_tokens=self.max_protein_crop_tokens,
                neighborhood_size=self.pocket_neighborhood_size,
                require_contiguous_monomer=True,
            )
            crop = crop_full_cross_payload(
                arrays,
                max_tokens=self.max_crop_tokens,
                max_protein_tokens=self.max_protein_crop_tokens,
                crop_indices=crop_indices,
            )
        else:
            assert self.pocket_annotations is not None
            annotation = self.pocket_annotations.get(
                str(row["protein_key"]),
                protein_tokens=int(row["protein_tokens"]),
            )
            protein_min_distance = torch.from_numpy(
                annotation.protein_residue_min_distance
            )
            crop_indices = select_pocket_annotation_crop(
                token_mask=torch.from_numpy(arrays["token_mask"]).bool(),
                chain_type=torch.from_numpy(arrays["chain_type"]).long(),
                protein_min_distance=protein_min_distance,
                max_tokens=self.max_crop_tokens,
                max_protein_tokens=self.max_protein_crop_tokens,
                neighborhood_size=self.pocket_neighborhood_size,
                require_contiguous_monomer=(
                    self.crop_mode == "distogram_target_consensus"
                ),
            )
            crop = crop_full_cross_payload(
                arrays,
                max_tokens=self.max_crop_tokens,
                max_protein_tokens=self.max_protein_crop_tokens,
                crop_indices=crop_indices,
            )
        return CachedAffinityItem(
            features=crop,
            label=float(row["p_activity"]),
            assay_key=str(row["assay_key"]),
            canonical_smiles=str(row["canonical_smiles"]),
            shape_bucket=(
                int(row["shape_bucket"]) if row.get("shape_bucket") is not None else None
            ),
            origin=str(row["origin"]),
            record_id=str(row["record_id"]),
            ranking_group_id=ranking_group_id,
        )


def _bfloat16_bit_view(values: object, *, name: str) -> torch.Tensor:
    """View portable uint16 BF16 bits without a CPU float32 conversion."""
    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise ValueError(f"{name} must contain uint16 BF16 bit patterns.")
    array = np.ascontiguousarray(array)
    return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)


def encoded_target_compact_features(
    arrays: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Validate one target compact payload while retaining sparse BF16 values."""
    required = {
        "s_inputs_bf16",
        "s_lm_bf16",
        "token_mask",
        "chain_type",
        "crop_indices",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_feature_values_bf16",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"80k target compact affinity payload lacks fields: {missing}")
    singles = _bfloat16_bit_view(arrays["s_inputs_bf16"], name="s_inputs_bf16")
    single_lm = _bfloat16_bit_view(arrays["s_lm_bf16"], name="s_lm_bf16")
    token_mask = torch.from_numpy(np.ascontiguousarray(arrays["token_mask"])).bool()
    chain_type = torch.from_numpy(np.ascontiguousarray(arrays["chain_type"])).long()
    crop_indices = torch.from_numpy(np.ascontiguousarray(arrays["crop_indices"])).long()
    pair_indices = torch.from_numpy(np.ascontiguousarray(arrays["pair_indices"])).long()
    z_values = _bfloat16_bit_view(arrays["z_pair_values_bf16"], name="z_pair_values_bf16")
    distogram_values = _bfloat16_bit_view(
        arrays["distogram_feature_values_bf16"],
        name="distogram_feature_values_bf16",
    )
    length = len(token_mask)
    if length > 256:
        raise ValueError("80k target compact affinity payload exceeds 256 crop tokens.")
    if singles.ndim != 2 or single_lm.ndim != 2:
        raise ValueError("80k target compact singles must have shape [tokens, channels].")
    if singles.shape[0] != length or single_lm.shape[0] != length:
        raise ValueError("80k target compact singles must align with token_mask.")
    if chain_type.shape != (length,) or crop_indices.shape != (length,):
        raise ValueError("80k target compact token metadata must have equal length.")
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("80k target compact pair_indices must have shape [pairs, 2].")
    if z_values.ndim != 2 or len(z_values) != len(pair_indices):
        raise ValueError("80k target compact z values must align with pair_indices.")
    if distogram_values.shape != (len(pair_indices), 3):
        raise ValueError(
            "80k target compact distogram features must have shape [pairs, 3]."
        )
    if len(pair_indices):
        if pair_indices.min() < 0 or pair_indices.max() >= length:
            raise ValueError("80k target compact pair index is outside the crop.")
        flattened = pair_indices[:, 0] * length + pair_indices[:, 1]
        if len(torch.unique(flattened)) != len(flattened):
            raise ValueError("80k target compact pair indices must be unique.")
        left, right = pair_indices.unbind(dim=-1)
        if not bool((token_mask[left] & token_mask[right]).all()):
            raise ValueError("80k target compact pairs must reference valid tokens.")
        protein = chain_type == C.ChainType.PROTEIN.value
        ligand = chain_type == C.ChainType.LIGAND.value
        active = (protein[left] & ligand[right]) | (
            ligand[left] & (protein[right] | ligand[right])
        )
        if not bool(active.all()):
            raise ValueError("80k target compact pairs must contain only PL/LP/LL cells.")
    return {
        "s_inputs_bf16": singles,
        "s_lm_bf16": single_lm,
        "token_mask": token_mask,
        "chain_type": chain_type,
        "crop_indices": crop_indices,
        "pair_indices": pair_indices,
        "z_pair_values_bf16": z_values,
        "distogram_feature_values_bf16": distogram_values,
    }


def _collate_target_compact_batch(
    items: list[CachedAffinityItem | None],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Pad BF16 singles and concatenate sparse pairs without dense CPU tensors."""
    if max_tokens != 256:
        raise ValueError(
            "80k target compact affinity batches require exactly 256 tokens."
        )
    first = next(item for item in items if item is not None)
    if any(
        item is not None
        and (
            "z" in item.features
            or "distogram_features" in item.features
            or "pair_indices" not in item.features
        )
        for item in items
    ):
        raise ValueError("Affinity collation cannot mix sparse and dense records.")
    channel_s = first.features["s_inputs_bf16"].shape[-1]
    channel_lm = first.features["s_lm_bf16"].shape[-1]
    channel_z = first.features["z_pair_values_bf16"].shape[-1]
    batch_size = len(items)
    tensors: dict[str, torch.Tensor] = {
        "s_inputs_bf16": torch.zeros(
            (batch_size, max_tokens, channel_s), dtype=torch.bfloat16
        ),
        "s_lm_bf16": torch.zeros(
            (batch_size, max_tokens, channel_lm), dtype=torch.bfloat16
        ),
        "token_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "protein_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "ligand_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "target": torch.zeros((batch_size,), dtype=torch.float32),
        "valid_mask": torch.zeros((batch_size,), dtype=torch.bool),
        "group_index": torch.full((batch_size,), -1, dtype=torch.long),
        "ligand_index": torch.full((batch_size,), -1, dtype=torch.long),
    }
    sparse_indices: list[torch.Tensor] = []
    sparse_z: list[torch.Tensor] = []
    sparse_distogram: list[torch.Tensor] = []
    assay_keys: list[str | None] = [None] * batch_size
    canonical_smiles: list[str | None] = [None] * batch_size
    origins: list[str | None] = [None] * batch_size
    record_ids: list[str | None] = [None] * batch_size
    group_indices: dict[tuple[str, str | int], int] = {}
    ligand_indices: dict[str, int] = {}
    for batch_index, item in enumerate(items):
        if item is None:
            continue
        features = item.features
        length = len(features["token_mask"])
        if length > max_tokens:
            raise ValueError("80k target compact item exceeds the fixed token axis.")
        if (
            features["s_inputs_bf16"].shape != (length, channel_s)
            or features["s_lm_bf16"].shape != (length, channel_lm)
            or features["z_pair_values_bf16"].shape[-1] != channel_z
            or features["distogram_feature_values_bf16"].shape[-1] != 3
        ):
            raise ValueError("80k target compact batch mixes feature channel dimensions.")
        tensors["s_inputs_bf16"][batch_index, :length] = features["s_inputs_bf16"]
        tensors["s_lm_bf16"][batch_index, :length] = features["s_lm_bf16"]
        token_mask = features["token_mask"].bool()
        chain_type = features["chain_type"].long()
        tensors["token_mask"][batch_index, :length] = token_mask
        tensors["protein_mask"][batch_index, :length] = token_mask & (
            chain_type == C.ChainType.PROTEIN.value
        )
        tensors["ligand_mask"][batch_index, :length] = token_mask & (
            chain_type == C.ChainType.LIGAND.value
        )
        pair_indices = features["pair_indices"].long()
        batch_column = torch.full((len(pair_indices), 1), batch_index, dtype=torch.long)
        sparse_indices.append(torch.cat((batch_column, pair_indices), dim=-1))
        sparse_z.append(features["z_pair_values_bf16"])
        sparse_distogram.append(features["distogram_feature_values_bf16"])
        tensors["target"][batch_index] = item.label
        tensors["valid_mask"][batch_index] = True
        assay_keys[batch_index] = item.assay_key
        canonical_smiles[batch_index] = item.canonical_smiles
        origins[batch_index] = item.origin
        record_ids[batch_index] = item.record_id
        group_key: tuple[str, str | int] = (
            ("assay", item.assay_key)
            if item.ranking_group_id is None
            else ("logical", item.ranking_group_id)
        )
        tensors["group_index"][batch_index] = group_indices.setdefault(
            group_key, len(group_indices)
        )
        tensors["ligand_index"][batch_index] = ligand_indices.setdefault(
            item.canonical_smiles, len(ligand_indices)
        )
    return {
        **tensors,
        "pair_indices": torch.cat(sparse_indices, dim=0),
        "z_pair_values_bf16": torch.cat(sparse_z, dim=0),
        "distogram_feature_values_bf16": torch.cat(sparse_distogram, dim=0),
        "sparse_compact": True,
        "assay_keys": assay_keys,
        "canonical_smiles": canonical_smiles,
        "origins": origins,
        "record_ids": record_ids,
    }


def collate_affinity_batch(
    items: list[CachedAffinityItem | None],
    *,
    batch_size: int | None = None,
    assays_per_batch: int = 8,
    records_per_assay: int = 4,
    fixed_shape: bool = False,
) -> dict[str, Any]:
    """Pad a fixed assay-group batch while retaining explicit invalid slots."""
    expected_size = (
        batch_size if batch_size is not None else assays_per_batch * records_per_assay
    )
    if expected_size <= 0:
        raise ValueError("Affinity batch size must be positive.")
    if len(items) != expected_size:
        raise ValueError(
            f"Expected fixed batch size {expected_size}, received {len(items)}."
        )
    first = next((item for item in items if item is not None), None)
    if first is None:
        raise ValueError("An affinity batch must contain at least one valid record.")
    if fixed_shape:
        buckets = {item.shape_bucket for item in items if item is not None}
        if None in buckets or len(buckets) != 1:
            raise ValueError(
                "Fixed-shape affinity collation requires one non-null shape bucket."
            )
        max_tokens = int(next(iter(buckets)))
        if any(
            item.features["token_mask"].shape[0] > max_tokens
            for item in items
            if item is not None
        ):
            raise ValueError("Affinity item exceeds its assigned fixed shape bucket.")
    else:
        max_tokens = max(item.features["token_mask"].shape[0] for item in items if item)
    is_target_compact = "z_pair_values_bf16" in first.features and "z" not in (
        first.features
    )
    if is_target_compact:
        return _collate_target_compact_batch(items, max_tokens=max_tokens)
    channel_s = first.features["s_inputs"].shape[-1]
    channel_lm = first.features["s_lm"].shape[-1]
    channel_z = first.features["z"].shape[-1]
    channel_d = first.features["distogram_features"].shape[-1]
    batch_size = len(items)
    tensors: dict[str, torch.Tensor] = {
        "s_inputs": torch.zeros((batch_size, max_tokens, channel_s), dtype=torch.float32),
        "s_lm": torch.zeros((batch_size, max_tokens, channel_lm), dtype=torch.float32),
        "z": torch.zeros(
            (batch_size, max_tokens, max_tokens, channel_z), dtype=torch.float32
        ),
        "distogram_features": torch.zeros(
            (batch_size, max_tokens, max_tokens, channel_d), dtype=torch.float32
        ),
        "token_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "protein_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "ligand_mask": torch.zeros((batch_size, max_tokens), dtype=torch.bool),
        "target": torch.zeros((batch_size,), dtype=torch.float32),
        "valid_mask": torch.zeros((batch_size,), dtype=torch.bool),
        "group_index": torch.full((batch_size,), -1, dtype=torch.long),
        "ligand_index": torch.full((batch_size,), -1, dtype=torch.long),
    }
    assay_keys: list[str | None] = [None] * batch_size
    canonical_smiles: list[str | None] = [None] * batch_size
    origins: list[str | None] = [None] * batch_size
    record_ids: list[str | None] = [None] * batch_size
    group_indices: dict[tuple[str, str | int], int] = {}
    ligand_indices: dict[str, int] = {}
    for batch_index, item in enumerate(items):
        if item is None:
            continue
        length = item.features["token_mask"].shape[0]
        tensors["s_inputs"][batch_index, :length] = item.features["s_inputs"].float()
        tensors["s_lm"][batch_index, :length] = item.features["s_lm"].float()
        tensors["z"][batch_index, :length, :length] = item.features["z"].float()
        tensors["distogram_features"][batch_index, :length, :length] = item.features[
            "distogram_features"
        ].float()
        token_mask = item.features["token_mask"].bool()
        chain_type = item.features["chain_type"].long()
        tensors["token_mask"][batch_index, :length] = token_mask
        tensors["protein_mask"][batch_index, :length] = token_mask & (
            chain_type == C.ChainType.PROTEIN.value
        )
        tensors["ligand_mask"][batch_index, :length] = token_mask & (
            chain_type == C.ChainType.LIGAND.value
        )
        tensors["target"][batch_index] = item.label
        tensors["valid_mask"][batch_index] = True
        assay_keys[batch_index] = item.assay_key
        canonical_smiles[batch_index] = item.canonical_smiles
        origins[batch_index] = item.origin
        record_ids[batch_index] = item.record_id
        group_key: tuple[str, str | int]
        if item.ranking_group_id is None:
            group_key = ("assay", item.assay_key)
        else:
            group_key = ("logical", item.ranking_group_id)
        tensors["group_index"][batch_index] = group_indices.setdefault(
            group_key, len(group_indices)
        )
        tensors["ligand_index"][batch_index] = ligand_indices.setdefault(
            item.canonical_smiles, len(ligand_indices)
        )
    return {
        **tensors,
        "assay_keys": assay_keys,
        "canonical_smiles": canonical_smiles,
        "origins": origins,
        "record_ids": record_ids,
    }


def collate_affinity_eval(
    items: list[CachedAffinityItem | None],
    *,
    fixed_shape: bool = False,
) -> dict[str, Any]:
    """Pad a sequential validation batch to the same fixed tensor layout."""
    if len(items) > 32:
        raise ValueError("Affinity evaluation batches may contain at most 32 items.")
    return collate_affinity_batch(
        items + [None] * (32 - len(items)),
        fixed_shape=fixed_shape,
    )


def materialize_target_compact_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Scatter a transferred sparse batch into the head's fixed dense BF16 inputs."""
    if not batch.get("sparse_compact", False):
        return batch
    required = {
        "s_inputs_bf16",
        "s_lm_bf16",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_feature_values_bf16",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"Sparse compact affinity batch lacks fields: {missing}")
    s_inputs = batch["s_inputs_bf16"]
    s_lm = batch["s_lm_bf16"]
    pair_indices = batch["pair_indices"].long()
    z_values = batch["z_pair_values_bf16"]
    distogram_values = batch["distogram_feature_values_bf16"]
    if s_inputs.dtype != torch.bfloat16 or s_lm.dtype != torch.bfloat16:
        raise ValueError("80k target compact singles must stay BF16 through transfer.")
    if z_values.dtype != torch.bfloat16 or distogram_values.dtype != torch.bfloat16:
        raise ValueError(
            "80k target compact pair values must stay BF16 through transfer."
        )
    batch_size, tokens, _ = s_inputs.shape
    if tokens != 256 or s_lm.shape[:2] != (batch_size, tokens):
        raise ValueError("80k target compact materialization requires [B, 256] singles.")
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 3:
        raise ValueError("Batched sparse pair indices must have shape [pairs, 3].")
    if len(pair_indices) != len(z_values) or len(pair_indices) != len(distogram_values):
        raise ValueError("Batched sparse indices and values must align.")
    if len(pair_indices) and (
        pair_indices[:, 0].min() < 0
        or pair_indices[:, 0].max() >= batch_size
        or pair_indices[:, 1:].min() < 0
        or pair_indices[:, 1:].max() >= tokens
    ):
        raise ValueError("Batched sparse pair index is out of bounds.")
    z = z_values.new_zeros((batch_size, tokens, tokens, z_values.shape[-1]))
    distogram = distogram_values.new_zeros(
        (batch_size, tokens, tokens, distogram_values.shape[-1])
    )
    if len(pair_indices):
        sample, left, right = pair_indices.unbind(dim=-1)
        z[sample, left, right] = z_values
        distogram[sample, left, right] = distogram_values
    dense = dict(batch)
    for name in (
        "s_inputs_bf16",
        "s_lm_bf16",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_feature_values_bf16",
        "sparse_compact",
    ):
        dense.pop(name)
    dense.update(
        {
            "s_inputs": s_inputs,
            "s_lm": s_lm,
            "z": z,
            "distogram_features": distogram,
        }
    )
    return dense
