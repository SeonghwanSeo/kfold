"""Storage layouts for the affinity head's active pair representation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

import kfold.constants as C

from .cache import serialize_npz
from .crop import (
    crop_distogram_features,
    distogram_feature_maps,
    query_adaptive_final_crop_indices,
    select_ligand_preserving_crop_from_scores,
)
from .schema import (
    AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
    AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
    AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1,
    AFFINITY_FULL_CROSS_SCHEMAS,
)

PAIR_STORAGE_DENSE_SOURCE = "dense_source"
PAIR_STORAGE_CROPPED_DENSE = "cropped_dense"
PAIR_STORAGE_CROSS_ONLY_SPARSE = "cross_only_sparse_v1"
PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM = "cross_only_bf16_tri_distogram_v2"
PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS = "cross_only_bf16_tri_logits_v3"
PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS = "full_cross_bf16_tri_logits_v4"
PAIR_STORAGE_POCKET80K_TARGET_BF16 = "pocket80k_target_bf16_v1"
PAIR_STORAGE_POCKET80K_QUERY_ADAPTIVE_BF16 = "pocket80k_query_adaptive_bf16_v1"
AFFINITY_CACHE_SCHEMA_LEGACY_CROPPED = "affinity_cropped_legacy"
PAIR_STORAGE_MODES = (
    PAIR_STORAGE_DENSE_SOURCE,
    PAIR_STORAGE_CROPPED_DENSE,
    PAIR_STORAGE_CROSS_ONLY_SPARSE,
    PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
    PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
    PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS,
)
# Modes whose packer consumes a crop-local payload that already carries raw
# distogram logits rather than the three derived conditioning maps.
PAIR_STORAGE_LOGIT_MODES = (
    PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
    PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS,
)


@dataclass(frozen=True, kw_only=True)
class PairStoragePayload:
    """Encoded affinity input plus accounting for its retained pair cells."""

    arrays: dict[str, np.ndarray]
    mode: str
    stored_pair_count: int
    crop_pair_count: int
    stored_distogram_pair_count: int
    distogram_symmetry_max_abs: float | None


@dataclass(frozen=True, kw_only=True)
class PLDistogramProfile:
    """Coordinate-free PL reductions decoded from one immutable v1 payload."""

    protein_max_contact_probability: torch.Tensor
    protein_min_expected_distance: torch.Tensor
    protein_mean_normalized_entropy: torch.Tensor

    def exact_pocket_hlp(
        self,
        *,
        distance_cutoff: float = 15.0,
    ) -> tuple[float | None, int]:
        """Average entropy for PL pairs whose protein token is in the pocket."""
        if distance_cutoff <= 0:
            raise ValueError("distance_cutoff must be positive.")
        pocket = self.protein_min_expected_distance < distance_cutoff
        count = int(pocket.sum().item())
        if not count:
            return None, 0
        return float(self.protein_mean_normalized_entropy[pocket].mean().item()), count


def encode_bfloat16(values: np.ndarray) -> np.ndarray:
    """Store float values as portable uint16 BF16 bit patterns."""
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return tensor.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def decode_bfloat16(values: np.ndarray) -> np.ndarray:
    """Restore uint16 BF16 bit patterns to float32 NumPy values."""
    bits = torch.from_numpy(np.ascontiguousarray(values, dtype=np.uint16).view(np.int16))
    return bits.view(torch.bfloat16).float().numpy()


def _as_torch(arrays: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "s_inputs": torch.from_numpy(np.asarray(arrays["s_inputs"])),
        "s_lm": torch.from_numpy(np.asarray(arrays["s_lm"])),
        "z": torch.from_numpy(np.asarray(arrays["z"])),
        "token_mask": torch.from_numpy(np.asarray(arrays["token_mask"])).bool(),
        "chain_type": torch.from_numpy(np.asarray(arrays["chain_type"])).long(),
        "crop_indices": torch.from_numpy(np.asarray(arrays["crop_indices"])).long(),
        "contact_probability": torch.from_numpy(
            np.asarray(arrays["contact_probability"])
        ),
        "expected_distance": torch.from_numpy(np.asarray(arrays["expected_distance"])),
        "entropy": torch.from_numpy(np.asarray(arrays["distogram_entropy"])),
    }


def crop_dense_payload(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the complete, crop-local dense layout from a trunk payload."""
    tensors = _as_torch(arrays)
    crop = crop_distogram_features(
        s_inputs=tensors["s_inputs"],
        s_lm=tensors["s_lm"],
        z=tensors["z"],
        token_mask=tensors["token_mask"],
        chain_type=tensors["chain_type"],
        crop_indices=tensors["crop_indices"],
        contact_probability=tensors["contact_probability"],
        expected_distance=tensors["expected_distance"],
        entropy=tensors["entropy"],
    )
    return {key: np.ascontiguousarray(value.numpy()) for key, value in crop.items()}


def affinity_cross_pair_mask(
    token_mask: np.ndarray,
    chain_type: np.ndarray,
) -> np.ndarray:
    """Return the directed PL/LP/LL mask used by Boltz, TerraBind, and NESSO."""
    valid = np.asarray(token_mask, dtype=bool)
    chain_type = np.asarray(chain_type)
    protein = valid & (chain_type == C.ChainType.PROTEIN.value)
    ligand = valid & (chain_type == C.ChainType.LIGAND.value)
    return (
        (protein[:, None] & ligand[None, :])
        | (ligand[:, None] & protein[None, :])
        | (ligand[:, None] & ligand[None, :])
    )


def verify_symmetric_distogram_features(
    distogram_features: np.ndarray,
    token_mask: np.ndarray,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> float:
    """Validate crop-local distogram maps before lossless triangular packing."""
    if distogram_features.ndim != 3:
        raise ValueError("distogram_features must have shape [tokens, tokens, channels].")
    if distogram_features.shape[0] != distogram_features.shape[1]:
        raise ValueError("distogram_features must have square token axes.")
    valid = np.asarray(token_mask, dtype=bool)
    if len(valid) != distogram_features.shape[0]:
        raise ValueError("token_mask must align with distogram token axes.")
    pair_mask = valid[:, None] & valid[None, :]
    transpose = np.swapaxes(distogram_features, 0, 1)
    difference = np.abs(distogram_features - transpose)
    max_abs = float(difference[pair_mask].max()) if pair_mask.any() else 0.0
    if not np.allclose(
        distogram_features[pair_mask],
        transpose[pair_mask],
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError(
            "Distogram feature maps are not symmetric enough for triangular packing: "
            f"max_abs={max_abs:.6g}, atol={atol}, rtol={rtol}."
        )
    return max_abs


def pack_pair_storage(
    arrays: Mapping[str, np.ndarray],
    *,
    mode: str,
) -> PairStoragePayload:
    """Encode an extracted cache record with only the requested pair cells."""
    if mode not in PAIR_STORAGE_MODES:
        raise ValueError(f"Unsupported pair storage mode: {mode!r}.")
    if mode == PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS:
        return pack_full_cross_pair_storage(arrays, mode=mode)
    if mode == PAIR_STORAGE_DENSE_SOURCE:
        source_length = len(np.asarray(arrays["token_mask"]))
        crop_length = len(np.asarray(arrays["crop_indices"]))
        return PairStoragePayload(
            arrays={key: np.ascontiguousarray(value) for key, value in arrays.items()},
            mode=mode,
            stored_pair_count=source_length * source_length,
            crop_pair_count=crop_length * crop_length,
            stored_distogram_pair_count=source_length * source_length,
            distogram_symmetry_max_abs=None,
        )

    cropped = crop_dense_payload(arrays)
    if mode == PAIR_STORAGE_CROPPED_DENSE:
        crop_pair_count = len(cropped["token_mask"]) ** 2
        return PairStoragePayload(
            arrays=cropped,
            mode=mode,
            stored_pair_count=crop_pair_count,
            crop_pair_count=crop_pair_count,
            stored_distogram_pair_count=crop_pair_count,
            distogram_symmetry_max_abs=None,
        )
    return pack_cropped_pair_storage(cropped, mode=mode)


def pack_cropped_pair_storage(
    cropped: Mapping[str, np.ndarray],
    *,
    mode: str,
) -> PairStoragePayload:
    """Encode an already crop-local payload into one of the sparse layouts.

    The extractor slices ``z`` and the distogram on the GPU, so the dense
    ``[source, source, channels]`` tensors never reach host memory.  This entry
    point takes that crop-local payload directly.
    """
    if mode not in (
        PAIR_STORAGE_CROSS_ONLY_SPARSE,
        PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
        PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
    ):
        raise ValueError(f"Mode {mode!r} does not accept a crop-local payload.")
    crop_length = len(cropped["token_mask"])
    crop_pair_count = crop_length * crop_length
    pair_mask = affinity_cross_pair_mask(cropped["token_mask"], cropped["chain_type"])
    pair_indices = np.stack(np.nonzero(pair_mask), axis=-1).astype(np.uint16)
    pair_values = cropped["z"][pair_mask]

    if mode == PAIR_STORAGE_CROSS_ONLY_SPARSE:
        distogram_values = cropped["distogram_features"][pair_mask]
        return PairStoragePayload(
            arrays={
                "s_inputs": cropped["s_inputs"],
                "s_lm": cropped["s_lm"],
                "token_mask": cropped["token_mask"],
                "chain_type": cropped["chain_type"],
                "crop_indices": cropped["crop_indices"].astype(np.int32),
                "pair_indices": pair_indices,
                "z_pair_values": np.ascontiguousarray(pair_values),
                "distogram_feature_values": np.ascontiguousarray(distogram_values),
            },
            mode=mode,
            stored_pair_count=len(pair_indices),
            crop_pair_count=crop_pair_count,
            stored_distogram_pair_count=len(pair_indices),
            distogram_symmetry_max_abs=None,
        )

    if mode == PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS:
        source_key, stored_key = "distogram_logits", "distogram_tri_logits_bf16"
    else:
        source_key, stored_key = "distogram_features", "distogram_tri_values_bf16"
    if source_key not in cropped:
        raise KeyError(f"Mode {mode!r} requires a crop-local {source_key!r} array.")

    symmetry_max_abs = verify_symmetric_distogram_features(
        cropped[source_key],
        cropped["token_mask"],
    )
    triangular_mask = pair_mask & np.triu(np.ones((crop_length, crop_length), dtype=bool))
    triangular_indices = np.stack(np.nonzero(triangular_mask), axis=-1).astype(np.uint16)
    return PairStoragePayload(
        arrays={
            "s_inputs_bf16": encode_bfloat16(cropped["s_inputs"]),
            "s_lm_bf16": encode_bfloat16(cropped["s_lm"]),
            "token_mask": cropped["token_mask"],
            "chain_type": cropped["chain_type"],
            "crop_indices": cropped["crop_indices"].astype(np.int32),
            "pair_indices": pair_indices,
            "z_pair_values_bf16": encode_bfloat16(pair_values),
            "distogram_tri_indices": triangular_indices,
            stored_key: encode_bfloat16(cropped[source_key][triangular_mask]),
            "distogram_symmetry_max_abs": np.asarray(
                [symmetry_max_abs], dtype=np.float32
            ),
        },
        mode=mode,
        stored_pair_count=len(pair_indices),
        crop_pair_count=crop_pair_count,
        stored_distogram_pair_count=len(triangular_indices),
        distogram_symmetry_max_abs=symmetry_max_abs,
    )


def pack_compact_pocket_storage(
    arrays: Mapping[str, np.ndarray],
    *,
    cache_schema: str = AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1,
) -> PairStoragePayload:
    """Validate an already GPU-sliced pocket payload without expanding pairs."""
    if cache_schema != AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1:
        raise ValueError(f"Unsupported compact pocket schema: {cache_schema!r}.")
    required = {
        "s_inputs_bf16",
        "s_lm_bf16",
        "token_mask",
        "chain_type",
        "crop_indices",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_tri_indices",
        "distogram_tri_logits_bf16",
        "distogram_symmetry_max_abs",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Compact pocket payload lacks fields: {missing}")
    token_mask = np.asarray(arrays["token_mask"], dtype=bool)
    chain_type = np.asarray(arrays["chain_type"], dtype=np.int64)
    crop_indices = np.asarray(arrays["crop_indices"], dtype=np.int32)
    if chain_type.shape != token_mask.shape or crop_indices.shape != token_mask.shape:
        raise ValueError("Compact pocket token fields must have equal length.")
    pair_indices = _cross_pair_indices(token_mask, chain_type)
    triangular_indices = _triangular_cross_indices(token_mask, chain_type)
    if not np.array_equal(np.asarray(arrays["pair_indices"]), pair_indices):
        raise ValueError("Compact pocket pair indices do not match PL/LP/LL cells.")
    if not np.array_equal(
        np.asarray(arrays["distogram_tri_indices"]), triangular_indices
    ):
        raise ValueError("Compact pocket distogram indices are not triangular PL/LP/LL.")
    pair_values = np.asarray(arrays["z_pair_values_bf16"], dtype=np.uint16)
    logits = np.asarray(arrays["distogram_tri_logits_bf16"], dtype=np.uint16)
    if len(pair_values) != len(pair_indices) or len(logits) != len(triangular_indices):
        raise ValueError("Compact pocket values do not align with their indices.")
    symmetry = float(np.asarray(arrays["distogram_symmetry_max_abs"]).item())
    output = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    output["cache_schema"] = np.asarray(cache_schema)
    return PairStoragePayload(
        arrays=output,
        mode=PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
        stored_pair_count=len(pair_indices),
        crop_pair_count=len(token_mask) ** 2,
        stored_distogram_pair_count=len(triangular_indices),
        distogram_symmetry_max_abs=symmetry,
    )


def pack_pocket80k_target_storage(
    arrays: Mapping[str, np.ndarray],
    *,
    cache_schema: str = AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
) -> PairStoragePayload:
    """Validate the final cropped 80k payload without decoding its BF16 values."""
    if cache_schema not in {
        AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
        AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
    }:
        raise ValueError(f"Unsupported 80k target-consensus schema: {cache_schema!r}.")
    required = {
        "s_inputs_bf16",
        "s_lm_bf16",
        "token_mask",
        "chain_type",
        "crop_indices",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_feature_values_bf16",
        "pocket_hlp_15a",
        "pocket_residue_count",
        "consensus_protein_tokens",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"80k target-consensus payload lacks fields: {missing}")
    token_mask = np.asarray(arrays["token_mask"], dtype=bool)
    chain_type = np.asarray(arrays["chain_type"], dtype=np.int64)
    crop_indices = np.asarray(arrays["crop_indices"], dtype=np.int32)
    if chain_type.shape != token_mask.shape or crop_indices.shape != token_mask.shape:
        raise ValueError("80k target-consensus token fields must have equal length.")
    expected_pairs = _cross_pair_indices(token_mask, chain_type)
    pair_indices = np.asarray(arrays["pair_indices"], dtype=np.uint16)
    if not np.array_equal(pair_indices, expected_pairs):
        raise ValueError("80k target-consensus pair indices do not match PL/LP/LL cells.")
    z_values = np.asarray(arrays["z_pair_values_bf16"], dtype=np.uint16)
    distogram = np.asarray(arrays["distogram_feature_values_bf16"], dtype=np.uint16)
    if len(z_values) != len(pair_indices):
        raise ValueError("80k target-consensus z values do not align with pair indices.")
    if distogram.shape != (len(pair_indices), 3):
        raise ValueError(
            "80k target-consensus distogram features must have shape [pairs, 3]."
        )
    if np.asarray(arrays["s_inputs_bf16"]).shape[0] != len(token_mask) or np.asarray(
        arrays["s_lm_bf16"]
    ).shape[0] != len(token_mask):
        raise ValueError("80k target-consensus singles must align with crop tokens.")
    output = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    output["cache_schema"] = np.asarray(cache_schema)
    return PairStoragePayload(
        arrays=output,
        mode=PAIR_STORAGE_POCKET80K_TARGET_BF16,
        stored_pair_count=len(pair_indices),
        crop_pair_count=len(token_mask) ** 2,
        stored_distogram_pair_count=len(pair_indices),
        distogram_symmetry_max_abs=None,
    )


def _payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def query_adaptive_delta_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Digest a delta without recursively hashing its own digest field."""
    return _payload_sha256(
        {key: value for key, value in arrays.items() if key != "delta_sha256"}
    )


def pack_query_adaptive_delta(
    arrays: Mapping[str, np.ndarray],
    *,
    base_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bind a query-adaptive delta to one immutable canonical target payload."""
    required = {
        "contract_version",
        "extra_protein_source_indices",
        "extra_s_inputs_bf16",
        "extra_s_lm_bf16",
        "delta_pair_source_indices",
        "delta_z_pair_values_bf16",
        "delta_distogram_feature_values_bf16",
        "query_residue_order",
        "query_window_order",
        "accepted_query_windows",
        "target_consensus_window_order",
        "query_protein_expected_distance_bf16",
        "query_protein_entropy_bf16",
        "checkpoint_sha256",
        "crop_contract_version",
        "pocket_manifest_sha256",
        "base_crop_sha256",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Query-adaptive delta lacks fields: {missing}")
    if str(np.asarray(arrays["contract_version"]).item()) != (
        "query_adaptive_delta20_v1"
    ):
        raise ValueError("Query-adaptive delta has the wrong contract.")
    if str(np.asarray(arrays["crop_contract_version"]).item()) != (
        "target_consensus_100_v1"
    ):
        raise ValueError("Query-adaptive delta has the wrong base crop contract.")
    expected_crop_sha = hashlib.sha256(
        np.asarray(base_arrays["crop_indices"], dtype=np.int32).tobytes()
    ).hexdigest()
    if str(np.asarray(arrays["base_crop_sha256"]).item()) != expected_crop_sha:
        raise ValueError("Adaptive delta base-crop digest is invalid.")
    extra = np.asarray(arrays["extra_protein_source_indices"], dtype=np.int32)
    if extra.ndim != 1 or len(extra) > 40 or len(np.unique(extra)) != len(extra):
        raise ValueError("Adaptive delta must contain at most 40 unique proteins.")
    if len(extra) and not np.all(extra[1:] > extra[:-1]):
        raise ValueError("Adaptive extra protein indices must be sorted.")
    base_sources = np.asarray(base_arrays["crop_indices"], dtype=np.int32)
    if set(extra.tolist()).intersection(base_sources.tolist()):
        raise ValueError("Adaptive extra proteins overlap the canonical crop.")
    query_tokens = len(np.asarray(arrays["query_protein_expected_distance_bf16"]))
    if (extra < 0).any() or (extra >= query_tokens).any():
        raise ValueError("Adaptive extra protein index is outside the source protein.")
    for name in ("extra_s_inputs_bf16", "extra_s_lm_bf16"):
        if np.asarray(arrays[name], dtype=np.uint16).shape[0] != len(extra):
            raise ValueError("Adaptive extra singles must align with protein indices.")
    pair_indices = np.asarray(arrays["delta_pair_source_indices"], dtype=np.int32)
    z_values = np.asarray(arrays["delta_z_pair_values_bf16"], dtype=np.uint16)
    distogram = np.asarray(arrays["delta_distogram_feature_values_bf16"], dtype=np.uint16)
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("Adaptive delta source pairs must have shape [pairs, 2].")
    if len(pair_indices) != len(z_values) or distogram.shape != (len(pair_indices), 3):
        raise ValueError("Adaptive delta pair values do not align.")
    base_chain = np.asarray(base_arrays["chain_type"], dtype=np.int64)
    ligand_sources = base_sources[base_chain == C.ChainType.LIGAND.value]
    expected_pairs = np.asarray(
        [(int(protein), int(ligand)) for protein in extra for ligand in ligand_sources]
        + [(int(ligand), int(protein)) for protein in extra for ligand in ligand_sources],
        dtype=np.int32,
    ).reshape(-1, 2)
    if not np.array_equal(pair_indices, expected_pairs):
        raise ValueError("Adaptive delta must store both directions for every PL pair.")
    output = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    output["base_payload_sha256"] = np.asarray(_payload_sha256(base_arrays))
    output["delta_sha256"] = np.asarray(query_adaptive_delta_sha256(output))
    return output


def repack_target_with_query_adaptive_delta(
    base_arrays: Mapping[str, np.ndarray],
    delta_arrays: Mapping[str, np.ndarray],
) -> PairStoragePayload:
    """Offline-repack canonical target features into a frozen 80/20 ablation."""
    if cache_schema_from_payload(base_arrays) not in {
        AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_COMPACT_V1,
        AFFINITY_CACHE_SCHEMA_POCKET80K_TARGET_DIRECT_V2,
    }:
        raise ValueError("Adaptive repack requires canonical target-consensus input.")
    recorded_delta = str(np.asarray(delta_arrays["delta_sha256"]).item())
    if recorded_delta != query_adaptive_delta_sha256(delta_arrays):
        raise ValueError("Adaptive delta digest is invalid.")
    base_sha = _payload_sha256(base_arrays)
    if str(np.asarray(delta_arrays["base_payload_sha256"]).item()) != base_sha:
        raise ValueError("Adaptive delta belongs to a different canonical payload.")
    base_sources = np.asarray(base_arrays["crop_indices"], dtype=np.int32)
    base_chain = np.asarray(base_arrays["chain_type"], dtype=np.int64)
    extra = np.asarray(
        delta_arrays["extra_protein_source_indices"], dtype=np.int32
    ).tolist()
    final_sources = query_adaptive_final_crop_indices(
        base_crop_indices=torch.from_numpy(base_sources),
        base_chain_type=torch.from_numpy(base_chain),
        extra_protein_source_indices=torch.tensor(extra, dtype=torch.long),
        target_consensus_window_order=torch.from_numpy(
            np.asarray(delta_arrays["target_consensus_window_order"], dtype=np.int32)
        ),
        accepted_query_windows=torch.from_numpy(
            np.asarray(delta_arrays["accepted_query_windows"], dtype=np.int32)
        ),
        max_tail_tokens=40,
    ).tolist()
    if len(final_sources) > 256:
        raise ValueError("Adaptive repack exceeds the 256-token budget.")
    base_position = {int(source): index for index, source in enumerate(base_sources)}
    extra_position = {int(source): index for index, source in enumerate(extra)}
    s_inputs = []
    s_lm = []
    chain_type = []
    token_mask = []
    for source in final_sources:
        if source in base_position:
            index = base_position[source]
            s_inputs.append(base_arrays["s_inputs_bf16"][index])
            s_lm.append(base_arrays["s_lm_bf16"][index])
            chain_type.append(base_arrays["chain_type"][index])
            token_mask.append(base_arrays["token_mask"][index])
        else:
            index = extra_position[source]
            s_inputs.append(delta_arrays["extra_s_inputs_bf16"][index])
            s_lm.append(delta_arrays["extra_s_lm_bf16"][index])
            chain_type.append(C.ChainType.PROTEIN.value)
            token_mask.append(True)
    chain_array = np.asarray(chain_type, dtype=np.int64)
    mask_array = np.asarray(token_mask, dtype=bool)
    expected_pairs = _cross_pair_indices(mask_array, chain_array)
    source_pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for pair, z_value, d_value in zip(
        np.asarray(base_arrays["pair_indices"], dtype=np.int64),
        base_arrays["z_pair_values_bf16"],
        base_arrays["distogram_feature_values_bf16"],
        strict=True,
    ):
        source_pairs[(int(base_sources[pair[0]]), int(base_sources[pair[1]]))] = (
            z_value,
            d_value,
        )
    for pair, z_value, d_value in zip(
        np.asarray(delta_arrays["delta_pair_source_indices"], dtype=np.int64),
        delta_arrays["delta_z_pair_values_bf16"],
        delta_arrays["delta_distogram_feature_values_bf16"],
        strict=True,
    ):
        source_pairs[(int(pair[0]), int(pair[1]))] = (z_value, d_value)
    z_values = []
    distogram_values = []
    for left, right in expected_pairs.tolist():
        source_pair = (final_sources[left], final_sources[right])
        try:
            z_value, d_value = source_pairs[source_pair]
        except KeyError as exc:
            raise ValueError(f"Adaptive delta lacks final pair {source_pair}.") from exc
        z_values.append(z_value)
        distogram_values.append(d_value)
    output = {
        "cache_schema": np.asarray(
            AFFINITY_CACHE_SCHEMA_POCKET80K_QUERY_ADAPTIVE_80_20_V1
        ),
        "s_inputs_bf16": np.ascontiguousarray(np.stack(s_inputs)),
        "s_lm_bf16": np.ascontiguousarray(np.stack(s_lm)),
        "token_mask": mask_array,
        "chain_type": chain_array,
        "crop_indices": np.asarray(final_sources, dtype=np.int32),
        "pair_indices": expected_pairs,
        "z_pair_values_bf16": np.ascontiguousarray(np.stack(z_values)),
        "distogram_feature_values_bf16": np.ascontiguousarray(np.stack(distogram_values)),
        "pocket_hlp_15a": np.asarray(base_arrays["pocket_hlp_15a"]),
        "pocket_residue_count": np.asarray(base_arrays["pocket_residue_count"]),
        "base_payload_sha256": np.asarray(base_sha),
        "adaptive_delta_sha256": np.asarray(recorded_delta),
        "crop_contract_version": np.asarray("query_adaptive_80_20_v1"),
    }
    return PairStoragePayload(
        arrays=output,
        mode=PAIR_STORAGE_POCKET80K_QUERY_ADAPTIVE_BF16,
        stored_pair_count=len(expected_pairs),
        crop_pair_count=len(final_sources) ** 2,
        stored_distogram_pair_count=len(expected_pairs),
        distogram_symmetry_max_abs=None,
    )


def cache_schema_from_payload(arrays: Mapping[str, np.ndarray]) -> str:
    """Return the explicit schema tag, treating pre-v4 payloads as legacy."""
    value = arrays.get("cache_schema")
    if value is None:
        return AFFINITY_CACHE_SCHEMA_LEGACY_CROPPED
    return str(np.asarray(value).item())


def is_full_cross_payload(arrays: Mapping[str, np.ndarray]) -> bool:
    """Whether a payload stores full-source singles and sparse cross pairs."""
    return cache_schema_from_payload(arrays) in AFFINITY_FULL_CROSS_SCHEMAS


def _cross_pair_indices(token_mask: np.ndarray, chain_type: np.ndarray) -> np.ndarray:
    pair_mask = affinity_cross_pair_mask(token_mask, chain_type)
    return np.stack(np.nonzero(pair_mask), axis=-1).astype(np.uint16)


def _triangular_cross_indices(
    token_mask: np.ndarray, chain_type: np.ndarray
) -> np.ndarray:
    length = len(token_mask)
    pair_mask = affinity_cross_pair_mask(token_mask, chain_type)
    triangular = pair_mask & np.triu(np.ones((length, length), dtype=bool))
    return np.stack(np.nonzero(triangular), axis=-1).astype(np.uint16)


def _array_or_sparse(
    arrays: Mapping[str, np.ndarray],
    *,
    sparse_indices_key: str,
    sparse_values_key: str,
    expected_indices: np.ndarray,
    dense_key: str,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read active values either from a GPU-sliced sparse payload or a dense test one."""
    if sparse_indices_key in arrays or sparse_values_key in arrays:
        if sparse_indices_key not in arrays or sparse_values_key not in arrays:
            raise KeyError(
                f"Sparse payload requires both {sparse_indices_key!r} and "
                f"{sparse_values_key!r}."
            )
        indices = np.asarray(arrays[sparse_indices_key], dtype=np.uint16)
        values = np.asarray(arrays[sparse_values_key])
        if not np.array_equal(indices, expected_indices):
            raise ValueError(f"{sparse_indices_key} does not exactly match active pairs.")
        if len(values) != len(indices):
            raise ValueError(f"{sparse_values_key} does not align with its indices.")
        return indices, values
    if dense_key not in arrays:
        raise KeyError(
            f"Payload requires either sparse {sparse_values_key!r} or dense "
            f"{dense_key!r}."
        )
    dense = np.asarray(arrays[dense_key])
    if dense.ndim != 3 or dense.shape[:2] != mask.shape:
        raise ValueError(f"{dense_key} must have shape [source, source, channels].")
    return expected_indices, dense[mask]


def _encoded_bfloat16(
    arrays: Mapping[str, np.ndarray], *, float_key: str, bits_key: str
) -> np.ndarray:
    """Use GPU-preserved BF16 bits when available, otherwise encode FP values."""
    if bits_key in arrays:
        bits = np.asarray(arrays[bits_key], dtype=np.uint16)
        return np.ascontiguousarray(bits)
    if float_key not in arrays:
        raise KeyError(f"Payload requires {float_key!r} or {bits_key!r}.")
    return encode_bfloat16(np.asarray(arrays[float_key]))


def pack_full_cross_pair_storage(
    arrays: Mapping[str, np.ndarray],
    *,
    mode: str = PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS,
    cache_schema: str = AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
) -> PairStoragePayload:
    """Persist full-source singles plus PL/LP/LL values, never protein--protein.

    This is the v4 cache boundary.  The frozen trunk still sees the complete
    protein--ligand source system, but only active cross/ligand pair values
    cross the GPU-to-host boundary.  The 256-token ligand-preserving crop is
    deliberately deferred to the training dataset.
    """
    if mode != PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS:
        raise ValueError(f"Unsupported full-cross storage mode: {mode!r}.")
    if cache_schema not in AFFINITY_FULL_CROSS_SCHEMAS:
        raise ValueError(f"Unsupported full-cross cache schema: {cache_schema!r}.")
    required = {"token_mask", "chain_type"}
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Full-cross cache payload lacks fields: {missing}")
    token_mask = np.asarray(arrays["token_mask"], dtype=bool)
    chain_type = np.asarray(arrays["chain_type"], dtype=np.int64)
    length = len(token_mask)
    if chain_type.shape != (length,):
        raise ValueError("chain_type must align with full-source token_mask.")
    for name in ("s_inputs", "s_lm"):
        singles_key = f"{name}_bf16" if f"{name}_bf16" in arrays else name
        singles = np.asarray(arrays[singles_key])
        if singles.ndim != 2 or singles.shape[0] != length:
            raise ValueError(f"{name} must have shape [source_tokens, channels].")
    pair_mask = affinity_cross_pair_mask(token_mask, chain_type)
    pair_indices = _cross_pair_indices(token_mask, chain_type)
    if "z_pair_values_bf16" in arrays:
        if "z_pair_indices" not in arrays:
            raise KeyError("z_pair_values_bf16 requires z_pair_indices.")
        sparse_indices = np.asarray(arrays["z_pair_indices"], dtype=np.uint16)
        if not np.array_equal(sparse_indices, pair_indices):
            raise ValueError("z_pair_indices does not exactly match active pairs.")
        pair_bits = np.asarray(arrays["z_pair_values_bf16"], dtype=np.uint16)
        if len(pair_bits) != len(pair_indices):
            raise ValueError("z_pair_values_bf16 does not align with z_pair_indices.")
    else:
        _, pair_values = _array_or_sparse(
            arrays,
            sparse_indices_key="z_pair_indices",
            sparse_values_key="z_pair_values",
            expected_indices=pair_indices,
            dense_key="z",
            mask=pair_mask,
        )
        pair_bits = encode_bfloat16(pair_values)
    triangular_mask = pair_mask & np.triu(np.ones((length, length), dtype=bool))
    triangular_indices = _triangular_cross_indices(token_mask, chain_type)
    if "distogram_tri_logits_bf16" in arrays:
        if "distogram_tri_indices" not in arrays:
            raise KeyError("distogram_tri_logits_bf16 requires distogram_tri_indices.")
        sparse_indices = np.asarray(arrays["distogram_tri_indices"], dtype=np.uint16)
        if not np.array_equal(sparse_indices, triangular_indices):
            raise ValueError(
                "distogram_tri_indices does not exactly match triangular active pairs."
            )
        triangular_bits = np.asarray(arrays["distogram_tri_logits_bf16"], dtype=np.uint16)
        if len(triangular_bits) != len(triangular_indices):
            raise ValueError(
                "distogram_tri_logits_bf16 does not align with distogram_tri_indices."
            )
    else:
        _, triangular_logits = _array_or_sparse(
            arrays,
            sparse_indices_key="distogram_tri_indices",
            sparse_values_key="distogram_tri_logits",
            expected_indices=triangular_indices,
            dense_key="distogram_logits",
            mask=triangular_mask,
        )
        triangular_bits = encode_bfloat16(triangular_logits)
    if "distogram_logits" in arrays:
        symmetry_max_abs = verify_symmetric_distogram_features(
            np.asarray(arrays["distogram_logits"]), token_mask
        )
    elif "distogram_symmetry_max_abs" in arrays:
        symmetry_max_abs = float(np.asarray(arrays["distogram_symmetry_max_abs"]).item())
    else:
        raise KeyError(
            "GPU-sliced full-cross payload must carry distogram_symmetry_max_abs."
        )
    return PairStoragePayload(
        arrays={
            "cache_schema": np.asarray(cache_schema),
            "source_tokens": np.asarray([length], dtype=np.int32),
            "s_inputs_bf16": _encoded_bfloat16(
                arrays, float_key="s_inputs", bits_key="s_inputs_bf16"
            ),
            "s_lm_bf16": _encoded_bfloat16(
                arrays, float_key="s_lm", bits_key="s_lm_bf16"
            ),
            "token_mask": token_mask,
            "chain_type": chain_type,
            "pair_indices": pair_indices,
            "z_pair_values_bf16": pair_bits,
            "distogram_tri_indices": triangular_indices,
            "distogram_tri_logits_bf16": triangular_bits,
            "distogram_symmetry_max_abs": np.asarray(
                [symmetry_max_abs], dtype=np.float32
            ),
        },
        mode=mode,
        stored_pair_count=len(pair_indices),
        # This legacy field is retained in generic cache indexes. In v4 it is
        # explicitly the full source-square count, not a stored crop size.
        crop_pair_count=length * length,
        stored_distogram_pair_count=len(triangular_indices),
        distogram_symmetry_max_abs=symmetry_max_abs,
    )


def _full_cross_fields(
    arrays: Mapping[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not is_full_cross_payload(arrays):
        raise ValueError(
            "Expected supported full-cross cache schema, got "
            f"{cache_schema_from_payload(arrays)!r}."
        )
    required = {
        "source_tokens",
        "token_mask",
        "chain_type",
        "pair_indices",
        "distogram_tri_indices",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Full-cross cache payload lacks fields: {missing}")
    token_mask = torch.from_numpy(np.asarray(arrays["token_mask"])).bool()
    chain_type = torch.from_numpy(np.asarray(arrays["chain_type"])).long()
    source_tokens = int(np.asarray(arrays["source_tokens"]).item())
    if source_tokens != len(token_mask) or chain_type.shape != token_mask.shape:
        raise ValueError("Full-cross source token metadata is inconsistent.")
    if "s_inputs_bf16" in arrays:
        s_inputs = torch.from_numpy(decode_bfloat16(arrays["s_inputs_bf16"]))
    elif "s_inputs" in arrays:
        s_inputs = torch.from_numpy(np.asarray(arrays["s_inputs"], dtype=np.float32))
    else:
        raise KeyError("Full-cross cache payload lacks s_inputs values.")
    if "s_lm_bf16" in arrays:
        s_lm = torch.from_numpy(decode_bfloat16(arrays["s_lm_bf16"]))
    elif "s_lm" in arrays:
        s_lm = torch.from_numpy(np.asarray(arrays["s_lm"], dtype=np.float32))
    else:
        raise KeyError("Full-cross cache payload lacks s_lm values.")
    if s_inputs.shape[0] != source_tokens or s_lm.shape[0] != source_tokens:
        raise ValueError(
            "Full-cross single representations do not cover every source token."
        )
    return s_inputs, s_lm, token_mask, chain_type, source_tokens


def full_cross_pl_distogram_profile(
    arrays: Mapping[str, np.ndarray],
    *,
    token_mask: torch.Tensor | None = None,
    chain_type: torch.Tensor | None = None,
    chunk_pairs: int = 16_384,
) -> PLDistogramProfile:
    """Reduce v1 triangular PL logits without materializing any dense pair map."""
    if chunk_pairs <= 0:
        raise ValueError("chunk_pairs must be positive.")
    if cache_schema_from_payload(arrays) != AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1:
        raise ValueError(
            "Distogram target consensus requires affinity_full_cross_v1 payloads."
        )
    if (token_mask is None) != (chain_type is None):
        raise ValueError("token_mask and chain_type must be supplied together.")
    if token_mask is None:
        token_mask = torch.from_numpy(np.asarray(arrays["token_mask"])).bool()
        chain_type = torch.from_numpy(np.asarray(arrays["chain_type"])).long()
    assert chain_type is not None
    if token_mask.ndim != 1 or chain_type.shape != token_mask.shape:
        raise ValueError("Full-cross token metadata is inconsistent.")
    protein = (token_mask & (chain_type == C.ChainType.PROTEIN.value)).numpy()
    ligand = (token_mask & (chain_type == C.ChainType.LIGAND.value)).numpy()
    protein_indices = np.flatnonzero(protein)
    ligand_indices = np.flatnonzero(ligand)
    if not len(protein_indices) or not len(ligand_indices):
        raise ValueError("Full-cross training crop requires protein and ligand tokens.")
    indices = np.asarray(arrays["distogram_tri_indices"], dtype=np.int64)
    is_pl = (protein[indices[:, 0]] & ligand[indices[:, 1]]) | (
        ligand[indices[:, 0]] & protein[indices[:, 1]]
    )
    selected = np.flatnonzero(is_pl)
    if len(selected) != len(protein_indices) * len(ligand_indices):
        raise ValueError("Full-cross distogram payload does not contain every PL pair.")
    protein_position = np.full(len(token_mask), -1, dtype=np.int64)
    protein_position[protein_indices] = np.arange(len(protein_indices), dtype=np.int64)
    contact_score = torch.full((len(protein_indices),), -torch.inf)
    distance_score = torch.full((len(protein_indices),), torch.inf)
    entropy_sum = torch.zeros((len(protein_indices),), dtype=torch.float32)
    entropy_count = torch.zeros((len(protein_indices),), dtype=torch.float32)
    if "distogram_tri_logits_bf16" in arrays:
        logits_values = decode_bfloat16(arrays["distogram_tri_logits_bf16"])
    elif "distogram_tri_logits" in arrays:
        logits_values = np.asarray(arrays["distogram_tri_logits"], dtype=np.float32)
    else:
        raise KeyError("Full-cross cache payload lacks triangular distogram logits.")
    num_bins = int(logits_values.shape[-1])
    centers = torch.linspace(
        2.0 + 20.0 / (2 * num_bins),
        22.0 - 20.0 / (2 * num_bins),
        num_bins,
    )
    for start in range(0, len(selected), chunk_pairs):
        chosen = selected[start : start + chunk_pairs]
        pair_indices = indices[chosen]
        pair_logits = torch.from_numpy(logits_values[chosen])
        probabilities = pair_logits.softmax(dim=-1)
        contact = probabilities[:, centers <= 8.0].sum(dim=-1)
        distance = (probabilities * centers).sum(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1
        ) / np.log(num_bins)
        global_protein = np.where(
            protein[pair_indices[:, 0]], pair_indices[:, 0], pair_indices[:, 1]
        )
        positions = torch.from_numpy(protein_position[global_protein])
        contact_score.scatter_reduce_(
            0, positions, contact, reduce="amax", include_self=True
        )
        distance_score.scatter_reduce_(
            0, positions, distance, reduce="amin", include_self=True
        )
        entropy_sum.scatter_add_(0, positions, entropy)
        entropy_count.scatter_add_(0, positions, torch.ones_like(entropy))
    if not torch.all(entropy_count == len(ligand_indices)):
        raise ValueError("Full-cross payload does not contain one PL logit per pair.")
    return PLDistogramProfile(
        protein_max_contact_probability=contact_score,
        protein_min_expected_distance=distance_score,
        protein_mean_normalized_entropy=entropy_sum / entropy_count,
    )


def _pl_crop_scores(
    arrays: Mapping[str, np.ndarray],
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    chunk_pairs: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible tuple view of the public PL profile reducer."""
    profile = full_cross_pl_distogram_profile(
        arrays,
        token_mask=token_mask,
        chain_type=chain_type,
        chunk_pairs=chunk_pairs,
    )
    return (
        profile.protein_max_contact_probability,
        profile.protein_min_expected_distance,
        profile.protein_mean_normalized_entropy,
    )


def crop_full_cross_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
    pocket_distance_cutoff: float | None = None,
    use_entropy_tiebreak: bool = False,
    crop_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create a train crop from a full-cross record.

    Supplying ``crop_indices`` selects a target-level precomputed pocket crop;
    leaving it unset derives a per-complex crop directly from cached PL
    distogram logits.
    """
    s_inputs, s_lm, token_mask, chain_type, source_tokens = _full_cross_fields(arrays)
    if crop_indices is None:
        contact_score, distance_score, entropy_score = _pl_crop_scores(
            arrays, token_mask=token_mask, chain_type=chain_type
        )
        crop_indices = select_ligand_preserving_crop_from_scores(
            token_mask=token_mask,
            chain_type=chain_type,
            contact_score=contact_score,
            distance_score=distance_score,
            entropy_score=entropy_score if use_entropy_tiebreak else None,
            pocket_distance_cutoff=pocket_distance_cutoff,
            max_tokens=max_tokens,
            max_protein_tokens=max_protein_tokens,
        )
    else:
        crop_indices = crop_indices.long().sort().values
        if crop_indices.ndim != 1 or not len(crop_indices):
            raise ValueError("Pocket crop indices must be a non-empty vector.")
        if crop_indices[0] < 0 or crop_indices[-1] >= source_tokens:
            raise ValueError("Pocket crop indices are outside the source token range.")
        if len(torch.unique(crop_indices)) != len(crop_indices):
            raise ValueError("Pocket crop indices must be unique.")
    crop_length = len(crop_indices)
    source_to_crop = np.full(source_tokens, -1, dtype=np.int64)
    crop_numpy = crop_indices.numpy()
    source_to_crop[crop_numpy] = np.arange(crop_length, dtype=np.int64)

    pair_indices = np.asarray(arrays["pair_indices"], dtype=np.int64)
    keep_pair = (source_to_crop[pair_indices[:, 0]] >= 0) & (
        source_to_crop[pair_indices[:, 1]] >= 0
    )
    if "z_pair_values_bf16" in arrays:
        z_values = torch.from_numpy(
            decode_bfloat16(np.asarray(arrays["z_pair_values_bf16"])[keep_pair])
        )
    elif "z_pair_values" in arrays:
        z_values = torch.from_numpy(
            np.asarray(arrays["z_pair_values"], dtype=np.float32)[keep_pair]
        )
    else:
        raise KeyError("Full-cross cache payload lacks active z values.")
    z = torch.zeros((crop_length, crop_length, z_values.shape[-1]), dtype=z_values.dtype)
    kept_pairs = pair_indices[keep_pair]
    z[
        torch.from_numpy(source_to_crop[kept_pairs[:, 0]]),
        torch.from_numpy(source_to_crop[kept_pairs[:, 1]]),
    ] = z_values

    triangular_indices = np.asarray(arrays["distogram_tri_indices"], dtype=np.int64)
    keep_triangular = (source_to_crop[triangular_indices[:, 0]] >= 0) & (
        source_to_crop[triangular_indices[:, 1]] >= 0
    )
    if "distogram_tri_logits_bf16" in arrays:
        logits_values = torch.from_numpy(
            decode_bfloat16(
                np.asarray(arrays["distogram_tri_logits_bf16"])[keep_triangular]
            )
        )
    elif "distogram_tri_logits" in arrays:
        logits_values = torch.from_numpy(
            np.asarray(arrays["distogram_tri_logits"], dtype=np.float32)[keep_triangular]
        )
    else:
        raise KeyError("Full-cross cache payload lacks triangular distogram logits.")
    logits = torch.zeros(
        (crop_length, crop_length, logits_values.shape[-1]), dtype=logits_values.dtype
    )
    stored = torch.zeros((crop_length, crop_length), dtype=torch.bool)
    kept_triangular = triangular_indices[keep_triangular]
    left = torch.from_numpy(source_to_crop[kept_triangular[:, 0]])
    right = torch.from_numpy(source_to_crop[kept_triangular[:, 1]])
    logits[left, right] = logits_values
    logits[right, left] = logits_values
    stored[left, right] = True
    stored[right, left] = True
    contact, expected_distance, entropy = distogram_feature_maps(logits[None])
    distogram_features = (
        torch.stack((contact[0], expected_distance[0], entropy[0]), dim=-1)
        * stored[..., None]
    )
    return {
        "s_inputs": s_inputs[crop_indices],
        "s_lm": s_lm[crop_indices],
        "z": z,
        "token_mask": token_mask[crop_indices],
        "chain_type": chain_type[crop_indices],
        "distogram_features": distogram_features,
        "crop_indices": crop_indices,
    }


def unpack_full_cross_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    include_logits: bool = False,
) -> dict[str, torch.Tensor]:
    """Restore a full-cross record to dense source tensors for bounded audits.

    Training must use :func:`crop_full_cross_payload` instead; reconstructing
    source-sized dense pairs is intentionally only an audit/debug path.
    """
    s_inputs, s_lm, token_mask, chain_type, source_tokens = _full_cross_fields(arrays)
    pair_indices = torch.from_numpy(np.asarray(arrays["pair_indices"])).long()
    if "z_pair_values_bf16" in arrays:
        z_values = torch.from_numpy(decode_bfloat16(arrays["z_pair_values_bf16"]))
    elif "z_pair_values" in arrays:
        z_values = torch.from_numpy(np.asarray(arrays["z_pair_values"], dtype=np.float32))
    else:
        raise KeyError("Full-cross cache payload lacks active z values.")
    _validate_pair_indices(pair_indices, z_values, name="z")
    z = torch.zeros(
        (source_tokens, source_tokens, z_values.shape[-1]), dtype=z_values.dtype
    )
    z[pair_indices[:, 0], pair_indices[:, 1]] = z_values
    triangular_indices = torch.from_numpy(
        np.asarray(arrays["distogram_tri_indices"])
    ).long()
    if "distogram_tri_logits_bf16" in arrays:
        logits_values = torch.from_numpy(
            decode_bfloat16(arrays["distogram_tri_logits_bf16"])
        )
    elif "distogram_tri_logits" in arrays:
        logits_values = torch.from_numpy(
            np.asarray(arrays["distogram_tri_logits"], dtype=np.float32)
        )
    else:
        raise KeyError("Full-cross cache payload lacks triangular distogram logits.")
    _validate_pair_indices(triangular_indices, logits_values, name="distogram")
    logits = torch.zeros(
        (source_tokens, source_tokens, logits_values.shape[-1]), dtype=logits_values.dtype
    )
    left, right = triangular_indices[:, 0], triangular_indices[:, 1]
    logits[left, right] = logits_values
    logits[right, left] = logits_values
    stored = torch.zeros((source_tokens, source_tokens), dtype=torch.bool)
    stored[left, right] = True
    stored[right, left] = True
    contact, expected_distance, entropy = distogram_feature_maps(logits[None])
    result = {
        "s_inputs": s_inputs,
        "s_lm": s_lm,
        "z": z,
        "token_mask": token_mask,
        "chain_type": chain_type,
        "distogram_features": torch.stack(
            (contact[0], expected_distance[0], entropy[0]), dim=-1
        )
        * stored[..., None],
        "crop_indices": torch.arange(source_tokens),
    }
    if include_logits:
        result["distogram_logits"] = logits
    return result


def _validate_pair_indices(
    pair_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    name: str,
) -> None:
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError(f"{name} indices must have shape [pairs, 2].")
    if len(pair_indices) != len(values):
        raise ValueError(f"{name} indices and values must have equal length.")


def unpack_cross_only_bf16_tri_distogram_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    include_logits: bool = False,
) -> dict[str, torch.Tensor]:
    """Restore BF16 directed z and triangular symmetric distogram features.

    A v3 payload stores raw logits.  They are returned only when
    ``include_logits`` is set, so existing consumers that forward this dict
    straight into the head keep their exact key set.
    """
    stores_logits = "distogram_tri_logits_bf16" in arrays
    distogram_key = (
        "distogram_tri_logits_bf16" if stores_logits else "distogram_tri_values_bf16"
    )
    required = {
        "s_inputs_bf16",
        "s_lm_bf16",
        "token_mask",
        "chain_type",
        "crop_indices",
        "pair_indices",
        "z_pair_values_bf16",
        "distogram_tri_indices",
        distogram_key,
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"BF16 cross-only cache payload lacks fields: {missing}")
    token_mask = torch.from_numpy(np.asarray(arrays["token_mask"])).bool()
    length = len(token_mask)
    z_values = torch.from_numpy(decode_bfloat16(arrays["z_pair_values_bf16"]))
    z_indices = torch.from_numpy(np.asarray(arrays["pair_indices"])).long()
    _validate_pair_indices(z_indices, z_values, name="z")
    z = torch.zeros((length, length, z_values.shape[-1]), dtype=z_values.dtype)
    z[z_indices[:, 0], z_indices[:, 1]] = z_values

    distogram_values = torch.from_numpy(decode_bfloat16(arrays[distogram_key]))
    distogram_indices = torch.from_numpy(
        np.asarray(arrays["distogram_tri_indices"])
    ).long()
    _validate_pair_indices(distogram_indices, distogram_values, name="distogram")
    distogram = torch.zeros(
        (length, length, distogram_values.shape[-1]),
        dtype=distogram_values.dtype,
    )
    left = distogram_indices[:, 0]
    right = distogram_indices[:, 1]
    distogram[left, right] = distogram_values
    distogram[right, left] = distogram_values

    unpacked = {
        "s_inputs": torch.from_numpy(decode_bfloat16(arrays["s_inputs_bf16"])),
        "s_lm": torch.from_numpy(decode_bfloat16(arrays["s_lm_bf16"])),
        "z": z,
        "token_mask": token_mask,
        "chain_type": torch.from_numpy(np.asarray(arrays["chain_type"])).long(),
        "crop_indices": torch.from_numpy(np.asarray(arrays["crop_indices"])).long(),
    }
    if stores_logits:
        contact, expected_distance, entropy = distogram_feature_maps(distogram[None])
        features = torch.stack((contact[0], expected_distance[0], entropy[0]), dim=-1)
        # Unstored cells hold all-zero logits, and softmax turns those into a
        # uniform distribution rather than zero. Re-apply the stored mask so
        # the derived maps match the v2 payload exactly outside PL/LP/LL.
        stored = torch.zeros((length, length), dtype=torch.bool)
        stored[left, right] = True
        stored[right, left] = True
        features = features * stored[..., None]
        unpacked["distogram_features"] = features
        if include_logits:
            unpacked["distogram_logits"] = distogram
    else:
        unpacked["distogram_features"] = distogram
    return unpacked


def unpack_cross_only_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    include_logits: bool = False,
) -> dict[str, torch.Tensor]:
    """Restore a cross-only payload to dense crop-local tensors for the head."""
    if is_full_cross_payload(arrays):
        return unpack_full_cross_payload(arrays, include_logits=include_logits)
    if "z_pair_values_bf16" in arrays:
        return unpack_cross_only_bf16_tri_distogram_payload(
            arrays, include_logits=include_logits
        )
    required = {
        "s_inputs",
        "s_lm",
        "token_mask",
        "chain_type",
        "crop_indices",
        "pair_indices",
        "z_pair_values",
        "distogram_feature_values",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Cross-only cache payload lacks fields: {missing}")
    token_mask = torch.from_numpy(np.asarray(arrays["token_mask"])).bool()
    length = len(token_mask)
    z_values = torch.from_numpy(np.asarray(arrays["z_pair_values"]))
    distogram_values = torch.from_numpy(np.asarray(arrays["distogram_feature_values"]))
    pair_indices = torch.from_numpy(np.asarray(arrays["pair_indices"])).long()
    _validate_pair_indices(pair_indices, z_values, name="z")
    _validate_pair_indices(pair_indices, distogram_values, name="distogram")
    z = torch.zeros((length, length, z_values.shape[-1]), dtype=z_values.dtype)
    distogram_features = torch.zeros(
        (length, length, distogram_values.shape[-1]), dtype=distogram_values.dtype
    )
    z[pair_indices[:, 0], pair_indices[:, 1]] = z_values
    distogram_features[pair_indices[:, 0], pair_indices[:, 1]] = distogram_values
    return {
        "s_inputs": torch.from_numpy(np.asarray(arrays["s_inputs"])),
        "s_lm": torch.from_numpy(np.asarray(arrays["s_lm"])),
        "z": z,
        "token_mask": token_mask,
        "chain_type": torch.from_numpy(np.asarray(arrays["chain_type"])).long(),
        "distogram_features": distogram_features,
        "crop_indices": torch.from_numpy(np.asarray(arrays["crop_indices"])).long(),
    }


def payload_size_comparison(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return exact compressed NPZ sizes for every layout the payload supports.

    Logit-carrying modes are skipped when the caller only has the three derived
    conditioning maps, so a v2 payload still yields a v1/v2 comparison.
    """
    sizes: dict[str, int] = {}
    for mode in PAIR_STORAGE_MODES:
        try:
            packed = pack_pair_storage(arrays, mode=mode)
        except KeyError:
            continue
        sizes[mode] = len(serialize_npz(packed.arrays))
    return sizes


def cropped_payload_size_comparison(
    cropped: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Compare sparse-layout sizes directly from a crop-local payload."""
    sizes: dict[str, int] = {}
    for mode in (
        PAIR_STORAGE_CROSS_ONLY_SPARSE,
        PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
        PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
    ):
        try:
            packed = pack_cropped_pair_storage(cropped, mode=mode)
        except KeyError:
            continue
        sizes[mode] = len(serialize_npz(packed.arrays))
    return sizes
