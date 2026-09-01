"""On-the-fly affinity prediction from one frozen K-Fold trunk pass."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

import kfold.constants as C
from kfold.model.modules.affinity_pairformer import AffinityPairformer
from kfold.training.affinity.crop import (
    distogram_feature_maps,
    protein_ligand_distogram_profile,
    select_pocket_annotation_crop,
)

AFFINITY_QUERY_WINDOW_CONTRACT_V1 = "affinity_per_query_distogram_window_v1"


@dataclass(frozen=True, kw_only=True)
class PerQueryAffinityConfig:
    """Production contract matching the cache-backed CASP16 scorer."""

    max_tokens: int = 256
    max_protein_tokens: int = 200
    neighborhood_size: int = 10
    cache_compatible_bfloat16: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.max_protein_tokens <= 0:
            raise ValueError("Affinity crop token limits must be positive.")
        if self.max_protein_tokens > self.max_tokens:
            raise ValueError("Protein token limit cannot exceed total token limit.")
        if self.neighborhood_size <= 0:
            raise ValueError("Affinity neighborhood size must be positive.")


@dataclass(frozen=True, kw_only=True)
class PerQueryAffinityInputs:
    """One cropped, batched input for :class:`AffinityPairformer`."""

    s_inputs: torch.Tensor
    s_lm: torch.Tensor
    z: torch.Tensor
    distogram_features: torch.Tensor
    token_mask: torch.Tensor
    protein_mask: torch.Tensor
    ligand_mask: torch.Tensor
    crop_indices: torch.Tensor

    def head_kwargs(self) -> dict[str, torch.Tensor]:
        """Return only tensors consumed by the affinity head."""
        return {
            "s_inputs": self.s_inputs,
            "s_lm": self.s_lm,
            "z": self.z,
            "distogram_features": self.distogram_features,
            "token_mask": self.token_mask,
            "protein_mask": self.protein_mask,
            "ligand_mask": self.ligand_mask,
        }


def sha256_file(path: str | Path) -> str:
    """Return a checkpoint digest without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_affinity_head(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    num_blocks: int = 6,
) -> AffinityPairformer:
    """Load the frozen affinity readout used by training and benchmarks."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Affinity checkpoint does not contain a state dictionary.")
    model_state = {
        key.removeprefix("model.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    if not model_state:
        raise ValueError(
            "Head checkpoint has no AffinityRankingModule model.* state dict."
        )
    model = AffinityPairformer(AffinityPairformer.Config(num_blocks=num_blocks))
    model.load_state_dict(model_state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def _unbatch_one(name: str, value: torch.Tensor, unbatched_ndim: int) -> torch.Tensor:
    if value.ndim == unbatched_ndim:
        return value
    if value.ndim == unbatched_ndim + 1 and value.shape[0] == 1:
        return value[0]
    raise ValueError(
        f"{name} must be unbatched or have batch size one, got {tuple(value.shape)}."
    )


def _cache_compatible_float(
    value: torch.Tensor, *, cache_compatible_bfloat16: bool
) -> torch.Tensor:
    """Match the BF16 storage boundary used by the submitted CASP16 cache."""
    if cache_compatible_bfloat16:
        return value.to(torch.bfloat16).float()
    return value.float()


def _affinity_pair_mask(
    token_mask: torch.Tensor, chain_type: torch.Tensor
) -> torch.Tensor:
    protein = token_mask & (chain_type == C.ChainType.PROTEIN.value)
    ligand = token_mask & (chain_type == C.ChainType.LIGAND.value)
    return (
        (protein[:, None] & ligand[None, :])
        | (ligand[:, None] & protein[None, :])
        | (ligand[:, None] & ligand[None, :])
    )


def build_per_query_affinity_inputs(
    *,
    s_inputs: torch.Tensor,
    s_lm: torch.Tensor,
    z: torch.Tensor,
    distogram_logits: torch.Tensor,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    config: PerQueryAffinityConfig | None = None,
) -> PerQueryAffinityInputs:
    """Crop one full-query trunk output exactly as the submitted CASP16 scorer.

    The full protein--ligand query first passes through the frozen trunk and
    distogram head.  Its PL expected-distance profile then orders same-chain
    10-token protein windows.  Every ligand token is retained, and windows are
    accepted until the 256-total/200-protein budget would overflow.
    """
    config = config or PerQueryAffinityConfig()
    s_inputs = _unbatch_one("s_inputs", s_inputs, 2)
    s_lm = _unbatch_one("s_lm", s_lm, 2)
    z = _unbatch_one("z", z, 3)
    distogram_logits = _unbatch_one("distogram_logits", distogram_logits, 3)
    token_mask = _unbatch_one("token_mask", token_mask, 1).bool()
    chain_type = _unbatch_one("chain_type", chain_type, 1).long()
    length = len(token_mask)
    if s_inputs.shape[0] != length or s_lm.shape[0] != length:
        raise ValueError("Single representations must align with token metadata.")
    if z.shape[:2] != (length, length):
        raise ValueError("Pair representation must have shape [L, L, C].")
    if distogram_logits.shape[:2] != (length, length):
        raise ValueError("Distogram logits must have shape [L, L, bins].")

    s_inputs = _cache_compatible_float(
        s_inputs, cache_compatible_bfloat16=config.cache_compatible_bfloat16
    )
    s_lm = _cache_compatible_float(
        s_lm, cache_compatible_bfloat16=config.cache_compatible_bfloat16
    )
    z = _cache_compatible_float(
        z, cache_compatible_bfloat16=config.cache_compatible_bfloat16
    )
    distogram_logits = _cache_compatible_float(
        distogram_logits,
        cache_compatible_bfloat16=config.cache_compatible_bfloat16,
    )

    profile = protein_ligand_distogram_profile(
        logits=distogram_logits,
        token_mask=token_mask,
        chain_type=chain_type,
    )
    crop_indices = select_pocket_annotation_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        protein_min_distance=profile.protein_min_expected_distance,
        max_tokens=config.max_tokens,
        max_protein_tokens=config.max_protein_tokens,
        neighborhood_size=config.neighborhood_size,
        require_contiguous_monomer=True,
    )
    pair = crop_indices[:, None], crop_indices[None, :]
    crop_token_mask = token_mask[crop_indices]
    crop_chain_type = chain_type[crop_indices]
    active_pairs = _affinity_pair_mask(crop_token_mask, crop_chain_type)
    contact, expected_distance, entropy = distogram_feature_maps(distogram_logits[None])
    distogram_features = torch.stack(
        (
            contact[0][pair],
            expected_distance[0][pair],
            entropy[0][pair],
        ),
        dim=-1,
    )
    distogram_features = distogram_features * active_pairs[..., None]
    cropped_z = z[pair] * active_pairs[..., None]
    protein_mask = crop_token_mask & (crop_chain_type == C.ChainType.PROTEIN.value)
    ligand_mask = crop_token_mask & (crop_chain_type == C.ChainType.LIGAND.value)
    return PerQueryAffinityInputs(
        s_inputs=s_inputs[crop_indices][None],
        s_lm=s_lm[crop_indices][None],
        z=cropped_z[None],
        distogram_features=distogram_features[None],
        token_mask=crop_token_mask[None],
        protein_mask=protein_mask[None],
        ligand_mask=ligand_mask[None],
        crop_indices=crop_indices,
    )


class PerQueryAffinityPredictor(torch.nn.Module):
    """Attach a trained affinity head to structure inference without a cache."""

    def __init__(
        self,
        head: AffinityPairformer,
        *,
        checkpoint_sha256: str,
        config: PerQueryAffinityConfig | None = None,
    ) -> None:
        super().__init__()
        if len(checkpoint_sha256) != 64:
            raise ValueError("Affinity checkpoint SHA-256 must contain 64 hex digits.")
        self.head = head
        self.checkpoint_sha256 = checkpoint_sha256
        self.config = config or PerQueryAffinityConfig()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        config: PerQueryAffinityConfig | None = None,
        num_blocks: int = 6,
    ) -> PerQueryAffinityPredictor:
        return cls(
            load_affinity_head(path, device=device, num_blocks=num_blocks),
            checkpoint_sha256=sha256_file(path),
            config=config,
        )

    def forward(
        self,
        *,
        s_inputs: torch.Tensor,
        s_lm: torch.Tensor,
        z: torch.Tensor,
        distogram_logits: torch.Tensor,
        token_mask: torch.Tensor,
        chain_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device_type = z.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            inputs = build_per_query_affinity_inputs(
                s_inputs=s_inputs,
                s_lm=s_lm,
                z=z,
                distogram_logits=distogram_logits,
                token_mask=token_mask,
                chain_type=chain_type,
                config=self.config,
            )
            prediction = self.head(**inputs.head_kwargs()).float()
        return {
            "p_activity": prediction,
            "crop_indices": inputs.crop_indices,
            "crop_token_count": torch.tensor(
                len(inputs.crop_indices), device=inputs.crop_indices.device
            ),
            "protein_token_count": inputs.protein_mask.sum(),
            "ligand_token_count": inputs.ligand_mask.sum(),
        }


def affinity_prediction_record(
    output: Mapping[str, torch.Tensor],
    predictor: PerQueryAffinityPredictor,
) -> dict[str, object]:
    """Create the stable JSON payload written by both inference entrypoints."""
    required = {
        "p_activity",
        "crop_indices",
        "crop_token_count",
        "protein_token_count",
        "ligand_token_count",
    }
    missing = sorted(required - set(output))
    if missing:
        raise KeyError(f"Affinity inference output lacks fields: {missing}")
    return {
        "schema_version": "kfold_affinity_prediction_v1",
        "prediction_p_activity": float(output["p_activity"].reshape(-1)[0].item()),
        "crop_contract": AFFINITY_QUERY_WINDOW_CONTRACT_V1,
        "crop_indices": output["crop_indices"].cpu().tolist(),
        "crop_token_count": int(output["crop_token_count"].item()),
        "protein_token_count": int(output["protein_token_count"].item()),
        "ligand_token_count": int(output["ligand_token_count"].item()),
        "affinity_checkpoint_sha256": predictor.checkpoint_sha256,
        "max_tokens": predictor.config.max_tokens,
        "max_protein_tokens": predictor.config.max_protein_tokens,
        "neighborhood_size": predictor.config.neighborhood_size,
        "cache_compatible_bfloat16": predictor.config.cache_compatible_bfloat16,
    }


def attach_affinity_prediction(
    model_output: dict[str, dict[str, torch.Tensor]],
    *,
    predictor: PerQueryAffinityPredictor,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
) -> None:
    """Consume returned trunk embeddings and attach one per-query prediction."""
    try:
        trunk = model_output.pop("trunk")
        distogram_logits = model_output["distogram"]["distogram"]
    except KeyError as exc:
        raise KeyError(
            "Affinity inference requires returned trunk and distogram outputs."
        ) from exc
    model_output["affinity"] = predictor(
        s_inputs=trunk["s_inputs"],
        s_lm=trunk["s_lm"],
        z=trunk["z"],
        distogram_logits=distogram_logits,
        token_mask=token_mask,
        chain_type=chain_type,
    )
