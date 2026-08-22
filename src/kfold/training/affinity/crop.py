"""Prediction-only ligand-preserving crops for affinity features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import kfold.constants as C

DEFAULT_AFFINITY_SHAPE_BUCKETS = (64, 96, 128, 160, 192, 224, 256)


@dataclass(frozen=True, kw_only=True)
class ProteinLigandDistogramProfile:
    """Coordinate-free query pocket statistics from one 80k distogram."""

    protein_min_expected_distance: torch.Tensor
    protein_mean_normalized_entropy: torch.Tensor
    pocket_hlp_15a: float | None
    pocket_residue_count: int


@dataclass(frozen=True, kw_only=True)
class PocketWindowTrace:
    """Accepted Boltz windows and the resulting deterministic crop."""

    indices: torch.Tensor
    protein_residue_order: torch.Tensor
    accepted_windows: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class QueryAdaptiveDeltaSelection:
    """Query-prioritized protein windows outside a canonical target crop."""

    extra_protein_source_indices: torch.Tensor
    query_residue_order: torch.Tensor
    query_window_order: torch.Tensor
    accepted_query_windows: torch.Tensor


def head_crop_token_count(
    *,
    protein_tokens: int,
    ligand_tokens: int,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
) -> int:
    """Return the exact token count before batch-local padding.

    The affinity crop retains every ligand token and then selects up to the
    protein/contact budget.  This count is metadata-only and matches
    :func:`select_ligand_preserving_crop` whenever the cached token accounting
    is valid.
    """
    if protein_tokens <= 0 or ligand_tokens <= 0:
        raise ValueError("Protein and ligand token counts must be positive.")
    if max_tokens <= 0 or max_protein_tokens <= 0:
        raise ValueError("Affinity crop limits must be positive.")
    if ligand_tokens >= max_tokens:
        raise ValueError("Ligand token count exhausts the crop budget.")
    return ligand_tokens + min(
        protein_tokens,
        max_protein_tokens,
        max_tokens - ligand_tokens,
    )


def shape_bucket_for_tokens(
    token_count: int,
    *,
    buckets: tuple[int, ...] = DEFAULT_AFFINITY_SHAPE_BUCKETS,
) -> int:
    """Return the smallest static tensor bucket that contains ``token_count``."""
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    ordered = tuple(sorted(set(buckets)))
    if not ordered or ordered[0] <= 0:
        raise ValueError("shape buckets must be positive.")
    for bucket in ordered:
        if token_count <= bucket:
            return bucket
    raise ValueError(
        f"Affinity crop has {token_count} tokens, above largest shape bucket "
        f"{ordered[-1]}."
    )


def distogram_feature_maps(
    logits: torch.Tensor,
    *,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contact probability, expected distance, and normalized entropy."""
    if logits.ndim != 4:
        raise ValueError("Distogram logits must have shape [B, L, L, bins].")
    num_bins = logits.shape[-1]
    centers = torch.linspace(
        min_dist + (max_dist - min_dist) / (2 * num_bins),
        max_dist - (max_dist - min_dist) / (2 * num_bins),
        num_bins,
        dtype=logits.dtype,
        device=logits.device,
    )
    probabilities = logits.softmax(dim=-1)
    contact = probabilities[..., centers <= contact_cutoff].sum(dim=-1)
    expected_distance = (probabilities * centers).sum(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = entropy / torch.log(torch.tensor(float(num_bins), device=logits.device))
    return contact, expected_distance, entropy


def distogram_pair_features(
    logits: torch.Tensor,
    *,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    contact_cutoff: float = 8.0,
) -> torch.Tensor:
    """Return three conditioning values for an arbitrary list of pair logits."""
    if logits.ndim != 2:
        raise ValueError("Pair distogram logits must have shape [pairs, bins].")
    num_bins = logits.shape[-1]
    values = logits.float()
    centers = torch.linspace(
        min_dist + (max_dist - min_dist) / (2 * num_bins),
        max_dist - (max_dist - min_dist) / (2 * num_bins),
        num_bins,
        dtype=values.dtype,
        device=values.device,
    )
    probabilities = values.softmax(dim=-1)
    contact = probabilities[:, centers <= contact_cutoff].sum(dim=-1)
    expected = (probabilities * centers).sum(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        dim=-1
    ) / torch.log(torch.tensor(float(num_bins), dtype=values.dtype, device=values.device))
    return torch.stack((contact, expected, entropy), dim=-1)


def protein_ligand_distogram_profile(
    *,
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    distance_cutoff: float = 15.0,
) -> ProteinLigandDistogramProfile:
    """Reduce one full 80k distogram to TerraBind-style PL pocket statistics."""
    if logits.ndim != 3 or logits.shape[:2] != (len(token_mask), len(token_mask)):
        raise ValueError("One-system distogram logits must have shape [L, L, bins].")
    if token_mask.ndim != 1 or chain_type.shape != token_mask.shape:
        raise ValueError("Token mask and chain type must be matching vectors.")
    if distance_cutoff <= 0:
        raise ValueError("Pocket distance cutoff must be positive.")
    valid = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    protein = valid[chain_type[valid] == C.ChainType.PROTEIN.value]
    ligand = valid[chain_type[valid] == C.ChainType.LIGAND.value]
    if not len(protein) or not len(ligand):
        raise ValueError("80k distogram profile requires protein and ligand tokens.")
    values = logits[protein][:, ligand].float()
    num_bins = values.shape[-1]
    centers = torch.linspace(
        2.0 + 20.0 / (2 * num_bins),
        22.0 - 20.0 / (2 * num_bins),
        num_bins,
        dtype=values.dtype,
        device=values.device,
    )
    probabilities = values.softmax(dim=-1)
    pl_distance = (probabilities * centers).sum(dim=-1)
    pl_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        dim=-1
    ) / torch.log(torch.tensor(float(num_bins), dtype=values.dtype, device=values.device))
    protein_distance = pl_distance.amin(dim=-1)
    protein_entropy = pl_entropy.mean(dim=-1)
    pocket = protein_distance < distance_cutoff
    pocket_count = int(pocket.sum().item())
    hlp = float(pl_entropy[pocket].mean().item()) if pocket_count else None
    return ProteinLigandDistogramProfile(
        protein_min_expected_distance=protein_distance,
        protein_mean_normalized_entropy=protein_entropy,
        pocket_hlp_15a=hlp,
        pocket_residue_count=pocket_count,
    )


def select_ligand_preserving_crop(
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    contact_probability: torch.Tensor,
    expected_distance: torch.Tensor,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
) -> torch.Tensor:
    """Choose a stable predicted-contact crop without target coordinates.

    All ligand tokens are retained. Protein tokens are ranked by their highest
    predicted contact probability to any ligand atom, then by their lowest
    expected distance, then by original token index.
    """
    if token_mask.ndim != 1 or chain_type.ndim != 1:
        raise ValueError("token_mask and chain_type must be one-dimensional.")
    if len(token_mask) != len(chain_type):
        raise ValueError("token_mask and chain_type must have equal length.")
    if contact_probability.shape != (len(token_mask), len(token_mask)):
        raise ValueError("contact_probability must have shape [L, L].")
    if expected_distance.shape != contact_probability.shape:
        raise ValueError("expected_distance must match contact_probability.")

    valid = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    ligand = valid[chain_type[valid] == C.ChainType.LIGAND.value]
    protein = valid[chain_type[valid] == C.ChainType.PROTEIN.value]
    if len(ligand) == 0 or len(protein) == 0:
        raise ValueError("Affinity crop requires at least one protein and ligand token.")
    if len(ligand) >= max_tokens:
        raise ValueError("Ligand token count exhausts the crop budget.")

    contact_score = contact_probability[protein][:, ligand].amax(dim=-1)
    distance_score = expected_distance[protein][:, ligand].amin(dim=-1)
    return select_ligand_preserving_crop_from_scores(
        token_mask=token_mask,
        chain_type=chain_type,
        contact_score=contact_score,
        distance_score=distance_score,
        max_tokens=max_tokens,
        max_protein_tokens=max_protein_tokens,
    )


def select_ligand_preserving_crop_from_scores(
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    contact_score: torch.Tensor,
    distance_score: torch.Tensor,
    entropy_score: torch.Tensor | None = None,
    pocket_distance_cutoff: float | None = None,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
) -> torch.Tensor:
    """Select the crop from per-protein PL scores without materializing PP.

    ``contact_score`` and ``distance_score`` are ordered by the valid protein
    token positions.  This is the train-time counterpart of
    :func:`select_ligand_preserving_crop`: a full-cross cache can derive the
    scores from sparse PL distogram values without ever decoding a dense
    source-sized pair matrix.
    """
    if token_mask.ndim != 1 or chain_type.ndim != 1:
        raise ValueError("token_mask and chain_type must be one-dimensional.")
    if len(token_mask) != len(chain_type):
        raise ValueError("token_mask and chain_type must have equal length.")
    valid = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    ligand = valid[chain_type[valid] == C.ChainType.LIGAND.value]
    protein = valid[chain_type[valid] == C.ChainType.PROTEIN.value]
    if len(ligand) == 0 or len(protein) == 0:
        raise ValueError("Affinity crop requires at least one protein and ligand token.")
    if len(ligand) >= max_tokens:
        raise ValueError("Ligand token count exhausts the crop budget.")
    if contact_score.shape != (len(protein),):
        raise ValueError("contact_score must contain one value per protein token.")
    if distance_score.shape != contact_score.shape:
        raise ValueError("distance_score must match contact_score.")
    if entropy_score is not None and entropy_score.shape != contact_score.shape:
        raise ValueError("entropy_score must match contact_score.")
    if pocket_distance_cutoff is not None and pocket_distance_cutoff <= 0:
        raise ValueError("pocket_distance_cutoff must be positive.")
    protein_budget = min(max_protein_tokens, max_tokens - len(ligand), len(protein))
    candidates = torch.arange(len(protein))
    if pocket_distance_cutoff is not None:
        pocket = candidates[distance_score < pocket_distance_cutoff]
        candidates = pocket if len(pocket) else torch.argmin(distance_score)[None]
    # Stable lexicographic sort. The legacy path preserves contact-first
    # ranking. The distogram-pocket path follows expected distance and uses
    # lower normalized entropy as its confidence tie-break.
    order = candidates[torch.argsort(protein[candidates], stable=True)]
    if pocket_distance_cutoff is None:
        order = order[torch.argsort(distance_score[order], stable=True)]
        order = order[torch.argsort(-contact_score[order], stable=True)]
    else:
        order = order[torch.argsort(-contact_score[order], stable=True)]
        if entropy_score is not None:
            order = order[torch.argsort(entropy_score[order], stable=True)]
        order = order[torch.argsort(distance_score[order], stable=True)]
    selected = torch.cat((ligand, protein[order[:protein_budget]])).sort().values
    return selected


def select_pocket_annotation_crop(
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    protein_min_distance: torch.Tensor,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
    neighborhood_size: int = 10,
    require_contiguous_monomer: bool = False,
) -> torch.Tensor:
    """Crop around one target-level pocket annotation as in Boltz-2 Algorithm 3.

    ``protein_min_distance`` is ordered by valid protein token, comes from the
    target-level precomputation, and is therefore shared across every ligand
    complex for that target.  ``neighborhood_size`` is the minimum number of
    contiguous protein tokens around a selected pocket residue, matching
    Boltz-2 Algorithm 3.  Every ligand token remains in the crop.
    """
    return trace_pocket_annotation_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        protein_min_distance=protein_min_distance,
        max_tokens=max_tokens,
        max_protein_tokens=max_protein_tokens,
        neighborhood_size=neighborhood_size,
        require_contiguous_monomer=require_contiguous_monomer,
    ).indices


def trace_pocket_annotation_crop(
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    protein_min_distance: torch.Tensor,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
    neighborhood_size: int = 10,
    require_contiguous_monomer: bool = False,
) -> PocketWindowTrace:
    """Return Algorithm-3 crop indices together with accepted window order."""
    if neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be positive.")
    valid = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    ligand = valid[chain_type[valid] == C.ChainType.LIGAND.value]
    protein = valid[chain_type[valid] == C.ChainType.PROTEIN.value]
    if len(ligand) == 0 or len(protein) == 0:
        raise ValueError("Affinity crop requires at least one protein and ligand token.")
    if len(ligand) >= max_tokens:
        raise ValueError("Ligand token count exhausts the crop budget.")
    if protein_min_distance.shape != (len(protein),):
        raise ValueError(
            "Pocket annotation must contain one minimum distance per protein token."
        )
    if require_contiguous_monomer:
        if len(protein) < neighborhood_size:
            raise ValueError(
                "Pocket crop requires at least one complete protein neighborhood."
            )
        if len(protein) > 1 and not torch.all(protein[1:] == protein[:-1] + 1):
            raise ValueError(
                "The v1 distogram consensus contract requires one contiguous "
                "monomer protein token block."
            )

    residue_order = torch.argsort(protein_min_distance, stable=True)
    selected = {int(index) for index in ligand.tolist()}
    selected_protein: set[int] = set()
    accepted_windows: list[torch.Tensor] = []
    for position in residue_order.tolist():
        window_size = min(neighborhood_size, len(protein))
        left = max(0, int(position) - window_size // 2)
        left = min(left, len(protein) - window_size)
        right = left + window_size - 1
        window = {int(index) for index in protein[left : right + 1].tolist()}
        new_protein = window - selected_protein
        if not new_protein:
            continue
        if (
            len(selected) + len(new_protein) > max_tokens
            or len(selected_protein) + len(new_protein) > max_protein_tokens
        ):
            break
        selected.update(new_protein)
        selected_protein.update(new_protein)
        accepted_windows.append(protein[left : right + 1].clone())
    if not selected_protein:
        raise ValueError("Pocket crop cannot retain any protein neighborhood.")
    return PocketWindowTrace(
        indices=torch.tensor(sorted(selected), dtype=torch.long),
        protein_residue_order=residue_order,
        accepted_windows=torch.stack(accepted_windows),
    )


def select_query_adaptive_delta(
    *,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    base_crop_indices: torch.Tensor,
    query_min_expected_distance: torch.Tensor,
    query_mean_normalized_entropy: torch.Tensor,
    max_tokens: int = 256,
    max_protein_tokens: int = 200,
    neighborhood_size: int = 10,
    adaptive_fraction: float = 0.2,
    distance_cutoff: float = 15.0,
) -> QueryAdaptiveDeltaSelection:
    """Reserve query 15 A windows outside the canonical crop without changing it."""
    if not 0 < adaptive_fraction <= 1 or distance_cutoff <= 0:
        raise ValueError("Adaptive delta limits must be positive and bounded.")
    valid = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    ligand = valid[chain_type[valid] == C.ChainType.LIGAND.value]
    protein = valid[chain_type[valid] == C.ChainType.PROTEIN.value]
    if len(protein) < neighborhood_size or not len(ligand):
        raise ValueError("Adaptive delta requires ligand and one protein window.")
    if len(protein) > 1 and not torch.all(protein[1:] == protein[:-1] + 1):
        raise ValueError("Adaptive delta requires one contiguous monomer protein.")
    if query_min_expected_distance.shape != (len(protein),):
        raise ValueError("Adaptive query distance must align with protein tokens.")
    if query_mean_normalized_entropy.shape != (len(protein),):
        raise ValueError("Adaptive query entropy must align with protein tokens.")
    candidates = torch.nonzero(
        query_min_expected_distance < distance_cutoff, as_tuple=False
    ).squeeze(-1)
    order = candidates[torch.argsort(candidates, stable=True)]
    order = order[torch.argsort(query_mean_normalized_entropy[order], stable=True)]
    order = order[torch.argsort(query_min_expected_distance[order], stable=True)]
    source_residue_order = protein[order]
    windows: list[torch.Tensor] = []
    for position in order.tolist():
        left = max(0, int(position) - neighborhood_size // 2)
        left = min(left, len(protein) - neighborhood_size)
        windows.append(protein[left : left + neighborhood_size].clone())
    window_order = (
        torch.stack(windows)
        if windows
        else torch.empty((0, neighborhood_size), dtype=torch.long)
    )
    protein_budget = min(max_protein_tokens, max_tokens - len(ligand), len(protein))
    quota = min(40, int(protein_budget * adaptive_fraction))
    base = {int(index) for index in base_crop_indices.tolist()}
    extra: set[int] = set()
    accepted: list[torch.Tensor] = []
    for window in windows:
        new = {int(index) for index in window.tolist()} - base - extra
        if not new:
            continue
        if len(extra) + len(new) > quota:
            break
        extra.update(new)
        accepted.append(window)
    return QueryAdaptiveDeltaSelection(
        extra_protein_source_indices=torch.tensor(sorted(extra), dtype=torch.long),
        query_residue_order=source_residue_order,
        query_window_order=window_order,
        accepted_query_windows=(
            torch.stack(accepted)
            if accepted
            else torch.empty((0, neighborhood_size), dtype=torch.long)
        ),
    )


def query_adaptive_final_crop_indices(
    *,
    base_crop_indices: torch.Tensor,
    base_chain_type: torch.Tensor,
    extra_protein_source_indices: torch.Tensor,
    target_consensus_window_order: torch.Tensor,
    accepted_query_windows: torch.Tensor,
    max_tail_tokens: int = 40,
) -> torch.Tensor:
    """Replay the direct whole-window 80/20 replacement from stored traces."""
    if base_crop_indices.ndim != 1 or base_chain_type.shape != base_crop_indices.shape:
        raise ValueError("Adaptive base crop indices and chain types must align.")
    if max_tail_tokens <= 0:
        raise ValueError("Adaptive target-tail limit must be positive.")
    base_sources = {int(value) for value in base_crop_indices.tolist()}
    base_protein = {
        int(source)
        for source, chain in zip(
            base_crop_indices.tolist(), base_chain_type.tolist(), strict=True
        )
        if int(chain) == C.ChainType.PROTEIN.value
    }
    extra = {int(value) for value in extra_protein_source_indices.tolist()}
    if base_sources.intersection(extra):
        raise ValueError("Adaptive extra proteins must be outside the canonical crop.")
    protected_query = {
        int(value)
        for value in accepted_query_windows.reshape(-1).tolist()
        if int(value) in base_protein
    }
    contributions: list[set[int]] = []
    seen: set[int] = set()
    for window in target_consensus_window_order:
        contribution = {
            int(value)
            for value in window.tolist()
            if int(value) in base_protein and int(value) not in seen
        }
        if contribution:
            seen.update(contribution)
            contributions.append(contribution)
    if seen != base_protein:
        raise ValueError("Target window trace does not reconstruct the canonical crop.")
    removed: set[int] = set()
    needed = len(extra)
    for contribution in reversed(contributions):
        candidates = contribution - protected_query
        if not candidates or len(removed) + len(candidates) > max_tail_tokens:
            continue
        removed.update(candidates)
        if len(removed) >= needed:
            break
    if len(removed) < needed:
        raise ValueError("Whole target windows cannot free the adaptive quota.")
    final_sources = sorted((base_sources - removed) | extra)
    return torch.tensor(final_sources, dtype=torch.long)


def crop_distogram_features(
    *,
    s_inputs: torch.Tensor,
    s_lm: torch.Tensor,
    z: torch.Tensor,
    token_mask: torch.Tensor,
    chain_type: torch.Tensor,
    crop_indices: torch.Tensor,
    contact_probability: torch.Tensor,
    expected_distance: torch.Tensor,
    entropy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Slice final features and retain only three Pairformer conditioning maps."""
    if s_inputs.ndim != 2 or s_lm.ndim != 2 or z.ndim != 3:
        raise ValueError("Expected unbatched [L,C], [L,C], and [L,L,C] features.")
    pair = crop_indices[:, None], crop_indices[None, :]
    distogram_features = torch.stack(
        (
            contact_probability[pair],
            expected_distance[pair],
            entropy[pair],
        ),
        dim=-1,
    )
    return {
        "s_inputs": s_inputs[crop_indices],
        "s_lm": s_lm[crop_indices],
        "z": z[pair],
        "token_mask": token_mask[crop_indices],
        "chain_type": chain_type[crop_indices],
        "distogram_features": distogram_features,
        "crop_indices": crop_indices,
    }


def ligand_protein_entropy(
    entropy: torch.Tensor,
    chain_type: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Average normalized distogram entropy over valid protein--ligand pairs."""
    protein = token_mask & (chain_type == C.ChainType.PROTEIN.value)
    ligand = token_mask & (chain_type == C.ChainType.LIGAND.value)
    mask = (protein[:, None] & ligand[None, :]) | (ligand[:, None] & protein[None, :])
    if not mask.any():
        raise ValueError("No valid protein--ligand pairs for entropy calculation.")
    return entropy[mask].mean()
