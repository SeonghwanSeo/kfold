"""Model validation metrics for structure prediction tasks."""

from collections.abc import Sequence

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.training.dataset.utils.permutation import get_aligned_true_coords
from kfold.utils.geometry.rigid_align import rigid_align


# ============================================================
# Symmetry correction for accurate metric computation
# ============================================================
def permute_label_coordinates(
    f_input: FoldingInput,
    pred_coords: torch.Tensor,
    full_struct_list: list[dict],
    symmetry_correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the best matching true coordinates to the predicted coordinates.
    Chain permutation and atom swaps.

    Parameters
    ----------
    f_input : FoldingInput
        Input features
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [B, Nsample, Natom, 3]
    full_struct_list : list[dict]
        Full structure information for each sample in the batch
    symmetry_correction : bool
        Whether to apply symmetry correction

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The true coordinates after permutation and the corresponding mask
    """
    B, Nsample, _, _ = pred_coords.shape
    if symmetry_correction:
        aligned_true_coords_list: list[torch.Tensor] = []
        resolved_mask_list: list[torch.Tensor] = []
        for batch_i in range(B):
            coords_i = pred_coords[batch_i]  # [Nsample, Natom, 3]
            symmetry_dict = full_struct_list[batch_i]["symmetry"]
            true_coords_aligned, mask = get_aligned_true_coords(
                coords_i,  # [Nsample, Natom, 3]
                f_input,
                symmetry_dict,
                index_batch=batch_i,
            )  # [Nsample, Natom, 3], [Nsample, Natom]
            aligned_true_coords_list.append(true_coords_aligned)
            resolved_mask_list.append(mask)
        true_coords_aligned = torch.stack(
            aligned_true_coords_list, dim=0
        )  # [B, Nsample, Natom, 3]
        mask = torch.stack(resolved_mask_list, dim=0)  # [B, Nsample, Natom]

    else:
        true_coords = f_input.atom.label_coords  # [B, Natom, 3]
        mask = f_input.atom.resolved_mask  # [B, Natom]
        # Expand to match pred_coords shape
        true_coords = true_coords[:, None, :, :].expand(
            -1, Nsample, -1, -1
        )  # [B, Nsample, Natom, 3]
        mask = mask[:, None, :].expand(-1, Nsample, -1)  # [B, Nsample, Natom]
        # Align true coordinates to predicted coordinates
        true_coords_aligned = rigid_align(true_coords, pred_coords, mask)

    return true_coords_aligned, mask


# ============================================================
# Metric computations
# ============================================================
def compute_rmsd(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    align: bool = False,
):
    """Compute RMSD between predicted and true coordinates.

    Parameters
    ----------
    a : torch.Tensor
        Coordinates 1, shape (*, Natom, 3)
    b : torch.Tensor
        Coordinates 2, shape (*, Natom, 3)
    mask : torch.Tensor | None
        Optional mask for valid atoms, shape (*, Natom)
    align : bool
        Whether to rigidly align a to b before RMSD computation
    """
    if align:
        a = rigid_align(a, b, mask)
    diff = (a - b).pow(2).sum(-1)  # (*, Natom)
    if mask is not None:
        assert mask.any(), "At least one atom must be valid in the mask."
        return (diff * mask).sum(-1).div(mask.sum(-1)).sqrt()
    else:
        return diff.mean(-1).sqrt()


def compute_pair_lddt(
    pdist_pred: torch.Tensor,
    pdist_true: torch.Tensor,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
):
    """Compute the lddt score from predicted and true distances.

    Parameters
    ----------
    d_predicted : torch.Tensor
        Predicted distances, shape (*, Natom, Natom)
    d_true : torch.Tensor
        Ground truth distances, shape (*, Natom, Natom)
    thresholds : Sequence[float]
        Distance error thresholds for lddt calculation
    """
    dtype = pdist_pred.dtype
    error = torch.abs(pdist_true - pdist_pred)
    scores = torch.zeros_like(error, dtype=dtype)
    for threshold in thresholds:
        scores += (error < threshold).to(dtype)
    return scores / len(thresholds)


def compute_validation_metric_singles(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    is_protein: torch.Tensor,
    is_dna: torch.Tensor,
    is_rna: torch.Tensor,
    is_ligand: torch.Tensor,
    asym_id: torch.Tensor,
    atom_mask: torch.Tensor,
):
    """Compute the validation metrics for different modalities.

    Parameters
    ----------
    true_coords : torch.Tensor
        Ground truth atom coordinates, Shape of [Natom, 3]
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [Natom, 3]
    is_protein : torch.Tensor
        Boolean mask for protein atoms, Shape of [Natom]
    is_dna : torch.Tensor
        Boolean mask for DNA atoms, Shape of [Natom]
    is_rna : torch.Tensor
        Boolean mask for RNA atoms, Shape of [Natom]
    is_ligand : torch.Tensor
        Boolean mask for ligand atoms, Shape of [Natom]
    asym_id : torch.Tensor
        Asymmetric unit IDs for atoms, Shape of [Natom]
    atom_mask : torch.Tensor
        Boolean mask for resolved atoms, Shape of [Natom]

    metric : dict[str, list[torch.Tensor]]

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        The metrics for each modality
    """
    assert pred_coords.shape == true_coords.shape, (
        "Predicted and true coordinates must have the same shape."
    )
    assert pred_coords.shape[0] == atom_mask.shape[0], (
        "Atom mask must have the same number of atoms as coordinates."
    )

    # Convert to float32
    dtype = torch.float32
    device = pred_coords.device
    pred_coords, true_coords = pred_coords.to(dtype), true_coords.to(dtype)

    # Remove masked atoms
    pred_coords = pred_coords[atom_mask]  # [Natom_resolved, 3]
    true_coords = true_coords[atom_mask]  # [Natom_resolved, 3]
    is_protein = is_protein[atom_mask]  # [Natom_resolved]
    is_dna = is_dna[atom_mask]  # [Natom_resolved]
    is_rna = is_rna[atom_mask]  # [Natom_resolved]
    is_ligand = is_ligand[atom_mask]  # [Natom_resolved]
    asym_id = asym_id[atom_mask]  # [Natom_resolved]
    del atom_mask  # No longer needed

    metrics: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}

    # === Compute RMSD === #
    # Assume the input coordinates are already aligned
    metrics["rmsd"] = compute_rmsd(true_coords, pred_coords, mask=None)
    weights["rmsd"] = torch.tensor(1.0, dtype=dtype, device=device)

    # === Compute LDDT per modality === #
    modality_mask = {
        "protein": is_protein,
        "dna": is_dna,
        "rna": is_rna,
        "ligand": is_ligand,
    }

    # Compute pairwise distance
    pdist_true = torch.norm(true_coords[:, None, :] - true_coords[None, :, :], dim=-1)
    pdist_pred = torch.norm(pred_coords[:, None, :] - pred_coords[None, :, :], dim=-1)
    lddt_score = compute_pair_lddt(pdist_pred, pdist_true)  # [Natom, Natom]

    # Compute masks
    # Use 30Å cutoff for DNA/RNA, 15Å cutoff for protein/ligand
    cutoff_mask_15 = pdist_true < 15.0
    cutoff_mask_30 = pdist_true < 30.0

    # Exclude self-pairs
    cutoff_mask_15.fill_diagonal_(False)
    cutoff_mask_30.fill_diagonal_(False)

    # Compute intra-chain metrics
    intra_mask = asym_id[:, None] == asym_id[None, :]
    for ctype in ["protein", "dna", "rna", "ligand"]:
        intra_name = f"intra_{ctype}"
        metric_name = f"lddt_{intra_name}"

        # Compute type mask
        type_mask = modality_mask[ctype]  # [Natom]
        type_mask = type_mask[:, None] & type_mask[None, :]  # [Natom, Natom]

        # Compute final mask
        if ctype in ("dna", "rna"):
            cutoff_mask = cutoff_mask_30
        else:
            cutoff_mask = cutoff_mask_15

        lddt_mask = cutoff_mask & intra_mask & type_mask

        # Compute LDDT
        total_pairs = lddt_mask.sum()
        lddt = (lddt_score * lddt_mask).sum() / total_pairs.clamp(1)
        metrics[metric_name] = lddt
        weights[metric_name] = (total_pairs > 0).float()

    # Compute interface metrics
    # NOTE: we only 6 interface types used in AlphaFold3 paper,
    # e.g., DNA-DNA interfaces are not computed.
    interface_mask = ~intra_mask
    for ctype1, ctype2 in (
        ("protein", "protein"),
        ("dna", "dna"),
        ("rna", "rna"),
        ("dna", "protein"),
        ("rna", "protein"),
        ("ligand", "protein"),
        ("dna", "ligand"),
        ("rna", "ligand"),
    ):
        interface_name = f"{ctype1}_{ctype2}"
        metric_name = f"lddt_inter_{interface_name}"

        # Compute type mask
        type_mask = modality_mask[ctype1][:, None] & modality_mask[ctype2][None, :]

        # Compute final mask
        if ctype1 in ("dna", "rna") or ctype2 in ("dna", "rna"):
            cutoff_mask = cutoff_mask_30
        else:
            cutoff_mask = cutoff_mask_15
        lddt_mask = cutoff_mask & interface_mask & type_mask
        total_pairs = lddt_mask.sum()
        lddt = (lddt_score * lddt_mask).sum() / total_pairs.clamp(1)
        metrics[metric_name] = lddt
        weights[metric_name] = (total_pairs > 0).float()

    # Compute complex lddt across all pairs
    lddt_mask = cutoff_mask_15
    total_pairs = lddt_mask.sum()
    complex_lddt = (lddt_score * lddt_mask).sum() / total_pairs.clamp(1)
    metrics["lddt"] = complex_lddt
    weights["lddt"] = (total_pairs > 0).float()

    return metrics, weights


def compute_validation_metrics(
    f_input: FoldingInput,
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    align: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute the validation metrics for different modalities.

    Parameters
    ----------
    f_input : FoldingInput
        Input features
    true_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction and alignment.
        Shape of [B, Nsample, Natom, 3]
    pred_coords : torch.Tensor
        Predicted atom coordinates
        Shape of [B, Nsample, Natom, 3]
    atom_mask : torch.Tensor
        Mask for resolved atoms for each corrected true structure.
        Shape of [B, Nsample, Natom]
    align : bool
        Whether to rigidly align true coordinates to predicted coordinates
        before metric computation. Can be skipped if true coordinates are
        already aligned (`permute_label_coordinates`).

    Returns
    -------
    dict[str, tuple[torch.Tensor, torch.Tensor]]
        The metric value and weight for each metric
    """

    B, Nsample, _, _ = pred_coords.shape
    device = pred_coords.device

    batch_indices = torch.arange(B, device=device)[:, None]
    token_idx = f_input.atom.token_index
    # [B, Ntoken] -> [B, Natom]
    is_protein = f_input.token.is_protein[batch_indices, token_idx]
    is_dna = f_input.token.is_dna[batch_indices, token_idx]
    is_rna = f_input.token.is_rna[batch_indices, token_idx]
    is_ligand = f_input.token.is_ligand[batch_indices, token_idx]
    asym_id = f_input.token.asym_id[batch_indices, token_idx]

    metric_keys = [
        ("rmsd", "min"),
        ("lddt", "max"),
        ("lddt_inter_protein_protein", "max"),
        ("lddt_inter_dna_dna", "max"),
        ("lddt_inter_rna_rna", "max"),
        ("lddt_inter_dna_protein", "max"),
        ("lddt_inter_rna_protein", "max"),
        ("lddt_inter_ligand_protein", "max"),
        ("lddt_inter_dna_ligand", "max"),
        ("lddt_inter_rna_ligand", "max"),
        ("lddt_intra_protein", "max"),
        ("lddt_intra_dna", "max"),
        ("lddt_intra_rna", "max"),
        ("lddt_intra_ligand", "max"),
    ]
    # All values
    all_metrics: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}
    all_weights: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}

    # Best values for each metric
    all_top5_metrics: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}
    all_top5_weights: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}

    # Values of the best sample (complex-wise lddt)
    # Introduced in Boltz1.
    all_best_complex_metrics: dict[str, list[torch.Tensor]] = {
        k: [] for k, _ in metric_keys
    }
    all_best_complex_weights: dict[str, list[torch.Tensor]] = {
        k: [] for k, _ in metric_keys
    }

    if align:
        # Align true coordinates to predicted coordinates
        true_coords = rigid_align(true_coords, pred_coords, mask=atom_mask)

    for b in range(B):
        values = []
        weights = []
        for n in range(Nsample):
            single_values, single_weights = compute_validation_metric_singles(
                true_coords[b, n],
                pred_coords[b, n],
                is_protein[b],
                is_dna[b],
                is_rna[b],
                is_ligand[b],
                asym_id[b],
                atom_mask[b, n],
            )
            values.append(single_values)
            weights.append(single_weights)

        # Store all values/weights for all samples
        for k, _ in metric_keys:
            all_metrics[k].extend([v[k] for v in values])
            all_weights[k].extend([w[k] for w in weights])

        # Store best values across samples for each metrics
        for k, agg in metric_keys:
            stacked_values = torch.stack([v[k] for v in values])
            stacked_weights = torch.stack([w[k] for w in weights])

            nan_mask = ~stacked_values.isfinite()
            stacked_weights[nan_mask] = 0.0
            if agg == "max":
                stacked_values[nan_mask] = -1e6
                best_idx = torch.argmax(stacked_values)
            else:
                stacked_values[nan_mask] = 1e6
                best_idx = torch.argmin(stacked_values)

            all_top5_metrics[k].append(stacked_values[best_idx])
            all_top5_weights[k].append(stacked_weights[best_idx])

        # Store the values of the best sample (highest-lddt)
        complex_lddts = torch.stack([v["lddt"] for v in values])
        best_complex_idx = torch.argmax(complex_lddts)
        for k, _ in metric_keys:
            all_best_complex_metrics[k].append(values[best_complex_idx][k])
            all_best_complex_weights[k].append(weights[best_complex_idx][k])

    # Store as tensors
    validation_metrics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # Store average rmsd
    for k, _ in metric_keys:
        v = torch.stack(all_metrics[k], dim=0)
        w = torch.stack(all_weights[k], dim=0)
        validation_metrics[f"avg_{k}"] = (v, w)

    # Store best-of-five metrics
    # TODO: in future, consider confidence top-1 scores as well
    for k, _ in metric_keys:
        v = torch.stack(all_top5_metrics[k], dim=0)
        w = torch.stack(all_top5_weights[k], dim=0)
        validation_metrics[k] = (v, w)

    for k, _ in metric_keys:
        v = torch.stack(all_top5_metrics[k], dim=0)
        w = torch.stack(all_top5_weights[k], dim=0)
        validation_metrics[f"best_{k}"] = (v, w)

    # Store best complex-wise lddt metrics
    for k, _ in metric_keys:
        v = torch.stack(all_best_complex_metrics[k], dim=0)
        w = torch.stack(all_best_complex_weights[k], dim=0)
        validation_metrics[f"complex_{k}"] = (v, w)

    return validation_metrics
