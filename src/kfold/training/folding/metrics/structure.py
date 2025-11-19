from collections.abc import Sequence

import torch

from kfold.data.model_input import FoldingInput
from kfold.training.folding.loss.diffusion import get_atom_weights, weighted_rigid_align
from kfold.utils.misc import expand_dim


def compute_pair_lddt(
    pdist_pred: torch.Tensor,
    pdist_true: torch.Tensor,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
):
    """Compute the lddt score from predicted and true distances.

    Parameters
    ----------
    d_predicted : torch.Tensor
        Predicted distances, shape (B, Natom, Natom)
    d_true : torch.Tensor
        Ground truth distances, shape (B, Natom, Natom)
    thresholds : Sequence[float]
        Distance error thresholds for lddt calculation
    """
    dtype = pdist_pred.dtype
    error = torch.abs(pdist_true - pdist_pred)
    scores = torch.zeros_like(error, dtype=dtype)
    for threshold in thresholds:
        scores += (error < threshold).to(dtype)
    return scores / len(thresholds)


def compute_rmsd(
    coords_pred: torch.Tensor,
    coords_true: torch.Tensor,
    mask: torch.Tensor,
):
    """Compute the rmsd score from predicted and true distances.

    Parameters
    ----------
    coords_pred : torch.Tensor
        Predicted atom coordinates, Shape of [Natom, 3]
    coords_true : torch.Tensor
        Ground truth atom coordinates, Shape of [Natom, 3]
    mask : torch.Tensor
        Boolean mask for resolved atoms, Shape of [Natom]

    Returns
    -------
    torch.Tensor
        The rmsd score between predicted and true coordinates
    """
    diff = ((coords_pred - coords_true) ** 2).sum(-1)
    masked_diff = diff * mask
    mse = masked_diff.sum() / mask.sum()
    rmsd = torch.sqrt(mse)
    return rmsd


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

    metrics: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}

    # === Compute RMSD === #
    rmsd = compute_rmsd(pred_coords, true_coords, atom_mask)
    metrics["rmsd"] = rmsd
    # TODO: to be discussed, should we weight by number of atoms?
    weights["rmsd"] = atom_mask.sum()

    # === Compute LDDT per modality === #
    modality_mask = {
        "protein": is_protein,
        "dna": is_dna,
        "rna": is_rna,
        "ligand": is_ligand,
    }

    # Compute pairwise distance
    pdist_true = torch.cdist(true_coords, true_coords)  # [Natom, Natom]
    pdist_pred = torch.cdist(pred_coords, pred_coords)  # [Natom, Natom]
    lddt_score = compute_pair_lddt(pdist_pred, pdist_true)  # [Natom, Natom]

    # Compute masks
    valid_mask = atom_mask[:, None] & atom_mask[None, :]
    valid_mask.diagonal().fill_(0)  # Exclude self-pairs
    local_mask_15 = pdist_true < 15.0
    local_mask_30 = pdist_true < 30.0  # For DNA/RNA intra-chains and interfaces

    # Compute intra-chain metrics
    intra_mask = asym_id[:, None] == asym_id[None, :]
    for ctype in ["protein", "dna", "rna", "ligand"]:
        intra_name = f"intra_{ctype}"
        metric_name = f"lddt_{intra_name}"

        # Compute type mask
        type_mask = modality_mask[ctype]  # [Natom]
        type_mask = type_mask[:, None] & type_mask[None, :]  # [Natom, Natom]

        # Compute local lddt mask
        # Use 30Å cutoff for DNA/RNA, 15Å cutoff for protein/ligand
        cutoff_mask = local_mask_30 if ctype in ("dna", "rna") else local_mask_15

        # Compute final mask
        lddt_mask = valid_mask & intra_mask & type_mask & cutoff_mask

        # Compute LDDT
        total_pairs = lddt_mask.sum()
        lddt = (lddt_score * lddt_mask).sum() / total_pairs.clamp(1)
        metrics[metric_name] = lddt
        weights[metric_name] = total_pairs

    # Compute interface metrics
    # NOTE: we only 6 interface types used in AlphaFold3 paper,
    # e.g., DNA-DNA interfaces are not computed.
    for ctype1, ctype2 in (
        ("protein", "protein"),
        ("dna", "protein"),
        ("rna", "protein"),
        ("ligand", "protein"),
        ("dna", "ligand"),
        ("rna", "ligand"),
    ):
        interface_name = f"{ctype1}_{ctype2}"
        metric_name = f"lddt_{interface_name}"

        # Compute type mask
        type_mask = modality_mask[ctype1][:, None] & modality_mask[ctype2][None, :]

        # Use 30Å cutoff for DNA/RNA, 15Å cutoff for protein/ligand
        # NOTE: While boltz1 uses 15Å for dna/rna interfaces, here we use 30Å
        # according to AF3 paper.
        cutoff_mask = (
            local_mask_30
            if ctype1 in ("dna", "rna") or ctype2 in ("dna", "rna")
            else local_mask_15
        )

        # Compute final mask
        lddt_mask = valid_mask & ~intra_mask & type_mask & cutoff_mask
        total_pairs = lddt_mask.sum()
        lddt = (lddt_score * lddt_mask).sum() / total_pairs.clamp(1)
        metrics[metric_name] = lddt
        weights[metric_name] = total_pairs

    # Compute complex lddt
    # TODO: do we consider all interface types here?
    # Currently, we use partial types only.
    overall_lddt = 0
    overall_weights = sum(weights.values())
    for k in metrics.keys():
        overall_lddt += metrics[k] * weights[k]
    overall_lddt /= overall_weights
    metrics["lddt"] = overall_lddt  # type: ignore
    weights["lddt"] = overall_weights

    return metrics, weights


def compute_validation_metrics(
    f_input: FoldingInput,
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute the validation metrics for different modalities.

    Parameters
    ----------
    f_input : FoldingInput
        Input features
    true_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction
        Shape of [B, Nsample, Natom, 3]
    pred_coords : torch.Tensor
        Predicted atom coordinates
        Shape of [B, Nsample, Natom, 3]
    atom_mask : torch.Tensor
        Mask for resolved atoms for each corrected true structure.
        Shape of [B, Nsample, Natom]

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
        ("lddt_protein_protein", "max"),
        ("lddt_dna_protein", "max"),
        ("lddt_rna_protein", "max"),
        ("lddt_ligand_protein", "max"),
        ("lddt_dna_ligand", "max"),
        ("lddt_rna_ligand", "max"),
        ("lddt_intra_protein", "max"),
        ("lddt_intra_dna", "max"),
        ("lddt_intra_rna", "max"),
        ("lddt_intra_ligand", "max"),
    ]
    # All values
    all_metrics: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}
    all_weights: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}

    # Best values for each metric
    all_best_metrics: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}
    all_best_weights: dict[str, list[torch.Tensor]] = {k: [] for k, _ in metric_keys}

    # Values of the best sample (complex-wise lddt)
    # Introduced in Boltz1.
    all_best_complex_metrics: dict[str, list[torch.Tensor]] = {
        k: [] for k, _ in metric_keys
    }
    all_best_complex_weights: dict[str, list[torch.Tensor]] = {
        k: [] for k, _ in metric_keys
    }

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
            if agg == "max":
                best_idx = torch.argmax(stacked_values)
            else:
                best_idx = torch.argmin(stacked_values)
            all_best_metrics[k].append(stacked_values[best_idx])
            all_best_weights[k].append(stacked_weights[best_idx])

        # Store the values of the best sample (highest-lddt)
        complex_lddts = torch.stack([v["lddt"] for v in values])
        best_complex_idx = torch.argmax(complex_lddts)
        for k, _ in metric_keys:
            all_best_complex_metrics[k].append(values[best_complex_idx][k])
            all_best_complex_weights[k].append(weights[best_complex_idx][k])

    # Store as tensors
    validation_metrics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k in all_metrics.keys():
        v = torch.stack(all_metrics[k], dim=0)
        w = torch.stack(all_weights[k], dim=0)
        validation_metrics[k] = (v, w)
    for k in all_best_metrics.keys():
        v = torch.stack(all_best_metrics[k], dim=0)
        w = torch.stack(all_best_weights[k], dim=0)
        validation_metrics[f"best/{k}"] = (v, w)
    for k in all_best_complex_metrics.keys():
        v = torch.stack(all_best_complex_metrics[k], dim=0)
        w = torch.stack(all_best_complex_weights[k], dim=0)
        validation_metrics[f"best_complex/{k}"] = (v, w)

    return validation_metrics


def permute_label_coordinates(
    f_input: FoldingInput,
    pred_coords: torch.Tensor,
    full_structure_dict: dict,
    symmetry_correction: bool = True,
    minimize_metric: str = "lddt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the best matching true coordinates to the predicted coordinates.
    Chain permutation and atom swaps.

    Parameters
    ----------
    f_input : FoldingInput
        Input features
    pred_coords : torch.Tensor
        Predicted atom coordinates, Shape of [B, Nsample, Natom, 3]
    full_structure_dict : dict
        Full structure dictionary containing symmetry information
    symmetry_correction : bool
        Whether to apply symmetry correction
    minimize_metric : str
        Metric to minimize when finding the best permutation during symmetry correction
        "lddt" or "rmsd"

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The true coordinates after permutation and the corresponding mask
    """
    B, Nsample, Natom, _ = pred_coords.shape
    if not symmetry_correction:
        """Perform weighted rigid alignment without symmetry correction."""

        true_coords = f_input.atom.label_coords  # [B, Natom, Nholo, 3]
        # HACK: we only consider the first bio-assembly
        true_coords = true_coords[:, :, 0, :]
        mask = f_input.atom.resolved_mask  # [B, Natom]

        # Weighted rigid alignment for best permutation
        weights = get_atom_weights(f_input)  # [B, Natom]

        # Expand to match pred_coords shape
        true_coords = expand_dim(
            true_coords, dim=1, n_repeat=Nsample
        )  # [B, Nsample, Natom, 3]
        weights = expand_dim(weights, 1, Nsample)  # [B, Nsample, Natom]
        mask = expand_dim(mask, 1, Nsample)  # [B, Nsample, Natom]
        aligned_coords = weighted_rigid_align(true_coords, pred_coords, weights, mask)
    else:
        raise NotImplementedError("Symmetry correction is not implemented yet.")

    return aligned_coords, mask
