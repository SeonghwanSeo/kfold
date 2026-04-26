"""Permutation alignment for confidence model training.

Differ to validation-time permutation alignment, the functions in this module
are designed to align the reference structure to the cropped coordinates.

- pred_coords: the predicted coordinates for the cropped region, which is the
    diffusion sample at each training step, i.e., all atoms are present.
- GT coords: the ground truth coordinates of the training structure, which may
    contain unresolved atoms. The GT can includes additional chains/residues
    outside the cropped region, so that the permutation alignment is performed
    on the full GT structure and the cropped Pred structure to find the optimal
    mapping from GT to Pred.

    Source:
    - RefStructure: the full GT structure, where the unresolved atoms are masked
        with NaN coordinates.
    - FoldingInput: the cropped GT coordinates (f_input.atom.label_coords) and
        masks (f_input.atom.resolved_mask), where the unresolved atoms are
        masked with 0.0 coordinates.
"""

import logging

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.geometry.rigid_align import get_rigid_transform, rigid_align

from .symmetry import ChainGroup, ChainSymmetry, ResidueSymmetry

# ChainGroup = tuple[int, ...]
# ChainSymmetry = list[ChainGroup]
# ResidueSymmetry = np.ndarray | None  # [num_permutations, num_atoms]

__all__ = ["get_aligned_true_coords"]

PROTEIN_CA_TYPE: int = C.atom.atom_name_to_index[C.AtomName.CA]
NUCLEIC_C1P_TYPE: int = C.atom.atom_name_to_index[C.AtomName.C1_PRIME]

logger = logging.getLogger(__name__)


def get_aligned_true_coords(
    ref_struct: RefStructure,
    f_input: FoldingInput,
    pred_coords: torch.Tensor,
    symmetry_dict: dict,
) -> torch.Tensor:
    """Get aligned ground truth structure to predicted coordinates.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    f_input : FoldingInput
        The input features of the cropped structure for model training.
    pred_coords : torch.Tensor
        The cropped predicted coordinates. Shape: [Natoms, 3]
    symmetry_dict : dict, optional
        The dictionary containing symmetry information:
        - "chain": dict[str, ChainSymmetry]
            key: Symmetry ID (<entity_id1>-<entity_id2>-...)
            value: list of groups of chain asym_ids that can be permuted.
        - "residue": dict[int, list[ResidueSymmetry]]
            key: asym_id of the chain
            value: list of ResidueSymmetry for each residue in the chain.

    Returns
    -------
    gt_coords_aligned : torch.Tensor
        The aligned ground truth coordinates. Shape: [Natoms, 3]

    NOTE (Seonghwan): Differ to f_input.atom.label_coords where the coordinates
    of unresolved atoms are 0.0, gt_coords_aligned is masked them with NaN to
    identify unresolved atoms after permutation.
    """
    assert not f_input.is_batched, (
        "Batch dimension should be removed before calling this function."
    )

    with torch.autocast(f_input.device.type, enabled=False), torch.no_grad():
        # 1. Multi-chain permutation alignment.
        try:
            gt_coords = _do_optimal_chain_permutation(
                ref_struct,
                f_input,
                pred_coords,
                symmetry_dict["chain"],
            )
        except Exception as e:
            # Fallback to using the cropped GT coordinates without permutation.
            logger.error(f"Chain permutation alignment failed: {e}")
            gt_coords, mask = f_input.atom.label_coords, f_input.atom.resolved_mask
            gt_coords = torch.where(mask[:, None], gt_coords, torch.nan)

        # 2. Rigid alignment.
        gt_coords = _rigid_align_gt_to_pred(f_input, gt_coords, pred_coords)

        # 3. Atomic permutation alignment.
        gt_coords = _do_optimal_atom_permutation(
            f_input,
            gt_coords,
            pred_coords,
            symmetry_dict["residue"],
        )

    return gt_coords


def _do_optimal_chain_permutation(
    ref_struct: RefStructure,
    f_input: FoldingInput,
    pred_coords: torch.Tensor,
    chain_symmetries: dict[str, ChainSymmetry],
) -> torch.Tensor:
    """Compute minimum RMSD coordinates considering chain permutation.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    f_input : FoldingInput
        The input features of the cropped structure for model training.
    pred_coords : torch.Tensor
        The cropped predicted coordinates. Shape: [Natoms, 3]
    chain_symmetries : dict[str, list[tuple[int, ...]]]
        A dict of groups of chain asym_ids that can be permuted among each other.

    Returns
    -------
    gt_coords_aligned : torch.Tensor
        The aligned ground truth coordinates. Shape: [Natoms, 3]
    """
    device = pred_coords.device

    # Extract the asym_ids of the chains present in the cropped region.
    pred_asym_ids: list[int] = f_input.chain.asym_id.tolist()
    pred_asym_ids = [i for i in pred_asym_ids if i > 0]  # Remove subsequent padding

    # Extract the necessary chains for permutation alignment.
    valid_asym_ids: set[int] = set(pred_asym_ids)
    valid_entity_ids = set(
        c.entity_id for c in ref_struct.chains if c.asym_id in valid_asym_ids
    )
    chains = [c for c in ref_struct.chains if c.entity_id in valid_entity_ids]

    chain_symmetries = chain_symmetries.copy()
    for k, groups in list(chain_symmetries.items()):
        if all(i not in valid_asym_ids for g in groups for i in g):
            del chain_symmetries[k]

    # Extract ground truth coordinates for chain permutation alignment.
    # NOTE: the unresolved atoms in RefStructure have coordinates of NaN.
    gt_coords_dict: dict[int, torch.Tensor] = {
        c.asym_id: torch.as_tensor(c.atom.coords, device=device) for c in chains
    }

    # Define the return function to get the aligned GT coordinates.
    def get_aligned_gt_coords(permuted_asym_ids: list[int] | None = None) -> torch.Tensor:
        """Get the cropped ground truth coordinates based on the given permutation."""
        if (permuted_asym_ids is None) or (permuted_asym_ids == pred_asym_ids):
            # No permutation, return the original GT coordinates with masking
            gt_coords = f_input.atom.label_coords
            gt_mask = f_input.atom.resolved_mask
            return torch.where(gt_mask[:, None], gt_coords, torch.nan)

        gt_coords_permuted = torch.zeros_like(pred_coords)
        st = 0
        for c_i, asym_id in enumerate(permuted_asym_ids):
            num_atoms: int = int(f_input.chain.num_atoms[c_i].item())
            end = st + num_atoms
            # Crop the GT chain coordinates.
            gt_chain_coords = gt_coords_dict[asym_id]
            atom_indices = f_input.atom.atom_index[st:end]
            gt_coords_permuted[st:end] = gt_chain_coords[atom_indices]
            st = end
        return gt_coords_permuted

    # Return early if there is no chain permutation.
    if all(len(group) == 1 for group in chain_symmetries.values()):
        return get_aligned_gt_coords()

    # Extract the chain coordinates in prediction.
    pred_center_coords_dict: dict[int, torch.Tensor] = {}
    pred_center_indices_dict: dict[int, torch.Tensor] = {}
    st = 0
    for c_i, asym_id in enumerate(pred_asym_ids):
        num_tokens = int(f_input.chain.num_tokens[c_i].item())
        end = st + num_tokens
        center_idcs = f_input.token.center_index[st:end]
        pred_center_coords_dict[asym_id] = pred_coords[center_idcs]
        pred_center_indices_dict[asym_id] = f_input.atom.atom_index[center_idcs]
        st = end

    # Prioritize chain groups for permutation alignment.
    anchor_group: ChainGroup = _get_anchor_chain_group(chains, chain_symmetries)

    # Find the optimal multi-chain permutation.
    pred_to_gt = _multi_chain_permutation_alignment(
        anchor_group,
        chain_symmetries,
        gt_coords_dict,
        pred_center_coords_dict,
        pred_center_indices_dict,
    )
    permuted_asym_ids = [pred_to_gt.get(asym_id, asym_id) for asym_id in pred_asym_ids]
    return get_aligned_gt_coords(permuted_asym_ids)


def _get_anchor_chain_group(
    chains: list[Chain], chain_symmetries: dict[str, ChainSymmetry]
) -> ChainGroup:
    """Prioritize chain groups for permutation alignment.

    Priority:
      1. Polymer chains
      2. Less symmetry
      3. More resolved atoms
      4. More residues/atoms

    Parameters
    ----------
    chains: list[Chain]
        The list of chains in the reference structure.
    chain_symmetries: dict[str, ChainSymmetry]
        A list of groups of chain asym_ids that can be permuted among each other.

    Returns
    -------
    anchor_group: ChainGroup
        The selected anchor group for permutation alignment.
    """
    asym_ids: list[int] = [c.asym_id for c in chains]

    # Prioritize polymer chains as anchors
    ctype_dict: dict[int, int] = {c.asym_id: 1 if c.is_polymer else 0 for c in chains}

    # Prioritize chains with less symmetry as anchors
    nsym_dict: dict[int, int] = {
        aid: len(gs) for gs in chain_symmetries.values() for g in gs for aid in g
    }

    # Prioritize chains with more resolved center atoms as anchors
    def get_n_resolved_atoms(c: Chain) -> int:
        if c.is_protein:
            center_coords = c.atom.coords[c.atom.name == "CA"]
        elif c.is_nucleic_acid:
            center_coords = c.atom.coords[c.atom.name == "C1'"]
        else:
            center_coords = c.atom.coords
        return np.isfinite(center_coords).all(-1).sum()

    nvalid_dict: dict[int, int] = {c.asym_id: get_n_resolved_atoms(c) for c in chains}

    # Prioritize chains with more residues as anchors
    len_dict: dict[int, int] = {c.asym_id: c.num_residues for c in chains}

    priority = {
        v: (ctype_dict[v], -nsym_dict[v], nvalid_dict[v], len_dict[v], v)
        for v in asym_ids
    }
    anchor: Chain = max(chains, key=lambda c: priority[c.asym_id])

    # Return group containing the anchor chain
    for sym in chain_symmetries.values():
        for group in sym:
            if anchor.asym_id in group:
                return group
    raise RuntimeError("Anchor chain group not found in symmetries.")


def _multi_chain_permutation_alignment(
    anchor_gt: ChainGroup,
    chain_symmetries: dict[str, ChainSymmetry],
    gt_coords_dict: dict[int, torch.Tensor],
    pred_center_coords_dict: dict[int, torch.Tensor],
    pred_center_indices_dict: dict[int, torch.Tensor],
) -> dict[int, int]:
    """Find the optimal multi-chain permutation mapping from ground truth to prediction.
    See Algorithm 3 "Multi-chain permutation alignment" in AlphaFold-Multimer paper.
    """
    pred_asym_ids = set(pred_center_coords_dict.keys())
    device = next(iter(gt_coords_dict.values())).device

    # Identify buckets
    buckets = [chain_symmetries[k] for k in sorted(chain_symmetries.keys())]
    group_to_bucket = {tuple(g): b for b in buckets for g in b}

    # Get GT masks
    gt_mask_dict = {
        i: torch.isfinite(coords).all(dim=-1).float()
        for i, coords in gt_coords_dict.items()
    }

    # A group is a candidate if at least one of its chains is in the crop.
    anchor_pred_list: list[ChainGroup] = group_to_bucket[anchor_gt]

    best_total_cost = float("inf")
    best_pred_to_gt_mapping: dict[int, int] = {}

    for anchor_pred in anchor_pred_list:
        # Line 1: Compute alignment (GT to Pred) using the anchor
        # NOTE: here we only consider the first chain (representative)
        # in the anchor group for alignment. The subsequent chains are
        # covalently-linked ligands to the representative polymer chain.
        anchor_gt_asym_id = anchor_gt[0]
        anchor_pred_asym_id = anchor_pred[0]
        if anchor_pred_asym_id not in pred_asym_ids:
            # Skip if the anchor chain in prediction is not present in the crop.
            continue

        x_pred_anchor = pred_center_coords_dict[anchor_pred_asym_id]
        indices_anchor = pred_center_indices_dict[anchor_pred_asym_id]
        # Extract the corresponding center coordinates and mask for the GT anchor chain.
        x_gt_anchor = gt_coords_dict[anchor_gt_asym_id][indices_anchor]
        mask_anchor = gt_mask_dict[anchor_gt_asym_id][indices_anchor]

        if mask_anchor.sum() < 3:
            # Not enough resolved atoms in crop, skip this anchor candidate.
            continue

        # Line 2: Apply the transform to GT coordinates.
        RT, T = get_rigid_transform(x_gt_anchor, x_pred_anchor, mask_anchor)
        gt_coords_aligned_dict: dict[int, torch.Tensor] = {
            i: coords @ RT + T for i, coords in gt_coords_dict.items()
        }

        # Greedy Assignment for all buckets
        total_cost = 0.0
        pred_to_gt = {}
        for bucket in buckets:
            # pred_groups: groups in this bucket that have at least one chain in the crop.
            pred_groups = [g for g in bucket if any(aid in pred_asym_ids for aid in g)]
            if not pred_groups:
                continue

            # gt_groups: all groups in this bucket.
            gt_groups = bucket

            # Line 3, 4: Compute the centroid of the chain groups in the bucket.
            ns, nt = len(pred_groups), len(gt_groups)
            com_pred = torch.zeros((ns, 3), device=device)
            com_gt = torch.zeros((ns, nt, 3), device=device)
            for s in range(ns):
                g_s = pred_groups[s]
                _g_s = [i for i in g_s if i in pred_asym_ids]
                _x_pred = torch.cat([pred_center_coords_dict[i] for i in _g_s])
                com_pred[s] = _x_pred.mean(dim=0)
                for t in range(nt):
                    g_t = gt_groups[t]
                    _x_gt = torch.cat(
                        [
                            gt_coords_aligned_dict[j][pred_center_indices_dict[i]]
                            for i, j in zip(g_s, g_t, strict=True)
                            if i in pred_asym_ids
                        ]
                    )
                    # NOTE: If no valid atoms, the centroid will be NaN.
                    com_gt[s, t] = _x_gt.nanmean(dim=0)

            if torch.isnan(com_gt).all():
                # If all centroids are NaN, skip this bucket without cost accumulation.
                continue

            # Line 5: Find the optimal permutation of GT groups to Pred groups.
            if len(bucket) > 1:
                perm = _find_optimal_chain_permutation(com_pred, com_gt)
            else:
                perm = [0]
            for s, t in enumerate(perm):
                g_s, g_t = pred_groups[s], gt_groups[t]
                cost = (com_pred[s] - com_gt[s, t]).norm(dim=-1).nansum().item()
                total_cost += cost
                for i, j in zip(g_s, g_t, strict=True):
                    if i in pred_asym_ids:
                        pred_to_gt[i] = j

        if total_cost < best_total_cost:
            best_total_cost, best_pred_to_gt_mapping = total_cost, pred_to_gt

    if best_total_cost == float("inf"):
        print("Warning: No valid anchor found for chain permutation alignment.")
        return {asym_id: asym_id for asym_id in pred_asym_ids}

    return best_pred_to_gt_mapping


def _find_optimal_chain_permutation(
    com_pred: torch.Tensor, com_gt: torch.Tensor
) -> list[int]:
    """Find the optimal chain permutation based on centroid distances.
    See Algorithm 4 "Find Optimal Permutation" in AlphaFold-Multimer paper.

    Parameters
    ----------
    com_pred : torch.Tensor
        The centroids of the predicted chains. Shape: [Ns, 3]
        Applied with different resolved atom masks for each GT group: Nt.
    com_gt : torch.Tensor
        The centroids of the ground truth chains. Shape: [Ns, Nt, 3]

    Returns
    -------
    perm: list[int]
        The optimal permutation of GT groups to Pred groups.
    """
    # Line 1: Compute distance matrix between GT centroids and Pred centroids
    d = torch.norm(com_pred[:, None, :] - com_gt, dim=-1)  # [Ns, Nt]
    d.nan_to_num_(1e9)  # Replace NaN with large distance

    # Greedily assign each GT group to the closest Pred group
    perm: list[int] = [-1] * com_gt.shape[0]
    for s in range(com_pred.shape[0]):  # For each Pred group
        # Line 3: Find the closest GT group for the Pred group
        best_idx = int(torch.argmin(d[s]).item())
        # Line 4: Mask out the assigned GT group
        d[:, best_idx] = float("inf")
        # Update the mapping
        perm[s] = best_idx
    assert all(i >= 0 for i in perm), "Not all GT groups are assigned."
    return perm


# ============================================================
# Geometric alignment.
# ============================================================
def _rigid_align_gt_to_pred(
    f_input: FoldingInput,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
) -> torch.Tensor:
    """Rigidly align GT coordinates to Pred coordinates using Kabsch algorithm.

    Parameters
    ----------
    f_input : FoldingInput
        The input features of the cropped structure for model training.
    gt_coords : torch.Tensor
        The ground truth coordinates. Shape: [Natom, 3]
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natom, 3]

    Returns
    -------
    gt_coords_aligned: torch.Tensor
        The aligned ground truth coordinates.
    """
    # First, try to use the residue centers for alignment, which provides
    # robust frame for atom-level permutation alignment.
    is_center = f_input.atom.resolved_mask & (
        (f_input.atom.atom_type == PROTEIN_CA_TYPE)
        | (f_input.atom.atom_type == NUCLEIC_C1P_TYPE)
    )
    anchor_index = torch.where(is_center)[0]
    if len(anchor_index) < 3:
        # If there are less than 3 resolved centers, fallback to using
        # token centers for alignment: residue centers + all-atoms for
        # non-standard residues and ligands.
        anchor_index = f_input.token.center_index[f_input.token.center_mask]

    # Alignment with Kabsch algorithm.
    mask = gt_coords.isfinite().all(dim=-1)
    gt_coords_aligned = rigid_align(gt_coords, pred_coords, mask, anchor_index)
    gt_coords_aligned[~mask] = torch.nan
    return gt_coords_aligned


# ============================================================
# Atom permutation alignment
# ============================================================
def _do_optimal_atom_permutation(
    f_input: FoldingInput,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    permutations: dict[int, list[ResidueSymmetry]],
) -> torch.Tensor:
    """Find the best residue/molecule permutation to minimize RMSD.

    Parameters
    ----------
    f_input : FoldingInput
        The input features of the cropped structure for model training.
    gt_coords : torch.Tensor
        The ground truth coordinates. Shape: [Natoms, 3]
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    permutations : dict[int, list[ResidueSymmetry]]
        The dictionary for atom permutations for each residue in each chain.

    Returns
    -------
    gt_coords_permuted: torch.Tensor
        The ground truth coordinates after optimal residue/molecule permutation.
    """
    gt_mask = gt_coords.isfinite().all(dim=-1)  # [Natoms]

    # Map every atom to its parent token to get asym_id and residue_index.
    num_atoms = gt_coords.shape[0]
    token_idx = f_input.atom.token_index[:num_atoms]
    atom_asym_ids: list[int] = f_input.token.asym_id[token_idx].tolist()
    atom_res_idxs: list[int] = f_input.token.residue_index[token_idx].tolist()
    is_standard: list[int] = f_input.token.is_standard[token_idx].tolist()

    # Boundaries: [0, N1, ..., Ntot], where atoms in [Ni, Ni+1) belong to
    # the same residue.
    atom_ref_uid = f_input.atom.ref_space_uid[:num_atoms]
    changes = atom_ref_uid[1:] != atom_ref_uid[:-1]
    change_indices = torch.where(changes)[0] + 1
    boundaries = [0] + change_indices.tolist() + [num_atoms]

    for i in range(len(boundaries) - 1):
        st, end = boundaries[i], boundaries[i + 1]
        n_atoms: int = end - st
        asym_id: int = atom_asym_ids[st]
        res_idx: int = atom_res_idxs[st]
        res_i: int = res_idx - 1  # Move to 0-based index for permutation lookup
        assert asym_id > 0, f"Invalid asym_id {asym_id} for atom index {st}."
        assert res_idx > 0, f"Invalid residue_index {res_idx} for atom index {st}."

        perms: np.ndarray | None = permutations[asym_id][res_i]
        if perms is None or len(perms) <= 1:
            # No ambiguity in this residue.
            continue

        n_perms, n_ref_atoms = perms.shape  # noqa
        if n_ref_atoms != n_atoms:
            # TODO: Implement partial permutation for non-standard residue/ligand
            # with missing atoms.
            if any(is_standard[st:end]):
                raise ValueError(
                    f"Incomplete atom set for standard residue at index {res_idx}. "
                    f"Expected {n_ref_atoms} atoms, got {n_atoms}."
                )
            continue

        x = pred_coords[st:end]
        x_gt = gt_coords[st:end]
        m = gt_mask[st:end]
        if not m.any():
            continue

        __do_optimal_atom_permutation_in_residue(x_gt, x, m, perms)

    return gt_coords


def __do_optimal_atom_permutation_in_residue(
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    permutations: ResidueSymmetry,
) -> None:
    """Resolve residue-level atom permutations (e.g., side-chain flips).

    This function uses a vectorized Mean Squared Error check for each residue
    to find the optimal atom mapping in the current pre-aligned frame.
    """
    perms_t = torch.as_tensor(permutations, device=gt_coords.device)
    x_pred = pred_coords[None, ...]  # [1, Natoms, 3]
    x_gt = gt_coords[perms_t]  # [Nperms, Natoms, 3]
    mask = gt_mask[perms_t]  # [Nperms, Natoms]

    # Compute squared distances between Ground Truth and Predicted coordinates.
    diff_sq = (x_gt - x_pred).pow(2).sum(-1)
    # Mask out unresolved atoms to prevent NaN propagation.
    diff_sq[~mask] = 0.0
    # The permutation with the minimum sum of squared distances is selected.
    cost = diff_sq.sum(-1)
    best_i = int(torch.argmin(cost).item())

    # Apply the optimal permutation to the coordinate tensor.
    if best_i > 0:
        # NOTE: The first permutation (index 0) is the identity mapping.
        # See `./symmetry.py`
        gt_coords[:] = x_gt[best_i]
