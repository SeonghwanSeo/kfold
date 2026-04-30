"""Permutation alignment for validation."""

import numpy as np
import torch

from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.geometry.rigid_align import get_rigid_transform, rigid_align

from .symmetry import ChainGroup, ChainSymmetry, ResidueSymmetry

__all__ = ["get_aligned_gt_structure"]

# ChainGroup = tuple[int, ...]
# ChainSymmetry = list[ChainGroup]
# ResidueSymmetry = np.ndarray | None  # [num_permutations, num_atoms]


def get_aligned_gt_structure(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    symmetry_dict: dict,
) -> RefStructure:
    """Get aligned ground truth structure to predicted coordinates.

    This function performs a multi-stage permutation alignment:
    1. Multi-chain permutation alignment to handle swappable chain groups
       (e.g., identical subunits in a homomer).
    2. Global rigid alignment using polymer backbone centers.
       This provides a robust frame for subsequent atomic permutation.
    3. Residue-level atom permutation alignment to handle side-chain symmetries
       and swappable atoms in small molecules.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    symmetry_dict : dict
        The dictionary containing symmetry information:
        - "chain": dict[str, ChainSymmetry]
            Key: Symmetry identifier (e.g., concatenated entity IDs).
            Value: List of groups of swappable chain asym_ids.
        - "residue": dict[int, list[ResidueSymmetry]]
            Key: Chain asym_id.
            Value: List of atom permutations (or None) for each residue in the chain.

    Returns
    -------
    ref_struct_aligned: RefStructure
        The reference structure with optimally permuted and rigid-aligned
        ground truth coordinates.
    """
    assert pred_coords.shape == (ref_struct.num_atoms, 3), (
        "Mismatch in number of atoms between coordinates and reference structure."
    )
    device = pred_coords.device

    # Validation alignment is performed in double precision for stability.
    with torch.autocast(device.type, enabled=False), torch.no_grad():
        # 1. Multi-chain permutation alignment.
        # Finds the optimal mapping of swappable homomer subunits.
        ref_struct = _do_optimal_chain_permutation(
            ref_struct,
            pred_coords,
            symmetry_dict["chain"],
        )
        gt_coords = torch.as_tensor(ref_struct.get_atom_coords(), device=device)

        # 2. Rigid alignment.
        # Align GT to Pred using polymer backbone (CA/C1') centers.
        # Backbone alignment is robust to side-chain/ligand flips.
        gt_coords = _rigid_align_gt_to_pred(ref_struct, gt_coords, pred_coords)

        # 3. Atomic permutation alignment.
        # Resolves side-chain symmetries and small molecule atom swappability.
        gt_coords = _do_optimal_atom_permutation(
            ref_struct,
            gt_coords,
            pred_coords,
            symmetry_dict["residue"],
        )

    # Update the reference structure with the new permuted/aligned coordinates.
    ref_struct = ref_struct.copy_with_new_coords(gt_coords.cpu().numpy())
    return ref_struct


# ============================================================
# Chain permutation alignment
# ============================================================
def _do_optimal_chain_permutation(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    chain_symmetries: dict[str, ChainSymmetry],
) -> RefStructure:
    """Compute minimum RMSD coordinates considering chain permutation.

    This implements a greedy trial-based alignment for multi-chain complexes,
    similar to the AlphaFold-Multimer approach.
    """
    if all(len(group) == 1 for group in chain_symmetries.values()):
        # No swappable chains exist.
        return ref_struct

    device = pred_coords.device
    gt_coords = torch.as_tensor(ref_struct.get_atom_coords(), device=device)

    chains = ref_struct.chains
    asym_ids = [aid for gs in chain_symmetries.values() for g in gs for aid in g]
    assert sorted(asym_ids) == sorted(c.asym_id for c in chains), (
        "Mismatch in number of chains between swappable groups and reference structure."
    )

    # Extract chain center coordinates for coarse-grain layout comparison.
    dev = pred_coords.device
    gt_chain_coords_dict: dict[int, torch.Tensor] = {}
    gt_center_coords_dict: dict[int, torch.Tensor] = {}
    pred_center_coords_dict: dict[int, torch.Tensor] = {}
    st = 0
    for c in ref_struct.chains:
        if c.is_protein:
            center_indices = np.where(c.atom.name == "CA")[0]
        elif c.is_nucleic_acid:
            center_indices = np.where(c.atom.name == "C1'")[0]
        else:
            # For ligands, use all atoms for centroid calculation.
            center_indices = np.arange(c.num_atoms)

        end = st + c.num_atoms
        center_atom_indices = torch.as_tensor(center_indices + st, device=dev)
        gt_chain_coords_dict[c.asym_id] = gt_coords[st:end]
        gt_center_coords_dict[c.asym_id] = gt_coords[center_atom_indices]
        pred_center_coords_dict[c.asym_id] = pred_coords[center_atom_indices]
        st = end
    del st

    # Prioritize chain groups (typically polymers) to act as the alignment anchor.
    anchor: ChainGroup = _get_anchor_chain_group(chains, chain_symmetries)

    # Find the optimal mapping from Predicted asym_ids to Ground Truth asym_ids.
    pred_to_gt_mapping: dict[int, int] = _find_multi_chain_permutation(
        anchor,
        gt_center_coords_dict,
        pred_center_coords_dict,
        chain_symmetries,
    )
    assert (
        set(pred_to_gt_mapping.keys())
        == set(pred_to_gt_mapping.values())
        == set(c.asym_id for c in chains)
    ), "Invalid chain mapping: keys and values must match the set of chain asym_ids."

    # Reconstruct the swapped GT structure based on the optimal mapping.
    asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in chains}
    new_chains: list[Chain] = [
        asym_id_to_chain[pred_to_gt_mapping[c.asym_id]] for c in chains
    ]
    ref_struct_permuted = RefStructure(
        chains=new_chains,
        connections=ref_struct.connections,
        metadata=ref_struct.metadata,
    )
    return ref_struct_permuted


def _get_anchor_chain_group(
    chains: list[Chain], chain_symmetries: dict[str, ChainSymmetry]
) -> ChainGroup:
    """Select the most stable chain group to serve as the alignment anchor.

    Priority:
      1. Polymer chains (Protein/Nucleic Acid)
      2. Chains with the least symmetry (fewer trials)
      3. Chains with more resolved atoms
      4. Longer chains
    """
    asym_ids: list[int] = [c.asym_id for c in chains]

    # Priority factors:
    ctype_dict: dict[int, int] = {c.asym_id: 1 if c.is_polymer else 0 for c in chains}
    nsym_dict: dict[int, int] = {
        aid: len(gs) for gs in chain_symmetries.values() for g in gs for aid in g
    }

    def get_n_resolved_atoms(c: Chain) -> int:
        if c.is_protein:
            center_coords = c.atom.coords[c.atom.name == "CA"]
        elif c.is_nucleic_acid:
            center_coords = c.atom.coords[c.atom.name == "C1'"]
        else:
            center_coords = c.atom.coords
        return np.isfinite(center_coords).all(-1).sum()

    nvalid_dict: dict[int, int] = {c.asym_id: get_n_resolved_atoms(c) for c in chains}
    len_dict: dict[int, int] = {c.asym_id: c.num_residues for c in chains}

    # Rank chains by priority tuple.
    priority = {
        v: (ctype_dict[v], -nsym_dict[v], nvalid_dict[v], len_dict[v], v)
        for v in asym_ids
    }
    anchor: Chain = max(chains, key=lambda c: priority[c.asym_id])

    # Return the symmetry group containing the best anchor chain.
    for sym in chain_symmetries.values():
        for group in sym:
            if anchor.asym_id in group:
                return group
    raise RuntimeError("Anchor chain group not found in symmetries.")


def _find_multi_chain_permutation(
    anchor_gt: ChainGroup,
    gt_coords_dict: dict[int, torch.Tensor],
    pred_coords_dict: dict[int, torch.Tensor],
    chain_symmetries: dict[str, ChainSymmetry],
) -> dict[int, int]:
    """Find the optimal chain mapping using a greedy search across anchor trials.
    See Algorithm 3 in the AlphaFold-Multimer paper.
    """
    buckets: list[ChainSymmetry] = [
        chain_symmetries[bucket_id] for bucket_id in sorted(chain_symmetries.keys())
    ]
    group_to_bucket: dict[ChainGroup, ChainSymmetry] = {g: b for b in buckets for g in b}

    # Compute static centroids for layout comparison.
    # Note: Using nanmean for GT to handle partially unresolved chains.
    pred_com_dict: dict[int, torch.Tensor] = {
        i: coords.mean(dim=-2) for i, coords in pred_coords_dict.items()
    }
    gt_com_dict: dict[int, torch.Tensor] = {
        i: coords.nanmean(dim=-2) for i, coords in gt_coords_dict.items()
    }
    if any(torch.isnan(com).any() for com in gt_com_dict.values()):
        raise RuntimeError(
            "All chains must have at least one resolved center atom for alignment."
        )

    best_total_cost = float("inf")
    best_pred_to_gt_mapping: dict[int, int] = {i: i for i in gt_coords_dict.keys()}

    # Anchor Ground Truth coordinates (static reference for trials).
    x_gt_anchor = gt_coords_dict[anchor_gt[0]]
    mask_anchor = x_gt_anchor.isfinite().all(dim=-1)

    # Trial loop: Try aligning the GT anchor to each of its predicted symmetry mates.
    anchor_pred_list: list[ChainGroup] = group_to_bucket[anchor_gt]
    for anchor_pred in anchor_pred_list:
        # Step 1: Compute rigid alignment (GT -> Pred) based on current anchor trial.
        x_pred_anchor = pred_coords_dict[anchor_pred[0]]
        RT, T = get_rigid_transform(x_gt_anchor, x_pred_anchor, mask_anchor)

        # Step 2: Apply alignment to all GT centroids.
        gt_com_dict_aligned: dict[int, torch.Tensor] = {
            i: com @ RT + T for i, com in gt_com_dict.items()
        }

        # Step 3: Greedily assign remaining chains in each symmetry bucket.
        total_cost: float = 0.0
        pred_to_gt: dict[int, int] = {}
        for bucket in buckets:
            rep_aids = [g[0] for g in bucket]
            rep_aid_to_group = {g[0]: g for g in bucket}

            _com_pred = torch.stack([pred_com_dict[i] for i in rep_aids])
            _com_gt = torch.stack([gt_com_dict_aligned[i] for i in rep_aids])

            if len(bucket) > 1:
                # Algorithm 4: Greedy bipartite matching.
                perm = _find_optimal_chain_permutation(_com_pred, _com_gt)
            else:
                perm = [0]

            # Map all chains in the linked groups (e.g., polymer + linked ligand).
            for s, t in enumerate(perm):
                g_s, g_t = rep_aid_to_group[rep_aids[s]], rep_aid_to_group[rep_aids[t]]
                pred_to_gt.update({i: j for i, j in zip(g_s, g_t, strict=True)})

            # Cumulative distance between centroids serves as the cost for this trial.
            matched_dist = torch.norm(_com_pred - _com_gt[perm], dim=-1).sum().item()
            total_cost += matched_dist

        # Update best global mapping if this trial yields lower total distance.
        if total_cost < best_total_cost:
            best_total_cost, best_pred_to_gt_mapping = total_cost, pred_to_gt

    return best_pred_to_gt_mapping


def _find_optimal_chain_permutation(
    com_pred: torch.Tensor, com_gt: torch.Tensor
) -> list[int]:
    """Greedily find the optimal permutation of GT groups to Pred groups.
    See Algorithm 4 in the AlphaFold-Multimer paper.
    """
    # Compute distance matrix between all pairs of centroids [Ns, Nt].
    d = torch.norm(com_pred[:, None, :] - com_gt[None, :, :], dim=-1)

    perm: list[int] = [-1] * com_gt.shape[0]
    for s in range(com_pred.shape[0]):
        # Match each Predicted group to the closest remaining Ground Truth group.
        best_idx = int(torch.argmin(d[s]).item())
        d[:, best_idx] = float("inf")
        perm[s] = best_idx

    assert all(i >= 0 for i in perm), "Failed to assign all GT groups."
    return perm


# ============================================================
# Geometric alignment.
# ============================================================
def _rigid_align_gt_to_pred(
    ref_struct: RefStructure,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
) -> torch.Tensor:
    """Rigidly align GT coordinates to Pred coordinates using Kabsch algorithm.

    NOTE:
    - We use polymer residue center atoms (CA/C1') for alignment to provide a
      stable reference frame that is invariant to side-chain/ligand flips.
    - This allows us to perform atomic permutations without repeating the
      SVD-based global alignment.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    gt_coords : torch.Tensor
        The ground truth coordinates. Shape: [Natom, 3]
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natom, 3]

    Returns
    -------
    gt_coords_aligned: torch.Tensor
        The aligned ground truth coordinates.
    """

    def get_is_center(c: Chain) -> np.ndarray:
        if c.is_protein:
            return c.atom.name == "CA"
        elif c.is_nucleic_acid:
            return c.atom.name == "C1'"
        else:
            return np.zeros(c.num_atoms, dtype=np.bool_)

    # Gather indices of backbone centers across all chains.
    is_center = np.concatenate([get_is_center(c) for c in ref_struct.chains])
    center_indices: np.ndarray = np.where(is_center)[0]
    assert len(center_indices) >= 3, "Insufficient center atoms for stable alignment."

    anchor_indices: torch.Tensor = torch.as_tensor(
        center_indices, device=gt_coords.device
    )
    mask = gt_coords.isfinite().all(dim=-1)

    gt_coords_aligned = rigid_align(gt_coords, pred_coords, mask, anchor_indices)
    gt_coords_aligned[~mask] = torch.nan
    return gt_coords_aligned


# ============================================================
# Atom permutation alignment
# ============================================================
def _do_optimal_atom_permutation(
    ref_struct: RefStructure,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    permutations: dict[int, list[ResidueSymmetry]],
) -> torch.Tensor:
    """Find the best residue/molecule permutation to minimize RMSD.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
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
    assert ref_struct.num_atoms == gt_coords.shape[0] == pred_coords.shape[0]
    atom_index = torch.arange(gt_coords.shape[0], device=gt_coords.device)
    gt_mask = gt_coords.isfinite().all(dim=-1)

    chain_atom_st = 0
    for c in ref_struct.chains:
        perms_in_chain = permutations[c.asym_id]

        for res_i in range(c.num_residues):
            perms: np.ndarray | None = perms_in_chain[res_i]
            if perms is None or len(perms) <= 1:
                continue

            st = chain_atom_st + c.residue.atom_starts[res_i]
            end = chain_atom_st + c.residue.atom_ends[res_i]

            x = pred_coords[st:end]
            x_gt = gt_coords[st:end]
            m = gt_mask[st:end]
            if not m.any():
                continue
            best_perm = __do_optimal_atom_permutation_in_residue(x_gt, x, m, perms)
            atom_index[st:end] = atom_index[st:end][best_perm]
        chain_atom_st = chain_atom_st + c.num_atoms

    gt_coords_permuted = gt_coords[atom_index]
    return gt_coords_permuted


def __do_optimal_atom_permutation_in_residue(
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    permutations: ResidueSymmetry,
) -> torch.Tensor:
    """Resolve residue-level atom permutations (e.g., side-chain flips).

    This function uses a vectorized Mean Squared Error check for each residue
    to find the optimal atom mapping in the current pre-aligned frame.
    """
    assert permutations is not None
    assert permutations.shape[1] == gt_coords.shape[0], "Permutation index mismatch."

    # Convert permutations to tensor for indexing.
    perms_t = torch.as_tensor(permutations, device=gt_coords.device)

    # Compute squared distances between Ground Truth and Predicted coordinates.
    x_pred = pred_coords[None, ...]  # [1, Natoms, 3]
    x_gt = gt_coords[perms_t]  # [Nperms, Natoms, 3]
    mask = gt_mask[perms_t]  # [Nperms, Natoms]
    diff_sq = (x_gt - x_pred).pow(2).sum(-1)
    # Mask out unresolved atoms to prevent NaN propagation.
    diff_sq[~mask] = 0.0
    # The permutation with the minimum sum of squared distances is selected.
    cost = diff_sq.sum(-1)
    best_i = int(torch.argmin(cost).item())
    return perms_t[best_i]
