"""Permutation alignment for validation."""

import itertools
import math

import numpy as np
import torch

from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.geometry.rigid_align import compute_rmsd, rigid_align

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
    if not pred_coords.isfinite().all():
        raise RuntimeError("Predicted coordinates contain NaN or Inf values.")

    # FIX: Defensively convert lists back to tuples to avoid unhashable type errors
    # caused by PyTorch DataLoader collation.
    chain_syms = symmetry_dict["chain"]
    if isinstance(chain_syms, dict):
        chain_syms = {k: [tuple(g) for g in v] for k, v in chain_syms.items()}

    # Validation alignment is performed in double precision for stability.
    with torch.autocast(device.type, enabled=False), torch.no_grad():
        # 1. Multi-chain permutation alignment.
        # Finds the optimal mapping of swappable homomer subunits.
        gt_to_pred = _do_optimal_chain_permutation(
            ref_struct,
            pred_coords,
            chain_syms,
        )

        # Swap GT chains according to the optimal mapping.
        asym_id_to_chain = {c.asym_id: c for c in ref_struct.chains}

        # FIX: Use .get() because non-swapped chains won't exist in the gt_to_pred dict.
        new_chains = [
            asym_id_to_chain[gt_to_pred.get(c.asym_id, c.asym_id)]
            for c in ref_struct.chains
        ]

        ref_struct = RefStructure(
            chains=new_chains,
            connections=ref_struct.connections,
            metadata=ref_struct.metadata,
        )
        gt_coords = torch.as_tensor(ref_struct.get_atom_coords(), device=device)

        # 2. Rigid alignment.
        # Align GT to Pred using polymer backbone (CA/C1') centers.
        # Backbone alignment is robust to side-chain/ligand flips.
        gt_coords = _rigid_align_gt_to_pred(ref_struct, gt_coords, pred_coords)

        # 3. Atomic permutation alignment.
        # Resolves side-chain symmetries and small molecule atom swappability.
        atom_index = _do_optimal_atom_permutation(
            ref_struct,
            gt_coords,
            pred_coords,
            symmetry_dict["residue"],
        )
        # Update the reference structure with the new permuted/aligned coordinates.
        gt_coords = gt_coords[atom_index]

        # 4. Final alignment
        gt_coords = _rigid_align_gt_to_pred(
            ref_struct, gt_coords, pred_coords, center_atom_only=False
        )

        ref_struct = ref_struct.copy_with_new_coords(gt_coords.cpu().numpy())
        return ref_struct


# ============================================================
# Chain permutation alignment
# ============================================================
def _generate_chain_permutations(
    chain_symmetries: dict[str, ChainSymmetry],
    max_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> list[dict[int, int]]:
    """Generate possible chain permutations mapping original asym_id to target asym_id."""
    rng = rng or np.random.default_rng()
    sorted_keys = sorted(chain_symmetries.keys())
    bucket_anchors_list = [
        chain_symmetries[k] for k in sorted_keys if len(chain_symmetries[k]) > 1
    ]

    if len(bucket_anchors_list) == 0:
        return [{}]  # Identity only

    total_perms = 1
    for anchors in bucket_anchors_list:
        total_perms *= math.factorial(len(anchors))

    anchor_mappings: list[dict[ChainGroup, ChainGroup]] = []

    if total_perms <= max_permutations:
        # Exhaustive Search
        per_bucket_perms = [
            list(itertools.permutations(anchors)) for anchors in bucket_anchors_list
        ]
        for combination in itertools.product(*per_bucket_perms):
            mapping: dict[ChainGroup, ChainGroup] = {}
            for original_anchors, permuted_anchors in zip(
                bucket_anchors_list, combination, strict=True
            ):
                for old_anchor, new_anchor in zip(
                    original_anchors, permuted_anchors, strict=True
                ):
                    mapping[old_anchor] = new_anchor
            anchor_mappings.append(mapping)
    else:
        # Random Sampling
        identity_map: dict[ChainGroup, ChainGroup] = {}
        for anchors in bucket_anchors_list:
            for a in anchors:
                identity_map[a] = a
        anchor_mappings.append(identity_map)

        seen_sigs = set()
        all_swappable_keys = sorted(identity_map.keys())
        sig = tuple(identity_map[k] for k in all_swappable_keys)
        seen_sigs.add(sig)

        for _ in range(max_permutations * 10):
            mapping = {}
            for anchors in bucket_anchors_list:
                perm = list(anchors)
                rng.shuffle(perm)
                for old_anchor, new_anchor in zip(anchors, perm, strict=True):
                    mapping[old_anchor] = new_anchor

            sig = tuple(mapping[k] for k in all_swappable_keys)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                anchor_mappings.append(mapping)
            if len(anchor_mappings) >= max_permutations:
                break

    # Expand to full chain mappings
    final_results: list[dict[int, int]] = []
    for anchor_map in anchor_mappings:
        full_mapping: dict[int, int] = {}
        for old_group, new_group in anchor_map.items():
            for old_id, new_id in zip(old_group, new_group, strict=True):
                if old_id != new_id:
                    full_mapping[old_id] = new_id
        if len(full_mapping) > 0:
            final_results.append(full_mapping)

    final_results.sort(key=lambda x: sorted(x.items()))
    return [{}] + final_results


# FIX: Renamed from _find_best_chain_permutation to _do_optimal_chain_permutation
# to match the function call in get_aligned_gt_structure.
def _do_optimal_chain_permutation(
    ref_struct: RefStructure,
    coords: torch.Tensor,
    chain_symmetries: dict[str, ChainSymmetry],
    max_permutations: int = 1000,
) -> dict[int, int]:
    """Compute minimum RMSD coordinates considering chain permutation."""
    dev = coords.device

    mappings = _generate_chain_permutations(chain_symmetries, max_permutations)

    if len(mappings) <= 1:
        return {c.asym_id: c.asym_id for c in ref_struct.chains}  # Identity mapping

    chain_coords_list: list[torch.Tensor] = []
    chain_center_list: list[torch.Tensor] = []
    chain_center_indices_list: list[torch.Tensor] = []
    atom_st = 0

    for c in ref_struct.chains:
        chain_coords = torch.from_numpy(c.atom.coords).to(dev)
        if c.is_protein:
            center_indices = np.where(c.atom.name == "CA")[0]
        elif c.is_nucleic_acid:
            center_indices = np.where(c.atom.name == "C4'")[0]
        else:
            center_indices = np.arange(c.num_atoms)
        center_indices = torch.from_numpy(center_indices).to(dev)

        chain_coords_list.append(chain_coords)
        chain_center_list.append(chain_coords[center_indices])
        chain_center_indices_list.append(center_indices + atom_st)
        atom_st += c.num_atoms

    center_index = torch.cat(chain_center_indices_list, dim=0)

    best_i: int = -1
    best_rmsd: float = float("inf")
    center_coords = coords[center_index]
    gt_center_coords = torch.empty_like(center_coords)
    gt_center_mask = torch.zeros(center_index.shape, dtype=torch.bool, device=dev)

    asym_id_to_idx = {c.asym_id: i for i, c in enumerate(ref_struct.chains)}

    for perm_i, mapping in enumerate(mappings):
        ptr = 0
        for c in ref_struct.chains:
            mapped_asym_id = mapping.get(c.asym_id, c.asym_id)
            c_i = asym_id_to_idx[mapped_asym_id]
            n_centers = chain_center_list[c_i].shape[0]
            gt_center_coords[ptr : ptr + n_centers] = chain_center_list[c_i]
            ptr += n_centers

        torch.all(gt_center_coords.isfinite(), dim=-1, out=gt_center_mask)

        rmsd = compute_rmsd(
            gt_center_coords, center_coords, gt_center_mask, align=True
        ).item()

        if rmsd < best_rmsd:
            best_rmsd, best_i = rmsd, perm_i

    return mappings[best_i]


# ============================================================
# Geometric alignment.
# ============================================================
def _rigid_align_gt_to_pred(
    ref_struct: RefStructure,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    center_atom_only: bool = True,
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

    if center_atom_only:
        # Gather indices of backbone centers across all chains.
        is_center = np.concatenate([get_is_center(c) for c in ref_struct.chains])
        center_indices: np.ndarray = np.where(is_center)[0]
        num_center_resolved = (
            torch.isfinite(gt_coords[center_indices]).all(dim=-1).sum().item()
        )
        assert num_center_resolved >= 3, "Insufficient center atoms for stable alignment."
        anchor_indices: torch.Tensor = torch.as_tensor(
            center_indices, device=gt_coords.device
        )
    else:
        anchor_indices = None

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
    atom_index: torch.Tensor
        The optimal atom index mapping from GT to Pred after permutation.
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
    return atom_index


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
    # Intra-geometric alignment.
    cost = compute_rmsd(x_gt, x_pred, mask, align=True, no_svd=True)
    best_i = int(torch.argmin(cost).item())
    return perms_t[best_i]
