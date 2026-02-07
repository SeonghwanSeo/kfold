import numpy as np
import torch

from kfold.data.types.structure import RefStructure
from kfold.utils.geometry.rigid_align import compute_rmsd, rigid_align

from .symmetry import ResidueSymmetry

__all__ = ["get_aligned_true_coords"]


def get_aligned_true_coords(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    find_best_permutation: bool = True,
    symmetry_dict: dict | None = None,
) -> RefStructure:
    """Get aligned ground truth coordinates with minimum RMSD considering symmetries.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    find_best_permutation : bool, optional
        Whether to find the best permutation (default: True).
    symmetry_dict : dict, optional
        The dictionary containing symmetry information:
            - "chain": list[list[int]]
            - "residue": list[list[int] or None]

    Returns
    -------
    ref_struct_aligned: RefStructure
        The reference structure with aligned ground truth coordinates.
    """
    # Remove padding
    original_num_atoms = ref_struct.num_atoms
    assert pred_coords.shape == (original_num_atoms, 3), (
        "Mismatch in number of atoms between coordinates and reference structure."
    )
    ref_struct = ref_struct.clone()
    if find_best_permutation:
        assert symmetry_dict is not None, (
            "Symmetry dictionary must be provided when find_best_permutation is True."
        )
        # Chain-swap for chain permutation
        ref_struct = _find_best_chain_permutation(
            ref_struct,
            pred_coords,
            symmetry_dict["chain"],
            deepcopy=False,
        )
        # Atom-swap for residue/molecule permutation
        _find_best_residue_permutation(
            ref_struct,
            pred_coords,
            symmetry_dict["residue"],
            align_coords=False,
            align_local_coords=True,
            deepcopy=False,
        )
    else:
        # Deepcopy the reference structure
        ref_struct = ref_struct.clone()

    # Create aligned coordinates
    dev = pred_coords.device
    gt_coords = torch.from_numpy(ref_struct.get_atom_coords()).to(dev)
    gt_mask = gt_coords.isfinite().all(dim=-1)
    aligned_gt_coords = rigid_align(gt_coords, pred_coords, gt_mask)
    aligned_gt_coords[~gt_mask] = float("nan")

    # Update the reference structure with aligned coordinates
    ref_struct = ref_struct.copy_with_new_coords(aligned_gt_coords.cpu().numpy())

    return ref_struct


def _find_best_chain_permutation(
    ref_struct: RefStructure,
    coords: torch.Tensor,
    permutations: list[list[int]],
    deepcopy: bool = True,
) -> RefStructure:
    """Compute minimum RMSD coordinates considering chain permutation.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    permutations : list[list[int]]
        The list of chain permutations (asym_id indices).

    Returns
    -------
    RefStructure
        reference structure with best chain permutation applied.
    """
    dev = coords.device

    if deepcopy:
        ref_struct = ref_struct.clone()

    if len(permutations) <= 1:
        # No chain permutation
        return ref_struct

    # Map asym_id to chain index
    new_permutations: list[list[int]] = []
    asym_id_to_chain_index = {c.asym_id: i for i, c in enumerate(ref_struct.chains)}
    for perm in permutations:
        new_permutations.append([asym_id_to_chain_index[asym_id] for asym_id in perm])
    permutations = new_permutations

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

    # NOTE: Assume the center index is static
    center_index = torch.cat(chain_center_indices_list, dim=0)  # [Ncenter]

    best_i: int = -1
    best_rmsd: float = float("inf")
    center_coords = coords[center_index]  # [Ncenter, 3]
    gt_center_coords = torch.empty_like(center_coords)  # [Ncenter, 3]
    gt_center_mask = torch.zeros(center_index.shape, dtype=torch.bool, device=dev)

    for perm_i, perm in enumerate(permutations):
        # Build permuted ground truth coordinates
        ptr = 0
        for c_i in perm:
            n_centers = chain_center_list[c_i].shape[0]
            gt_center_coords[ptr : ptr + n_centers] = chain_center_list[c_i]
            ptr += n_centers
        torch.all(gt_center_coords.isfinite(), dim=-1, out=gt_center_mask)
        # Compute RMSD on center atoms after rigid alignment
        rmsd = compute_rmsd(
            gt_center_coords, center_coords, gt_center_mask, align=True
        ).item()
        # Update best permutation
        if rmsd < best_rmsd:
            best_rmsd, best_i = rmsd, perm_i
    best_perm = permutations[best_i]

    # Build new RefStructure with permuted chains
    new_struct = RefStructure(
        chains=[ref_struct.chains[i] for i in best_perm],
        connections=ref_struct.connections,
        metadata=ref_struct.metadata,
    )
    return new_struct


def _find_best_residue_permutation(
    ref_struct: RefStructure,
    pred_coords: torch.Tensor,
    permutations: list[ResidueSymmetry],
    align_coords: bool = True,
    align_local_coords: bool = False,
    deepcopy: bool = True,
) -> RefStructure:
    """Find the best residue/molecule permutation to minimize RMSD.

    Parameters
    ----------
    ref_struct : RefStructure
        The reference structure containing ground truth coordinates and masks.
    pred_coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    permutations : list[ResidueSymmetry]
        The list of residue/molecule symmetries.
    align_coords : bool, optional
        Whether to align coordinates before swapping (default: False).
    align_local_coords : bool, optional
        Whether to align local coordinates when computing RMSD (default: False).
    deepcopy : bool, optional
        Whether to deepcopy the reference structure (default: True).
    """
    assert len(permutations) == ref_struct.num_residues, (
        "Mismatch in number of residues between permutations and reference structure."
    )
    if deepcopy:
        ref_struct = ref_struct.clone()

    if all(perm is None or len(perm) <= 1 for perm in permutations):
        # No residue/molecule permutation
        return ref_struct

    # Construct ground truth coordinates and mask
    dev = pred_coords.device
    gt_coords = torch.from_numpy(ref_struct.get_atom_coords()).to(dev)
    gt_mask = gt_coords.isfinite().all(dim=-1)  # [Natoms]

    # Align coordinates if needed
    if align_coords and not align_local_coords:
        gt_coords = rigid_align(gt_coords, pred_coords, gt_mask)

    # Find the best permutation greedily
    atom_st = 0
    perm_iterator = iter(permutations)

    for c in ref_struct.chains:
        num_atoms_per_residue: list[int] = c.residue.num_atoms.tolist()
        for res_i in range(c.num_residues):
            res_idx = res_i + 1  # 1-based residue index

            res_perms: list[int] | None = next(perm_iterator)
            atom_end = atom_st + num_atoms_per_residue[res_i]

            if res_perms is None or len(res_perms) <= 1:
                # No permutation for this residue
                atom_st = atom_end
                continue

            x = pred_coords[atom_st:atom_end]  # [Nres_atoms, 3]
            x_gt = gt_coords[atom_st:atom_end]  # [Nres_atoms, 3]
            m = gt_mask[atom_st:atom_end]  # [Nres_atoms]
            if not m.any():
                # No valid atoms to compare
                atom_st = atom_end
                continue

            # Try all swaps and find the best one
            best_i = -1
            best_rmsd = float("inf")
            for p_i, perm in enumerate(res_perms):
                _x_gt = x_gt[perm]  # [Nres_atoms, 3]
                _m = m[perm]  # [Nres_atoms]
                rmsd = compute_rmsd(
                    _x_gt, x, _m, align=align_local_coords, no_svd=True
                ).item()
                if rmsd < best_rmsd:
                    best_rmsd, best_i = rmsd, p_i

            # Apply the best swap
            if best_i >= 0:
                best_perm = res_perms[best_i]
                atom_slice = c.residue.get_atom_slice(res_idx)
                c.atom.coords[atom_slice] = c.atom.coords[atom_slice][best_perm]

            atom_st = atom_end

    return ref_struct
