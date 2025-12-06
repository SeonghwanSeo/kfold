import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.geometry.rigid_align import rigid_align
from kfold.utils.misc import pad_dim

from .symmetry import ResidueSymmetry

__all__ = ["get_aligned_true_coords"]


def compute_mse_loss(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute mean squared error loss with masking.

    Parameters
    ----------
    a : torch.Tensor
        First tensor. Shape: [..., Natoms, 3]
    b : torch.Tensor
        Second tensor. Shape: [..., Natoms, 3]
    mask : torch.Tensor
        Mask tensor. Shape: [..., Natoms]

    Returns
    -------
    loss : torch.Tensor
        The mean squared error loss.
    """
    diff = (a - b) * mask[..., None]
    n_resolved_atoms = mask.sum(dim=-1).clamp(min=1)
    loss = torch.sum(diff**2, dim=(-1, -2)) / n_resolved_atoms
    return loss


def find_best_chain_permutation(
    coords: torch.Tensor,
    center_index: torch.Tensor,
    all_alt_gt_coords: torch.Tensor,
    all_alt_resolved_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minimum RMSD coordinates considering chain permutation.

    Parameters
    ----------
    coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    center_index : torch.Tensor
        The center atom index for each token. Shape: [Ntoken]
    all_alt_gt_coords : torch.Tensor
        The alternative ground truth coordinates. Shape: [Nsym, Natoms, 3]
    all_alt_resolved_mask : torch.Tensor
        The resolved mask of alternative ground truth coordinates. Shape: [Nsym, Natoms]

    Returns
    -------
    gt_coords_aligned: torch.Tensor
        The aligned ground truth coordinates with minimum RMSD. Shape: [Natoms, 3]
    gt_resolved_mask: torch.Tensor
        The resolved mask of aligned ground truth coordinates. Shape: [Natoms]
    """

    num_symmetries = all_alt_gt_coords.shape[0]
    assert num_symmetries > 0, "There should be at least one symmetry (itself)."

    # 1. Find the best symmetry using coarse-grained alignment on center atoms
    center_coords = coords[center_index]  # [Ntoken, 3]
    gt_center_coords = all_alt_gt_coords[:, center_index, :]  # [Nsym, Ntoken, 3]
    center_mask = all_alt_resolved_mask[:, center_index]  # [Nsym, Ntoken]

    if num_symmetries == 1:
        # Only one symmetry (itself), skip search
        best_symmetry_index = 0
    elif not center_coords.isfinite().all():
        # If predicted center coords contain NaN or inf, skip symmetry search
        best_symmetry_index = 0
    else:
        best_symmetry_index: int = -1
        best_mse: float = float("inf")
        for s_i in range(num_symmetries):
            if not center_mask[s_i].any():
                # Skip if no resolved atoms
                continue
            try:
                gt_center_coords_aligned_i = rigid_align(
                    coords=gt_center_coords[s_i],
                    target=center_coords,
                    mask=center_mask[s_i],
                )
            except Exception as e:
                print("Warning: error in rigid alignment inside symmetry code: ", e)
                continue
            mse_i = compute_mse_loss(
                gt_center_coords_aligned_i, center_coords, center_mask[s_i]
            ).item()
            if mse_i < best_mse:
                best_mse = mse_i
                best_symmetry_index = s_i
        assert best_symmetry_index >= 0, "No valid symmetry found."

    # 2. Align the best symmetry with all atoms
    # NOTE: Since token-wise alighment can be overfitted to ligand atoms,
    # we perform a full-atom rigid alignment here instead of token-wise alignment.
    gt_coords = all_alt_gt_coords[best_symmetry_index]
    gt_resolved_mask = all_alt_resolved_mask[best_symmetry_index]
    gt_coords_aligned = rigid_align(
        coords=gt_coords,
        target=coords,
        mask=gt_resolved_mask,
    )
    return gt_coords_aligned, gt_resolved_mask


def find_best_mol_permutation(
    coords: torch.Tensor,
    gt_coords_aligned: torch.Tensor,
    gt_resolved_mask: torch.Tensor,
    mol_symmetries: list[ResidueSymmetry],
    align_coords: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minimum RMSD coordinates considering residue/molecule permutation.

    Parameters
    ----------
    coords : torch.Tensor
        The predicted coordinates. Shape: [Natoms, 3]
    gt_coords_aligned : torch.Tensor
        The aligned ground truth coordinates. Shape: [Natoms, 3]
    gt_resolved_mask : torch.Tensor
        The resolved mask of aligned ground truth coordinates. Shape: [Natoms]
    mol_symmetries : list[ResidueSymmetry]
        The list of atom swaps for residues/molecules.
    align_coords : bool, optional
        Whether to align coordinates after each swap (default: False).

    Returns
    -------
    gt_coords_aligned: torch.Tensor
        The aligned ground truth coordinates with minimum RMSD. Shape: [Natoms, 3]
    gt_resolved_mask: torch.Tensor
        The resolved mask of aligned ground truth coordinates. Shape: [Natoms]
    """

    if len(mol_symmetries) == 0:
        return gt_coords_aligned, gt_resolved_mask

    gt_coords = gt_coords_aligned.clone()
    gt_mask = gt_resolved_mask.clone()

    # Use MSE instead of RMSD for efficiency
    best_mse = compute_mse_loss(coords, gt_coords, gt_mask).item()

    # Inplace swap function to avoid extra memory allocation
    def swap(a: torch.Tensor, src: list[int], dst: list[int]) -> torch.Tensor:
        """Inplace swap of elements in tensor a at indices src and dst."""
        out = a.clone()
        out[dst] = a[src]
        return out

    # Find the best permutation greedily
    for atom_swaps in mol_symmetries:
        # Try all swaps and find the best one
        best_swap_i = -1
        best_mse_for_swap = best_mse
        for s_i, (src, dst) in enumerate(atom_swaps):
            # Swap atoms
            gt_coords_swap = swap(gt_coords, src, dst)
            gt_mask_swap = swap(gt_mask, src, dst)

            # Compute MSE after swap
            if align_coords:
                # Align after swap
                gt_coords_swap = rigid_align(
                    coords=gt_coords_swap, target=coords, mask=gt_mask_swap
                )
            mse = compute_mse_loss(coords, gt_coords_swap, gt_mask_swap).item()

            # Update best swap
            if mse < best_mse_for_swap:
                best_mse_for_swap = mse
                best_swap_i = s_i

        # Apply the best swap if it improves MSE
        if best_swap_i >= 0:
            src, dst = atom_swaps[best_swap_i]
            gt_coords = swap(gt_coords, src, dst)
            gt_mask = swap(gt_mask, src, dst)
            if align_coords:
                # Align after swap
                gt_coords = rigid_align(coords=gt_coords, target=coords, mask=gt_mask)
            best_mse = best_mse_for_swap

    return gt_coords, gt_mask


def get_aligned_true_coords(
    coords: torch.Tensor,
    f_input: FoldingInput,
    symmetry_dict: dict,
    index_batch: int,
):
    """Compute minimum RMSD coordinates considering symmetries.
    Parameters
    ----------
    coords : torch.Tensor
        The predicted coordinates. Shape: [Nsample, Natoms, 3]
    f_input : FoldingInput
        The folding input containing features.
        NOTE: This includes all samples in the batch.
    symmetry_dict : dict
        The dictionary containing symmetry information:
            - "alt_coordinates": torch.Tensor of shape [Nsym, Natoms, 3]
            - "alt_resolved_mask": torch.Tensor of shape [Nsym, Natoms]
            - "residue_symmetries": list[ResidueSymmetry]
            - "molecule_symmetries": list[ResidueSymmetry]
    index_batch : int
        The batch index.
    """
    # Remove padding
    original_num_atoms = f_input.num_atoms
    num_valid_atoms = f_input.atom.pad_mask[index_batch].sum().item()
    coords = coords[:, :num_valid_atoms, :]  # Remove padding

    # For efficient backbone alignment
    num_valid_tokens = f_input.token.pad_mask[index_batch].sum()
    center_index = f_input.token.center_index[index_batch][:num_valid_tokens]

    # For chain permutation
    all_alt_gt_coords = symmetry_dict["alt_coordinates"].to(coords.device)
    all_alt_resolved_mask = symmetry_dict["alt_resolved_mask"].to(coords.device)
    assert all_alt_gt_coords.shape[1] == num_valid_atoms, (
        "Mismatch in number of atoms between predicted coords and alt gt coords."
    )

    # For residue/molecule permutation
    residue_symmetries = symmetry_dict["residue_symmetries"]
    molecule_symmetries = symmetry_dict["molecule_symmetries"]

    num_samples = coords.shape[0]
    gt_coords_list = []
    gt_resolved_mask_list = []
    for i_sample in range(num_samples):
        coords_i = coords[i_sample]  # [Natoms, 3]

        # find the best chain permutation
        gt_coords_i, gt_resolved_mask_i = find_best_chain_permutation(
            coords_i,  # [Natoms, 3]
            center_index,  # [Ntoken]
            all_alt_gt_coords,  # [Nsym, Natoms, 3]
            all_alt_resolved_mask,  # [Nsym, Natoms]
        )  # [Natoms, 3], [Natoms]

        # find the best residue permutation (skip rigid alignment)
        gt_coords_i, gt_resolved_mask_i = find_best_mol_permutation(
            coords_i,
            gt_coords_i,
            gt_resolved_mask_i,
            residue_symmetries,
            align_coords=False,
        )  # [Natoms, 3], [Natoms]

        # Rigid alignment after residue permutation
        gt_coords_i = rigid_align(
            coords=gt_coords_i, target=coords_i, mask=gt_resolved_mask_i
        )

        # find the best molecule permutation (with rigid alignment)
        # TODO: if rigid alignment is bottleneck, consider skipping it here
        gt_coords_i, gt_resolved_mask_i = find_best_mol_permutation(
            coords_i,
            gt_coords_i,
            gt_resolved_mask_i,
            molecule_symmetries,
            align_coords=True,
        )  # [Natoms, 3], [Natoms]

        gt_coords_list.append(gt_coords_i)
        gt_resolved_mask_list.append(gt_resolved_mask_i)

    gt_coords_aligned = torch.stack(gt_coords_list, dim=0)  # [Nsample, Natoms, 3]
    gt_resolved_mask = torch.stack(gt_resolved_mask_list, dim=0)  # [Nsample, Natoms]

    # Pad back to original number of atoms
    gt_coords_aligned = pad_dim(
        gt_coords_aligned, dim=1, max_len=original_num_atoms, pad_value=0.0
    )
    gt_resolved_mask = pad_dim(
        gt_resolved_mask, dim=1, max_len=original_num_atoms, pad_value=False
    )

    return gt_coords_aligned, gt_resolved_mask
