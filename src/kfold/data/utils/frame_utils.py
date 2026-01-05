import numpy as np
import torch

from kfold.data.types import model_input


def compute_collinear_mask(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_ligand_frames_inplace(
    token_layout: model_input.TokenTensor,
    atom_layout: model_input.AtomTensor,
    chain_layout: model_input.ChainTensor,
):
    """Update frames for non-polymer chains."""

    num_chains = len(chain_layout)
    is_ligand = chain_layout.is_ligand
    num_tokens = chain_layout.num_tokens
    num_atoms = chain_layout.num_atoms
    token_starts = num_tokens.cumsum(0) - num_tokens
    atom_starts = num_atoms.cumsum(0) - num_atoms

    frames_index = token_layout.frames_index  # (Nt, 3)
    frames_mask = token_layout.frames_mask  # (Nt,)

    for i in range(num_chains):
        if not is_ligand[i] or num_atoms[i] < 3:
            continue

        assert num_atoms[i] == num_tokens[i], (
            "Ligand chain should have equal number of tokens and atoms."
        )
        atom_st, atom_end = atom_starts[i], atom_starts[i] + num_atoms[i]
        token_st, token_end = token_starts[i], token_starts[i] + num_tokens[i]

        coords = atom_layout.label_coords[atom_st:atom_end].reshape(-1, 3)
        dist_mat = torch.cdist(coords, coords, p=2)

        # Mask out unresolved atoms
        resolved_mask = atom_layout.resolved_mask[atom_st:atom_end]
        resolved_pair = resolved_mask[None, :] & resolved_mask[:, None]
        dist_mat = torch.where(resolved_pair, dist_mat, np.inf)

        # Get nearest neighbors
        # NOTE: the first index and third index are interchangeable
        indices = dist_mat.argsort(dim=1)
        frames = torch.stack([indices[:, 1], indices[:, 0], indices[:, 2]], dim=1)

        frames_index[token_st:token_end] = frames + atom_st
        frames_mask[token_st:token_end] = resolved_mask[frames].all(dim=1)

    coords = atom_layout.label_coords.reshape(-1, 3)
    frames_expanded = coords[frames_index]  # (Nt, 3, 3)
    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    frames_mask[~mask_collinear] = False
    frames_mask[~token_layout.resolved_mask] = False
