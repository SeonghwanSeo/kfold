from collections.abc import Callable
from functools import partial
from typing import TypeVar

import torch

_T = TypeVar("_T")


# === Atom-Token mapping functions === #
def broadcast_tokens_to_atoms(x: torch.Tensor, token_index: torch.Tensor) -> torch.Tensor:
    """Broadcast token features to atom features.

    Parameters
    ----------
    x: torch.Tensor
        Token features of shape (*, Ntoken) or (*, Ntoken, D)
    token_index: torch.Tensor
        Tensor of shape (*, Natom) mapping each atom to a token index.

    Returns
    -------
    x_atom: torch.Tensor
        Atom features of shape (*, Natom, D)
    """
    if x.ndim == token_index.ndim:
        return broadcast_tokens_to_atoms(x.unsqueeze(-1), token_index).squeeze(-1)

    # Expand indices to match the input dimensions.
    gather_shape = list(x.shape)
    gather_shape[-2] = token_index.shape[-1]  # Natom
    index_expanded = token_index.unsqueeze(-1).expand(*gather_shape)  # [*, Natom, D]

    # Gather token features for each atom based on the token index.
    out = torch.gather(x, dim=-2, index=index_expanded)  # [*, Natom, D]
    return out


def aggregate_atoms_to_tokens(
    x: torch.Tensor, token_index: torch.Tensor, mask: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Aggregate atom features to token features. (mean pooling)

    Parameters
    ----------
    x: torch.Tensor
        Atom features of shape (*, Natom, D)
    token_index: torch.Tensor
        Tensor of shape (*, Natom) mapping each atom to a token index.
    mask: torch.Tensor
        Tensor of shape (*, Natom) indicating valid atoms
    num_tokens: int
        The number of tokens.

    Returns
    -------
    x_token: torch.Tensor
        Token features of shape (*, Ntoken, D)
    """
    # Prepare indices
    trash_idx = num_tokens  # An out-of-range index for padding atoms.
    index = torch.where(mask, token_index, trash_idx)
    index_expanded = index.unsqueeze(-1).expand(*x.shape)

    # Prepare an output tensor with an extra slot for padding atoms
    out_shape = list(x.shape)
    out_shape[-2] = num_tokens + 1  # Add an extra slot for padding atoms

    # Scatter reduce atom features to token features
    with torch.autocast(x.device.type, enabled=False):
        out = torch.zeros(
            *out_shape, dtype=torch.float32, device=x.device
        ).scatter_reduce_(
            dim=-2, index=index_expanded, src=x.float(), reduce="mean", include_self=False
        )  # [*, Ntoken+1, D]

    # Remove the extra slot for padding atoms
    out = out[..., :num_tokens, :]
    return out.to(x.dtype).contiguous()


# === Local Attention Indexing === #
def build_atom_to_qk_fn(
    length: int, device: str | torch.device
) -> Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]:
    """Build indices for window-based local attention from atoms to query windows.

    Parameters
    ----------
    length: int
        The sequence length (number of atoms).
    device: str | torch.device
        The device on which to create the index tensors.

    Returns
    -------
    Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
        A function that takes an input tensor and a dimension, and returns the
        query and key tensors for local attention.
    """
    Lq, Lk = 32, 128  # noqa
    if length % 32 != 0:
        raise ValueError("Length must be divisible by 32")

    W: int = length // 32
    half_block_size = 16
    num_half_blocks = 2 * W
    h = 8  # 128 // 16

    # Block Logic
    start_offset = -(h // 2) + 1
    block_offsets = torch.arange(h, device=device) + start_offset
    window_starts = torch.arange(W, device=device).unsqueeze(-1) * 2
    block_indices = window_starts + block_offsets  # [W, h]

    # Pad mask for out-of-bounds
    pad_mask = (block_indices < 0) | (block_indices >= num_half_blocks)
    # [W, h] -> [W, Lk]
    pad_mask = pad_mask.repeat_interleave(half_block_size, dim=-1)

    # Clamp block indices to valid range
    block_indices = block_indices.clamp(min=0, max=num_half_blocks - 1)

    # Expand block indices to atom indices
    atom_offsets = torch.arange(half_block_size, device=device)

    # Broadcasting to construct full [W, Lk] index matrix
    gather_indices = (
        block_indices[..., None] * half_block_size + atom_offsets[None, None, ...]
    )
    gather_indices = gather_indices.view(W, Lk)

    func = partial(convert_atom_to_qk, gather_indices=gather_indices, pad_mask=pad_mask)
    return func


def convert_atom_to_qk(
    x: torch.Tensor,
    dim: int,
    gather_indices: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window-based indexing for local attention.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of shape [*, L, *]
    dim: int, optional
        Dimension along which to unflatten into query/key windows.

    Returns
    -------
    torch.Tensor
        Query tensor of shape [*, W, Lq, *]
    torch.Tensor
        Key tensor of shape [*, W, Lk, *]
    """
    W = gather_indices.shape[0]
    dim = dim % x.ndim

    # Get query
    x_q = x.unflatten(dim, sizes=(W, 32))

    # Get keys using gather indices
    x_k = x.index_select(dim, gather_indices.view(-1))
    x_k = x_k.unflatten(dim, sizes=(W, 128))

    # Apply Padding Mask
    mask_shape = [1] * x_k.ndim
    mask_shape[dim] = gather_indices.shape[0]  # W
    mask_shape[dim + 1] = gather_indices.shape[1]  # Lk
    x_k = x_k.masked_fill(pad_mask.view(*mask_shape), 0)
    return x_q, x_k
