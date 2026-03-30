import math
from collections.abc import Callable
from functools import partial
from typing import TypeVar, overload

import torch
from torch.types import Device

_T = TypeVar("_T")


def expand_dim(x: torch.Tensor, dim: int, n: int, add_dim: bool) -> torch.Tensor:
    """Expands a tensor x"""
    if add_dim:
        x = x.unsqueeze(dim)
    shape = [-1] * x.ndim
    shape[dim] = n
    return x.expand(shape)


def repeat_dim(x: torch.Tensor, dim: int, n: int, add_dim: bool) -> torch.Tensor:
    """Repeats a tensor x"""
    if add_dim:
        x = x.unsqueeze(dim)
    shape = [1] * x.ndim
    shape[dim] *= n
    return x.repeat(shape)


def exists(v) -> bool:
    return v is not None


def default(v: _T | None, d: _T) -> _T:
    return v if exists(v) else d  # type: ignore[return-value]


# === Atom-Token mapping functions === #
def broadcast_tokens_to_atoms(
    x: torch.Tensor,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """Broadcast token features to atom features.

    Parameters
    ----------
    x: torch.Tensor
        Token features of shape (*, Ntoken, D)
    token_index: torch.Tensor
        Tensor of shape (*, Natom) mapping each atom to a token index.

    Returns
    -------
    x_atom: torch.Tensor
        Atom features of shape (*, Natom, D)
    """

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
    # Prepare indices for scatter_reduce
    trash_idx = num_tokens  # An out-of-range index for padding atoms.
    index = torch.where(mask, token_index, trash_idx)
    index_expanded = index.unsqueeze(-1).expand(*x.shape)

    # Prepare an output tensor with an extra slot for padding atoms
    out_shape = list(x.shape)
    out_shape[-2] = num_tokens + 1  # Add an extra slot for padding atoms
    out = torch.zeros(*out_shape, dtype=x.dtype, device=x.device)  # [*, Ntoken + 1, D]

    # Scatter reduce atom features to token features
    out.scatter_reduce_(
        dim=-2, index=index_expanded, src=x, reduce="mean", include_self=False
    )
    # Remove the extra slot for padding atoms
    out = out[..., :num_tokens, :]
    return out.contiguous()


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


# === Centering and Random Augmentation === #
def center_random_augmentation(
    coords: torch.Tensor,
    mask: torch.Tensor,
    centering: bool = True,
    augmentation: bool = True,
    s_trans: float = 1.0,
    mask_to_zero: bool = True,
) -> torch.Tensor:
    """Centering and Random Augmentation
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """
    # Line 1
    if centering:
        coords = do_centering(coords, mask, mask_to_zero=False)

    if augmentation:
        # Line 2,4
        R = random_rotations(
            coords.shape[:-2], coords.dtype, coords.device
        )  # [..., 3, 3]
        coords = torch.einsum("...md,...ds->...ms", coords, R)  # noqa

        # Line 3,4
        if s_trans > 0.0:
            random_trans = torch.randn_like(coords[..., 0:1, :]) * s_trans
            coords = coords + random_trans

    # Mask out
    if mask_to_zero:
        coords = coords * mask[..., None]

    return coords


class CenterRandomAugmentation:
    """Centering and Random Augmentation Module
    See Section 3.7 Algorithm 19 CentreRandomAugmentation

    Usage)
    ```python
    augment = CenterRandomAugmentation(...)
    x = augment(x, mask=mask)
    x, y = augment(x, y, mask=mask)
    ```
    """

    def __init__(
        self,
        augmentation: bool = True,
        centering: bool = True,
        s_trans: float = 1.0,
        mask_to_zero: bool = True,
    ):
        self.augmentation: bool = augmentation
        self.centering: bool = centering
        self.s_trans: float = s_trans
        self.mask_to_zero: bool = mask_to_zero

    @overload
    def __call__(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def __call__(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.augment(*coords, mask=mask)

    @overload
    def augment(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def augment(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def augment(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """See Section 3.7 Algorithm 19 CentreRandomAugmentation

        Parameters
        ----------
        coords : torch.Tensor
            One or more tensors of shape (..., L, 3) representing atomic coordinates.
        mask : torch.Tensor
            A tensor of shape (..., L) representing the atom mask.
        """
        coords_list: list[torch.Tensor] = list(coords)
        # Check all input coords have the same batch size and number of atoms
        ref_coords = coords_list[0]
        coords_shape = ref_coords.shape
        for c in coords_list:
            assert c.shape == coords_shape, (
                "All input coordinate tensors must have the same batch size and length."
                f" Got {c.shape} vs {coords_shape}."
            )

        # Line 1
        if self.centering:
            coords_list = [do_centering(x, mask, mask_to_zero=False) for x in coords_list]

        if self.augmentation:
            # Line 2,4
            R = random_rotations(
                coords_shape[:-2], ref_coords.dtype, ref_coords.device
            )  # [..., 3, 3]
            rotate = lambda x: torch.einsum("...md,...ds->...ms", x, R)  # noqa
            coords_list = [rotate(x) for x in coords_list]

            # Line 3,4
            if self.s_trans > 0.0:
                random_trans = torch.randn_like(ref_coords[..., 0:1, :]) * self.s_trans
                coords_list = [x + random_trans for x in coords_list]

        # Mask out
        if self.mask_to_zero:
            coords_list = [x * mask[..., None] for x in coords_list]

        if len(coords) == 1:
            # Single tensor input, return tensor
            return coords_list[0]
        else:
            # Multiple tensor input, return list of tensors
            return tuple(coords_list)


def do_centering(
    coords: torch.Tensor, mask: torch.Tensor, mask_to_zero: bool = True
) -> torch.Tensor:
    """Centering of atom coordinates
    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (..., L, 3)
    mask : torch.Tensor
        Mask, shape (..., L)
    """

    total_mass = mask.sum(dim=-1, keepdim=True).clamp(1)
    center = (
        torch.sum(coords * mask[..., None], dim=-2, keepdim=True) / total_mass[..., None]
    )
    centered_coords = coords - center
    if mask_to_zero:
        centered_coords = centered_coords * mask[..., None]
    return centered_coords


# the following is copied from Torch3D, BSD License,
# Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    return torch.copysign(a, b)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    shape: tuple[int, ...], dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        shape: Shape of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (*shape, 3, 3).
    """
    n = math.prod(shape)
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)
