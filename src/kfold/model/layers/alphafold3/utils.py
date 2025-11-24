# started from code from https://github.com/jwohlwend/boltz, MIT License,

import math
from functools import lru_cache
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


def broadcast_tokens_to_atoms(
    x_token: torch.Tensor,
    token_index: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Broadcast token features to atom features.

    Parameters
    ----------
    x_token : torch.Tensor
        Token features of shape (..., Ntoken, D)
    token_index : torch.Tensor
        Token indices for each atom of shape (..., Natom)
    atom_mask : torch.Tensor (optional)
        Atom mask of shape (..., Natom)

    Returns
    -------
    torch.Tensor
        Atom features of shape (..., Natom, D)
    """
    # 1. Expand indices to match the feature dimension D
    # Shape: [..., Natom] -> [..., Natom, 1] -> [..., Natom, D]
    D = x_token.size(-1)
    index_expanded = token_index[..., None].expand(*token_index.shape, D)

    # 2. Gather (No reshaping needed, preserves arbitrary batch dims)
    # Gather along the token dimension (second to last dim)
    atom_features = torch.gather(x_token, dim=-2, index=index_expanded)

    # 3. Apply Masking (Type-safe and in-place efficient)
    if atom_mask is not None:
        # Zero out invalid atoms.
        atom_features.masked_fill_(~atom_mask.bool()[..., None], 0)

    return atom_features


def aggregate_atoms_to_tokens(
    x_atom: torch.Tensor,
    token_index: torch.Tensor,
    num_tokens: int,
    atom_mask: torch.Tensor | None = None,
    aggr: str = "mean",
) -> torch.Tensor:
    """Aggregate atom features to token features.

    Parameters
    ----------
    x_atom : torch.Tensor
        Atom features of shape (..., Natom, D)
    token_index : torch.Tensor
        Token indices for each atom of shape (..., Natom)
    num_tokens : int
        Number of tokens
    atom_mask : torch.Tensor (optional)
        Atom mask of shape (..., Natom)
    aggr : str
        Aggregation method: "mean" or "sum"

    Returns
    -------
    torch.Tensor
        Token features of shape (..., Ntoken, D)
    """
    # *batch_shapes, Natom, D
    *batch_shapes, num_atoms, D = x_atom.shape
    device = x_atom.device
    dtype = x_atom.dtype

    # 1. Initialize Output
    out = torch.zeros(
        *batch_shapes, num_tokens, D, device=device, dtype=dtype
    )  # (..., Ntoken, D)

    # 2. Prepare Inputs
    # Handle padding indices (-1) by clamping to 0.
    # (We mask the values later so adding to index 0 is safe)
    safe_index = token_index.clamp(min=0, max=num_tokens - 1)

    # Expand index for gather/scatter: (..., Natom) -> (..., Natom, D)
    index_expanded = safe_index.unsqueeze(-1).expand_as(x_atom)

    # 3. Masking
    # We perform masking on the input values.
    # If mask is None, we assume all atoms are valid.
    if atom_mask is not None:
        # (..., Natom, 1) broadcasting to (..., Natom, D)
        x_atom = x_atom * atom_mask.unsqueeze(-1).type(dtype)

    # 4. Scatter Sum (Numerator)
    # Sums x_atom into out at the specified indices
    out.scatter_add_(dim=-2, index=index_expanded, src=x_atom)

    # 5. Mean Handling (Denominator)
    if aggr == "mean":
        # Optimization: Count atoms per token using only 1 channel, not D.
        # Shape: (..., Ntoken, 1)
        atom_counts = torch.zeros(
            *batch_shapes, num_tokens, 1, device=device, dtype=dtype
        )

        # Create ones: (..., Natom, 1)
        ones = torch.ones(*batch_shapes, num_atoms, 1, device=device, dtype=dtype)

        if atom_mask is not None:
            ones = ones * atom_mask.unsqueeze(-1).type(dtype)

        # Scatter counts: indices must match src dim.
        # index: (..., Natom) -> (..., Natom, 1)
        index_counts = safe_index.unsqueeze(-1)

        atom_counts.scatter_add_(dim=-2, index=index_counts, src=ones)

        # Avoid division by zero
        atom_counts = atom_counts.clamp(min=1.0)

        # Broadcast division: (..., Ntoken, D) / (..., Ntoken, 1)
        out = out / atom_counts

    return out


class LocalAttentionIndex:
    def __init__(
        self,
        num_atoms: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        device: torch.device,
    ) -> None:
        """Indexer for local atom attention.

        Converts flat atom sequences into windowed query/key representations
        using efficient index gathering.
        """
        assert num_atoms % atoms_per_window_queries == 0

        self.L: int = num_atoms
        self.W: int = num_atoms // atoms_per_window_queries
        self.Lq: int = atoms_per_window_queries
        self.Lk: int = atoms_per_window_keys
        self.device = device

        # Check alignment
        half_block = self.Lq // 2
        assert self.Lk % half_block == 0

        # Pre-calculate the gather indices once
        # Shape: [W, Lk]
        gather_indices, pad_mask = self._build_gather_indices(
            self.W, self.Lq, self.Lk, device
        )
        self.gather_indices: torch.Tensor = gather_indices
        self.pad_mask: torch.Tensor = pad_mask

    @staticmethod
    @lru_cache(maxsize=2)
    def _build_gather_indices(
        W: int, Lq: int, Lk: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs the index map [W, Lk] mapping window rows to atom indices."""
        half_block_size = Lq // 2
        num_half_blocks = 2 * W
        h = Lk // (Lq // 2)  # Number of half-blocks per key window

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

        return gather_indices, pad_mask

    def to_query(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """Convert single tensor to query tensor.
        feature: [..., L, D] -> [..., W, Lq, D] (set dim=-2)
        mask, indices: [..., L] -> [..., W, Lq] (set dim=-1)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., L, D] or [..., L]
        dim : int, optional
            Dimension corresponding to the atom sequence (default: -2)
            should be -2 for features and -1 for masks or indices.

        Returns
        -------
        torch.Tensor
            Query tensor of shape [..., W, Lq, D] or [..., W, Lq]
        """
        # Efficient view (zero-copy)
        assert dim in (-2, -1), "Only supports dim -2 or -1 for unflattening."
        return x.unflatten(dim=dim, sizes=(self.W, self.Lq))

    def to_key(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """Convert single tensor to key tensor using index gathering.
        feature: [..., L, D] -> [..., W, Lk, D]
        mask, indices: [..., L] -> [..., W, Lk]

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., L, D] or [..., L]
        Returns
        -------
        torch.Tensor
            Key tensor of shape [..., W, Lk, D] or [..., W, Lk]
        """
        assert dim in (-2, -1), "Only supports dim -2 or -1 for unflattening."

        if dim == -1:
            # Add feature dim
            x = x.unsqueeze(-1)  # [..., L, 1]

        # gather_indices: [W, Lk]
        # x: [..., L, D]

        # Flatten batch dims for clean gathering: [Batch_Total, L, D]
        D = x.shape[-1]
        original_shape = x.shape
        x_flat = x.reshape(-1, self.L, D)

        # Indexing
        keys = x_flat[:, self.gather_indices]  # [Batch, W, Lk, D]

        # Apply padding mask
        keys.masked_fill_(self.pad_mask.unsqueeze(0).unsqueeze(-1), 0)

        # Reshape back
        out = keys.reshape(*original_shape[:-2], self.W, self.Lk, D)
        if dim == -1:
            out = out.squeeze(-1)  # [..., W, Lk]
        return out


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
