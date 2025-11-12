# started from code from https://github.com/jwohlwend/boltz, MIT License,

import math
from functools import lru_cache
from typing import TypeVar, overload

import torch
import torch.nn.functional as F
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


@lru_cache(maxsize=2)
def get_indexing_matrix(W: int, Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
    """Get indexing matrix for local attention.
    Cache the result for efficiency.
    """
    assert Lq % 2 == 0
    assert Lk % (Lq // 2) == 0

    h = Lk // (Lq // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * W, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(min=0, max=h + 1)
    index = index.view(W, 2, 2 * W)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * W, h * W).float()


class LocalAttentionIndexer:
    def __init__(
        self,
        num_atoms: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        device: torch.device,
    ) -> None:
        """Indexer for local atom attention.

        Converts flat atom sequences into windowed query/key representations
        for efficient local attention computation.

        Parameters
        ----------
        num_atoms : int
            Number of atoms (L).
        atoms_per_window_queries : int
            Atoms per window for queries (Lq).
        atoms_per_window_keys : int
            Atoms per window for keys (Lk).
        """
        assert num_atoms % atoms_per_window_queries == 0
        self.L: int = num_atoms
        self.W: int = num_atoms // atoms_per_window_queries
        self.Lq: int = atoms_per_window_queries
        self.Lk: int = atoms_per_window_keys
        self.indexing_matrix = get_indexing_matrix(self.W, self.Lq, self.Lk, device)

    def to_query(self, x: torch.Tensor) -> torch.Tensor:
        """Convert single tensor to query tensor using indexing matrix.
        [..., L, D] -> [..., K, Lq, D]
        """
        return x.unflatten(-2, (self.W, self.Lq))

    def to_key(self, x: torch.Tensor) -> torch.Tensor:
        """Convert single tensor to key tensor using indexing matrix.
        [..., L, D] -> [..., K, Lw, D]
        """
        # if dtype is not float, convert to float for einsum
        original_dtype = x.dtype
        if original_dtype == torch.long:
            x = x.float()

        original_shape = x.shape  # [..., L, D]
        L, D = original_shape[-2:]
        assert L == self.L
        W, Lq, Lk = self.W, self.Lq, self.Lk

        x = x.unflatten(-2, (2 * W, Lq // 2))  # [..., 2W, Lq/2, D]
        key = torch.einsum("... j i d, j k -> ... k i d", x, self.indexing_matrix)
        key = key.reshape(*original_shape[:-2], W, Lk, D)  # [..., W, Lk, D]
        return key.to(original_dtype)


def center_random_augmentation(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    centering: bool = True,
    random_rotate: bool = True,
) -> torch.Tensor:
    """Centering and Random Augmentation
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """

    coords_shape = coords.shape

    # Line 1
    if centering:
        center = torch.sum(
            coords * atom_mask[..., None], dim=-2, keepdim=True
        ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(1)
        coords = coords - center

    # Line 2,4
    if random_rotate:
        R = random_rotations(
            coords_shape[:-2], coords.dtype, coords.device
        )  # [..., 3, 3]
        rotate = lambda x: torch.einsum("...md,...ds->...ms", x, R)  # noqa
        coords = rotate(coords)

    # Line 3,4
    if s_trans > 0.0:
        random_trans = torch.randn_like(coords[..., 0:1, :]) * s_trans
        coords = coords + random_trans

    return coords


class CenterRandomAugmentation:
    """Centering and Random Augmentation Module
    See Section 3.7 Algorithm 19 CentreRandomAugmentation

    Usage)
    ```python
    augment = CenterRandomAugmentation(...)
    x = augment(x, atom_mask=mask)
    x, y = augment(x, y, atom_mask=mask)
    ```
    """

    def __init__(
        self, s_trans: float = 1.0, centering: bool = True, random_rotate: bool = True
    ):
        self.s_trans: float = s_trans
        self.centering: bool = centering
        self.random_rotate: bool = random_rotate

    @overload
    def __call__(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def __call__(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.augment(*coords, atom_mask=atom_mask)

    @overload
    def augment(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def augment(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        *others: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]: ...

    def augment(  # type: ignore[override]
        self,
        *coords: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """See Section 3.7 Algorithm 19 CentreRandomAugmentation

        Parameters
        ----------
        coords : torch.Tensor
            One or more tensors of shape (..., L, 3) representing atomic coordinates.
        atom_mask : torch.Tensor
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
            center = torch.sum(
                ref_coords * atom_mask[..., None], dim=-2, keepdim=True
            ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(1)
            coords_list = [x - center for x in coords_list]

        # Line 2,4
        if self.random_rotate:
            R = random_rotations(
                coords_shape[:-2], ref_coords.dtype, ref_coords.device
            )  # [..., 3, 3]
            rotate = lambda x: torch.einsum("...md,...ds->...ms", x, R)  # noqa
            coords_list = [rotate(x) for x in coords_list]

        # Line 3,4
        if self.s_trans > 0.0:
            random_trans = torch.randn_like(ref_coords[..., 0:1, :]) * self.s_trans
            coords_list = [x + random_trans for x in coords_list]

        if len(coords) == 1:
            # Single tensor input, return tensor
            return coords_list[0]
        else:
            # Multiple tensor input, return list of tensors
            return tuple(coords_list)


def center(atom_coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


def compute_random_augmentation(
    num_diffusion_samples: int,
    s_trans: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
):
    R = random_rotations(num_diffusion_samples, dtype=dtype, device=device)
    random_trans = (
        torch.randn((num_diffusion_samples, 1, 3), dtype=dtype, device=device) * s_trans
    )
    return R, random_trans


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
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


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
