import math
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def sum_array(array: ArrayT, dim: int, keepdim: bool = False) -> ArrayT:
    if isinstance(array, np.ndarray):
        return array.sum(dim, keepdims=keepdim)
    else:
        return array.sum(dim, keepdim=keepdim)


def mean_array(array: ArrayT, dim: int, keepdim: bool = False) -> ArrayT:
    if isinstance(array, np.ndarray):
        return array.mean(dim, keepdims=keepdim)
    else:
        return array.mean(dim, keepdim=keepdim)


def stack_array(arrays: Sequence[ArrayT], dim: int) -> ArrayT:
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=dim)
    else:
        return torch.stack(arrays, dim=dim)  # type: ignore


def get_center(coords: ArrayT, mask: ArrayT) -> ArrayT:
    """Get the mean position of the masked coordinates.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (*, L, 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (*, L).

    Returns
    -------
    np.ndarray | torch.Tensor
        The mean position of the masked coordinates of shape (*, 1, 3).

    """
    if not mask.any():
        raise ValueError("Mask has no True values; cannot compute center.")

    if isinstance(coords, np.ndarray):
        assert isinstance(mask, np.ndarray)
        total_mass = mask.sum(-1, keepdims=True, dtype=np.float32).clip(1)  # [..., 1]
        center = (
            np.sum(coords * mask[..., None], axis=-2, keepdims=True)
            / total_mass[..., None]
        )  # [..., 1, 3]
        return center
    else:
        assert isinstance(mask, torch.Tensor)
        total_mass = mask.sum(-1, keepdim=True).clamp(1)
        center = (
            torch.sum(coords * mask[..., None], dim=-2, keepdim=True)
            / total_mass[..., None]
        )
        return center


def do_centering(coords: ArrayT, mask: ArrayT, mask_to_zero: bool = True) -> ArrayT:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        The coordinates tensor of shape (*, L, 3).
    mask : np.ndarray | torch.Tensor
        The boolean mask tensor of shape (*, L).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.
        # NOTE: this is not used in Boltz. (Boltz's masked coords is -center_pos)

    Returns
    -------
    np.ndarray | torch.Tensor
        The centered coordinates tensor of shape (*, 3).

    """
    assert coords.ndim == mask.ndim + 1
    assert coords.ndim >= 2

    if not mask.any():
        # If no positions are masked, return coords as is
        return coords

    center_pos = get_center(coords, mask)  # shape (*, 1, 3)
    centered_coords = coords - center_pos

    if mask_to_zero:
        centered_coords = centered_coords * mask[..., None]

    return centered_coords  # type: ignore


def center_random_augmentation(
    coords: ArrayT,
    mask: ArrayT,
    s_trans: float = 1.0,
    centering: bool = True,
    random_rotate: bool = True,
    mask_to_zero: bool = True,
    rng: np.random.Generator | torch.Generator | None = None,
) -> ArrayT:
    """Centering and Random Augmentation (NumPy/Torch version)
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    if isinstance(coords, np.ndarray):
        assert isinstance(mask, np.ndarray)
        assert isinstance(rng, np.random.Generator | None)
        return _center_random_augmentation_npy(
            coords,
            mask,
            s_trans,
            centering,
            random_rotate,
            mask_to_zero,
            rng,
        )
    else:
        assert isinstance(mask, torch.Tensor)
        assert isinstance(rng, torch.Generator | None)
        return _center_random_augmentation_torch(
            coords,
            mask,
            s_trans,
            centering,
            random_rotate,
            mask_to_zero,
            rng,
        )


def _center_random_augmentation_npy(
    coords: np.ndarray,
    mask: np.ndarray,
    s_trans: float = 1.0,
    centering: bool = True,
    random_rotate: bool = True,
    mask_to_zero: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Centering and Random Augmentation (NumPy version)
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    # Line 1
    if centering:
        coords = do_centering(coords, mask)

    # Line 2,4
    if random_rotate:
        assert isinstance(rng, np.random.Generator | None)
        R = random_rotations_npy(
            coords.shape[:-2], dtype=np.float32, rng=rng
        )  # [..., 3, 3]
        coords = np.einsum("...md,...ds->...ms", coords, R)

    # Line 3,4
    if s_trans > 0.0:
        # Create random translation with same shape as coords[..., 0:1, :]
        trans_shape = list(coords.shape)
        trans_shape[-2] = 1  # The 'L' dimension becomes 1 for broadcasting

        if rng is None:
            noise = np.random.randn(*trans_shape).astype(coords.dtype)
        else:
            noise = rng.normal(size=trans_shape).astype(coords.dtype)
        random_trans = noise * s_trans
        coords = coords + random_trans

    if mask_to_zero:
        coords = coords * mask[..., None]

    return coords


def _center_random_augmentation_torch(
    coords: torch.Tensor,
    mask: torch.Tensor,
    s_trans: float = 1.0,
    centering: bool = True,
    random_rotate: bool = True,
    mask_to_zero: bool = True,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Centering and Random Augmentation
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """
    # Line 1
    if centering:
        coords = do_centering(coords, mask)

    # Line 2,4
    if random_rotate:
        R = random_rotations_torch(
            coords.shape[:-2], coords.dtype, coords.device, rng=rng
        )  # [..., 3, 3]
        coords = torch.einsum("...md,...ds->...ms", coords, R)  # noqa

    # Line 3,4
    if s_trans > 0.0:
        noise = torch.randn(
            coords[..., 0:1, :].shape,
            dtype=coords.dtype,
            device=coords.device,
            generator=rng,
        )
        coords = coords + noise * s_trans

    if mask_to_zero:
        coords = coords * mask[..., None]

    return coords


def _copysign(a: ArrayT, b: ArrayT) -> ArrayT:
    """
    Return a array where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source array.
        b: array whose signs will be used, of the same shape as a.

    Returns:
        Array of the same shape as a with the signs of b.
    """
    if isinstance(a, np.ndarray):
        return np.copysign(a, b)
    else:
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)


def random_rotations_npy(
    shape: tuple[int, ...], dtype: type | np.dtype, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        shape: Shape of rotation matrices batch (e.g., (batch_size,))
    """
    # Get random quaternions
    n = math.prod(shape)
    if rng is None:
        o = np.random.randn(n, 4).astype(dtype)
    else:
        o = rng.normal(size=(n, 4)).astype(dtype)
    s = (o * o).sum(axis=1)
    # Use broadcasting for division
    quaternions = o / _copysign(np.sqrt(s), o[:, 0])[:, np.newaxis]

    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)


def random_rotations_torch(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        shape: Shape of rotation matrices batch (e.g., (batch_size,))
    """
    # Get random quaternions
    n = math.prod(shape)
    print(n)

    o = torch.randn((n, 4), dtype=dtype, device=device, generator=rng)
    s = (o * o).sum(1)
    quaternions = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]

    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)


def quaternion_to_matrix(quaternions: ArrayT) -> ArrayT:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: array of shape (..., 4) (real part first)
    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    # Unbind equivalent in numpy is slicing
    r, i, j, k = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )

    # (..., ) -> (..., )
    two_s = 2.0 / sum_array(quaternions * quaternions, -1)

    o = stack_array(
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
        dim=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
