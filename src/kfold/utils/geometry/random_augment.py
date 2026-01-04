import math
from collections.abc import Sequence
from typing import TypeVar, overload

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
        # Expand mask: (..., L) -> (..., L, 1)
        mask_expanded = mask[..., None]

        # Sanitize coords: replace masked positions with 0.0 to prevent NaN propagation.
        # (NaN * 0.0 = NaN, so we must remove NaNs before math).
        safe_coords = np.where(mask_expanded.astype(bool, copy=False), coords, 0.0)

        total_mass = mask.sum(-1, keepdims=True, dtype=np.float32).clip(1)  # [..., 1]
        center = (
            np.sum(safe_coords, axis=-2, keepdims=True) / total_mass[..., None]
        )  # [..., 1, 3]
        return center
    else:
        assert isinstance(mask, torch.Tensor)
        mask_bool = mask.bool().unsqueeze(-1)

        # Sanitize coords
        safe_coords = coords.masked_fill(~mask_bool, 0.0)

        total_mass = mask.sum(-1, keepdim=True).clamp(1)
        center = torch.sum(safe_coords, dim=-2, keepdim=True) / total_mass[..., None]
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

    # get_center now handles NaNs internally
    center_pos = get_center(coords, mask)  # shape (*, 1, 3)
    centered_coords = coords - center_pos

    if mask_to_zero:
        if isinstance(centered_coords, np.ndarray):
            # Use where to cleanly zero out masked regions (handling potential NaNs)
            mask_expanded = mask[..., None].astype(bool, copy=False)
            centered_coords = np.where(mask_expanded, centered_coords, 0.0)
        elif isinstance(centered_coords, torch.Tensor):
            # Use masked_fill to cleanly zero out masked regions
            mask_bool = mask.bool().unsqueeze(-1)
            centered_coords = centered_coords.masked_fill(~mask_bool, 0.0)

    return centered_coords  # type: ignore


def center_random_augmentation(
    coords: ArrayT,
    mask: ArrayT,
    augmentation: bool = True,
    centering: bool = True,
    mask_to_zero: bool = True,
    s_trans: float = 1.0,
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
            centering,
            augmentation,
            s_trans,
            mask_to_zero,
            rng,
        )
    else:
        assert isinstance(mask, torch.Tensor)
        assert isinstance(rng, torch.Generator | None)
        return _center_random_augmentation_torch(
            coords,
            mask,
            centering,
            augmentation,
            s_trans,
            mask_to_zero,
            rng,
        )


def _center_random_augmentation_npy(
    coords: np.ndarray,
    mask: np.ndarray,
    centering: bool = True,
    augmentation: bool = True,
    s_trans: float = 1.0,
    mask_to_zero: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Centering and Random Augmentation (NumPy version)
    See Section 3.7 Algorithm 19 of the AlphaFold3 paper.
    """
    # Line 1
    if centering:
        coords = do_centering(coords, mask, mask_to_zero=False)

    if augmentation:
        rng = rng or np.random.default_rng()

        # Line 2,4
        R = random_rotations_npy(
            coords.shape[:-2], dtype=np.float32, rng=rng
        )  # [..., 3, 3]

        # Matrix multiplication is safe (NaNs stay local to invalid atoms)
        coords = np.einsum("...md,...ds->...ms", coords, R)

        # Line 3,4
        if s_trans > 0.0:
            # Create random translation with same shape as coords[..., 0:1, :]
            trans_shape = list(coords.shape)
            trans_shape[-2] = 1  # The 'L' dimension becomes 1 for broadcasting

            noise = rng.normal(size=trans_shape).astype(coords.dtype)
            random_trans = noise * s_trans
            coords = coords + random_trans

    if mask_to_zero:
        # Use np.where to ensure NaN values in masked regions become 0.0
        mask_expanded = mask[..., None].astype(bool)
        coords = np.where(mask_expanded, coords, 0.0)

    return coords


def _center_random_augmentation_torch(
    coords: torch.Tensor,
    mask: torch.Tensor,
    centering: bool = True,
    augmentation: bool = True,
    s_trans: float = 1.0,
    mask_to_zero: bool = True,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Centering and Random Augmentation
    See Section 3.7 Algorithm 19 CentreRandomAugmentation
    """
    # Line 1
    if centering:
        coords = do_centering(coords, mask, mask_to_zero=False)

    # Line 2,4
    if augmentation:
        R = random_rotations_torch(
            coords.shape[:-2], coords.dtype, coords.device, rng=rng
        )  # [..., 3, 3]
        coords = torch.einsum("...md,...ds->...ms", coords, R)

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
        # Use masked_fill to ensure NaN values in masked regions become 0.0
        mask_bool = mask.bool().unsqueeze(-1)
        coords = coords.masked_fill(~mask_bool, 0.0)

    return coords


def _copysign(a: ArrayT, b: ArrayT) -> ArrayT:
    """
    Return an array where each element has the absolute value taken from the,
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
        return torch.copysign(a, b)


def random_rotations_npy(
    shape: tuple[int, ...], dtype: type | np.dtype, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        shape: Shape of rotation matrices batch (e.g., (batch_size,))
    """
    # Get random quaternions
    n = math.prod(shape)
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
            R = random_rotations_torch(
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
            # Use masked_fill to handle NaNs correctly
            mask_bool = mask.bool().unsqueeze(-1)
            coords_list = [x.masked_fill(~mask_bool, 0.0) for x in coords_list]

        if len(coords) == 1:
            # Single tensor input, return tensor
            return coords_list[0]
        else:
            # Multiple tensor input, return list of tensors
            return tuple(coords_list)
