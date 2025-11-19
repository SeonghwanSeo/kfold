import math

import numpy as np


def do_centering(
    coords: np.ndarray, mask: np.ndarray, mask_to_zero: bool = True
) -> np.ndarray:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates tensor of shape (N, ..., 3).
    mask : np.ndarray
        The boolean mask tensor of shape (N,).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.
        # NOTE: this is not used in Boltz. (Boltz's masked coords is -center_pos)

    Returns
    -------
    np.ndarray
        The centered coordinates tensor of shape (N, ..., 3).

    """
    if not mask.any():
        # If no positions are masked, return coords as is
        return coords

    masked_coords = coords[mask]
    center_pos = masked_coords.mean(axis=0, keepdims=True)
    centered_coords = coords - center_pos
    if mask_to_zero:
        centered_coords[~mask] = 0.0
    return centered_coords  # type: ignore


def center_random_augmentation(
    coords: np.ndarray,
    mask: np.ndarray,
    s_trans: float = 1.0,
    centering: bool = True,
    random_rotate: bool = True,
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
        # coords.shape[:-2] gives the batch dimensions
        R = random_rotations(
            coords.shape[:-2], dtype=coords.dtype, rng=rng
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

    return coords


def _copysign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return a tensor where each element has the absolute value taken from a,
    with sign taken from b.
    """
    # NumPy has a native copysign function
    return np.copysign(a, b)


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
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
    two_s = 2.0 / (quaternions * quaternions).sum(axis=-1)

    o = np.stack(
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
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: np.dtype, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate random quaternions representing rotations."""
    if rng is None:
        o = np.random.randn(n, 4).astype(dtype)
    else:
        o = rng.normal(size=(n, 4)).astype(dtype)
    s = (o * o).sum(axis=1)
    # Use broadcasting for division
    o = o / _copysign(np.sqrt(s), o[:, 0])[:, np.newaxis]
    return o


def random_rotations(
    shape: tuple[int, ...], dtype: np.dtype, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate random rotations as 3x3 rotation matrices.
    Args:
        shape: Shape of rotation matrices batch (e.g., (batch_size,))
    """
    n = math.prod(shape)
    quaternions = random_quaternions(n, dtype=dtype, rng=rng)
    return quaternion_to_matrix(quaternions).reshape(*shape, 3, 3)
