import warnings
from typing import overload

import numpy as np
import torch


@overload
def rigid_align(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    anchor_index: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor: ...


def rigid_align(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    anchor_index: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Performs rigid alignment without weights

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the target coordinates.
    mask : np.ndarray | torch.Tensor
        Array of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | torch.Tensor | None, optional
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : torch.Tensor
        Tensor of shape (..., N, 3) containing the aligned coordinates.
    """
    if isinstance(coords, np.ndarray):
        return weighted_rigid_align_numpy(coords, target, None, mask, anchor_index)  # type: ignore
    elif isinstance(coords, torch.Tensor):
        return weighted_rigid_align_torch(coords, target, None, mask, anchor_index)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


@overload
def weighted_rigid_align(
    coords: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    mask: np.ndarray,
    anchor_index: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def weighted_rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    mask: torch.Tensor,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor: ...


def weighted_rigid_align(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    mask: np.ndarray | torch.Tensor,
    anchor_index: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Performs weighted rigid alignment of a set of coordinates to a target set using SVD.

    This function computes the optimal rigid transformation (rotation and translation)
    that aligns `coords` to `target`, minimizing the weighted mean squared error,
    with optional masking and numerical stability.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the target coordinates.
    weights : np.ndarray | torch.Tensor | None
        Array of shape (..., N) containing weights for each point.
    mask : np.ndarray | torch.Tensor
        Array of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | torch.Tensor | None, optional
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    if isinstance(coords, np.ndarray):
        return weighted_rigid_align_numpy(coords, target, weights, mask, anchor_index)  # type: ignore
    elif isinstance(coords, torch.Tensor):
        return weighted_rigid_align_torch(coords, target, weights, mask, anchor_index)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


def weighted_rigid_align_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    mask: np.ndarray,
    anchor_index: np.ndarray | None = None,
) -> np.ndarray:
    """
    Performs weighted rigid alignment of a set of coordinates to a target set using SVD.

    This function computes the optimal rigid transformation (rotation and translation)
    that aligns `coords` to `target`, minimizing the weighted mean squared error,
    with optional masking and numerical stability.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (..., N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (..., N, 3) representing the target coordinates.
    weights : np.ndarray | None (optional)
        Array of shape (..., N) containing weights for each point.
    mask : np.ndarray
        Array of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | None, optional
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : np.ndarray
        Array of shape (..., N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    if not np.any(mask):
        return coords

    if weights is None:
        weights = mask.astype(coords.dtype)
    else:
        weights = weights * mask

    # Select anchor points if provided
    if anchor_index is not None:
        anchor_coords = coords[..., anchor_index, :]
        anchor_target = target[..., anchor_index, :]
        anchor_weights = weights[..., anchor_index]
        RT, T = get_rigid_transform_numpy(anchor_coords, anchor_target, anchor_weights)
    else:
        RT, T = get_rigid_transform_numpy(coords, target, weights)

    # Apply transformation: coords @ RT + T
    # Matrix multiplication: (..., N, 3) @ (..., 3, 3) -> (..., N, 3)
    aligned_coords = np.matmul(coords, RT) + T[..., np.newaxis, :]

    return aligned_coords


def get_rigid_transform_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the rigid transformation that aligns `coords` to `target`.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (..., N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (..., N, 3) representing the target coordinates.
    weights : np.ndarray
        Array of shape (..., N) containing weights for each point.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : np.ndarray
        Array of shape (..., 3, 3) representing the rotation matrices.
    t : np.ndarray
        Array of shape (..., 3) representing the translation vectors.
    """
    dtype = coords.dtype

    # Ensure computation is at least float32 for SVD stability
    # (Matches the logic of torch.autocast to float32 in the original code)
    if not np.any(weights):
        # Return identity transform
        shape_prefix = coords.shape[:-2]
        R = np.eye(3, dtype=dtype)
        t = np.zeros((3,), dtype=dtype)
        if shape_prefix:
            R = np.tile(R, (*shape_prefix, 1, 1))
            t = np.tile(t, (*shape_prefix, 1))
        return R, t

    L = coords.shape[-2]
    if L < 4:
        warnings.warn(
            f"Point cloud has only {L} points (< 4). "
            "Weighted rigid alignment may not produce a unique rotation.",
            stacklevel=2,
        )

    w_sum = np.sum(weights, axis=-1) + eps

    # Expand weights for broadcasting: (..., N) -> (..., N, 1)
    w_expanded = weights[..., np.newaxis]

    # Weighted centroids
    coords_center = np.sum(coords * w_expanded, axis=-2) / w_sum[..., np.newaxis]
    target_center = np.sum(target * w_expanded, axis=-2) / w_sum[..., np.newaxis]

    # Center coordinates
    coords_centered = coords - coords_center[..., np.newaxis, :]
    target_centered = target - target_center[..., np.newaxis, :]

    # Covariance matrix H
    # Equivalent to torch.einsum("...ni, ...nj -> ...ij")
    H = np.einsum("...ni, ...nj -> ...ij", coords_centered * w_expanded, target_centered)

    try:
        # SVD: H = U S Vh
        # numpy.linalg.svd returns u, s, vh (where vh is V^H)
        U, _, Vh = np.linalg.svd(H)

        # Fixed reflection removal (Kabsch algorithm)
        # Check determinant of U @ Vh
        d = np.linalg.det(np.matmul(U, Vh))

        # Create correction matrix F
        F = np.eye(3, dtype=dtype)
        if H.ndim > 2:
            # Tile F for batch dimensions: (..., 3, 3)
            batch_shape = H.shape[:-2]
            F = np.tile(F, (*batch_shape, 1, 1))

        # Apply sign to the last element of the diagonal
        if F.ndim == 2:
            F[2, 2] = np.sign(d)
        else:
            F[..., 2, 2] = np.sign(d)

        # Transposed rotation matrix RT = U @ F @ Vh
        # This matches the PyTorch implementation logic
        RT = np.matmul(U, np.matmul(F, Vh))

    except np.linalg.LinAlgError as e:
        warnings.warn(
            f"SVD failed during weighted rigid alignment: {e}. "
            "Returning identity rotation.",
            stacklevel=2,
        )
        shape_prefix = coords.shape[:-2]
        RT = np.eye(3, dtype=dtype)
        if shape_prefix:
            RT = np.tile(RT, (*shape_prefix, 1, 1))

    # Compute translation: t = target_center - coords_center @ RT
    # Needs explicit dimension handling for matmul
    term2 = np.matmul(coords_center[..., np.newaxis, :], RT)
    t = target_center[..., np.newaxis, :] - term2
    t = t.squeeze(-2)  # Remove the singleton dimension

    return RT, t


@torch.no_grad()
def weighted_rigid_align_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    mask: torch.Tensor,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs weighted rigid alignment of a set of coordinates to a target set using SVD.

    This function computes the optimal rigid transformation (rotation and translation)
    that aligns `coords` to `target`, minimizing the weighted mean squared error,
    with optional masking and numerical stability.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (..., N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Tensor of shape (..., N, 3) representing the target coordinates.
    weights : torch.Tensor | None, optional
        Tensor of shape (..., N) containing weights for each point.
    mask : torch.Tensor
        Tensor of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : torch.Tensor | None, optional
        Tensor of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : torch.Tensor
        Tensor of shape (..., N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    original_dtype = coords.dtype

    if not mask.any():
        # If there are no valid atoms, return identical coords
        return coords

    if weights is None:
        weights = mask.to(dtype=coords.dtype)
    else:
        weights = weights * mask

    with torch.autocast(device_type=coords.device.type, dtype=torch.float32):
        if anchor_index is not None:
            anchor_coords = coords[..., anchor_index, :]
            anchor_target = target[..., anchor_index, :]
            anchor_weights = weights[..., anchor_index]
            RT, T = get_rigid_transform_torch(
                anchor_coords, anchor_target, anchor_weights
            )
        else:
            RT, T = get_rigid_transform_torch(coords, target, weights)
        aligned_coords = coords @ RT + T[..., None, :]

    return aligned_coords.to(original_dtype)


def get_rigid_transform_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the rigid transformation that aligns `coords` to `target`.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (..., N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Tensor of shape (..., N, 3) representing the target coordinates.
    weights : torch.Tensor
        Tensor of shape (..., N) containing weights for each point.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : torch.Tensor
        Tensor of shape (..., 3, 3) representing the rotation matrices.
    t : torch.Tensor
        Tensor of shape (..., 3) representing the translation vectors.
    """
    device = coords.device
    original_dtype = coords.dtype

    if not weights.any():
        # If there are no valid atoms, return identity transform
        R = torch.eye(3, dtype=original_dtype, device=device)
        R = R.tile(*coords.shape[:-2], 1, 1)
        t = torch.zeros(*coords.shape[:-2], 3, dtype=original_dtype, device=device)
        return R, t

    L = coords.shape[-2]
    if L < 4:
        warnings.warn(
            f"Point cloud has only {L} points (< 4). "
            "Weighted rigid alignment may not produce a unique rotation.",
            stacklevel=2,
        )

    w_sum = weights.sum(dim=-1) + eps
    coords_center = (coords * weights[..., None]).sum(dim=-2) / w_sum[..., None]
    target_center = (target * weights[..., None]).sum(dim=-2) / w_sum[..., None]

    coords = coords - coords_center[..., None, :]
    target = target - target_center[..., None, :]

    H = torch.einsum(
        "...ni, ...nj -> ...ij",
        coords * weights[..., None],
        target,
    )

    try:
        U, _, V = torch.linalg.svd(H)

        # Fixed reflection removal
        F = torch.eye(3, dtype=torch.float32, device=device)
        F = F.tile(*H.shape[:-2], 1, 1)
        F[..., -1, -1] = torch.sign(torch.linalg.det(U @ V))

        # Transposed rotation matrix
        RT = torch.einsum("...ij, ...jk, ...kl -> ...il", U, F, V)
    except RuntimeError as e:
        warnings.warn(
            f"SVD failed during weighted rigid alignment: {e}. "
            "Returning identity rotation.",
            stacklevel=2,
        )
        RT = torch.eye(3, dtype=torch.float32, device=device)
        RT = RT.tile(*coords.shape[:-2], 1, 1)

    # Compute translation
    t = target_center[..., None, :] - coords_center[..., None, :] @ RT
    t = t.squeeze(-2)

    return RT, t
