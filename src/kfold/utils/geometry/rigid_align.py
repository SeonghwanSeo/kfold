import warnings
from typing import overload

import numpy as np
import torch

__all__ = ["compute_rmsd", "rigid_align", "weighted_rigid_align"]


@overload
def compute_rmsd(
    coords: np.ndarray, target: np.ndarray, mask: np.ndarray, align: bool = False
) -> np.ndarray: ...


@overload
def compute_rmsd(
    coords: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, align: bool = False
) -> torch.Tensor: ...


def compute_rmsd(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    align: bool = False,
) -> np.ndarray | torch.Tensor:
    """
    computes the root mean square deviation (rmsd) between two sets of coordinates.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (..., N, 3) representing the target coordinates.
    mask : np.ndarray | torch.Tensor
        Array of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    align : bool, optional
        If True, perform rigid alignment before computing RMSD (default: False).

    Returns
    -------
    rmsd : np.ndarray | torch.Tensor
        RMSD values.
    """
    if align:
        coords = rigid_align(coords, target, mask)

    if isinstance(coords, np.ndarray):
        assert isinstance(target, np.ndarray) and isinstance(mask, np.ndarray)

        # Expand mask for broadcasting: (..., N) -> (..., N, 1)
        mask = mask.astype(bool, copy=False)
        mask_expanded = mask[..., np.newaxis]

        # Sanitize inputs: replace values with 0 where mask is 0.
        # This prevents NaN propagation because (NaN - x) * 0 = NaN.
        # We use np.where to ensure masked positions are strictly 0.0.
        safe_coords = np.where(mask_expanded, coords, 0.0)
        safe_target = np.where(mask_expanded, target, 0.0)

        diff = safe_coords - safe_target
        # Weighted sum of squares (masked positions contribute 0)
        mse = np.sum(diff**2, axis=(-2, -1)) / np.clip(
            np.sum(mask, axis=-1), a_min=1, a_max=None
        )
        rmsd = np.sqrt(mse)
        return rmsd

    elif isinstance(coords, torch.Tensor):
        assert isinstance(target, torch.Tensor) and isinstance(mask, torch.Tensor)

        # Expand mask for broadcasting
        mask_expanded = mask[..., None]
        mask_bool = mask_expanded.bool()

        # Sanitize inputs: replace values with 0 where mask is False (0).
        # We use masked_fill for efficiency and safety against NaNs in masked regions.
        safe_coords = coords.masked_fill(~mask_bool, 0.0)
        safe_target = target.masked_fill(~mask_bool, 0.0)

        diff = safe_coords - safe_target
        mse = torch.sum(diff**2, dim=(-2, -1)) / (torch.sum(mask, dim=-1).clamp(min=1))
        rmsd = torch.sqrt(mse)
        return rmsd
    else:
        raise TypeError(
            f"Unsupported array type: {type(coords)}. "
            "Expected np.ndarray or torch.Tensor."
        )


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
    aligned_coords : np.ndarray | torch.Tensor
        Array or tensor of shape (..., N, 3) containing the aligned coordinates.
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

    # Expand mask to match coordinate dimensions for sanitization
    mask_expanded = mask[..., np.newaxis].astype(bool)

    # Sanitize inputs: If there are NaNs in masked regions, they will propagate
    # during centroid calculation even if weights are zero (NaN * 0 = NaN).
    # We force masked regions to 0.0.
    coords = np.where(mask_expanded, coords, 0.0)
    target = np.where(mask_expanded, target, 0.0)

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
    # Note: Inputs (coords, target) are already sanitized (0 in masked regions),
    # so summation here is safe even if original data had NaNs in masked regions.
    coords_center = np.sum(coords * w_expanded, axis=-2) / w_sum[..., np.newaxis]
    target_center = np.sum(target * w_expanded, axis=-2) / w_sum[..., np.newaxis]

    # Center coordinates
    coords_centered = coords - coords_center[..., np.newaxis, :]
    target_centered = target - target_center[..., np.newaxis, :]

    # Re-apply mask implicitly by multiplying weights (masked regions become 0 again)
    # This is redundant if sanitized, but ensures correctness if weights vary.
    H = np.einsum("...ni, ...nj -> ...ij", coords_centered * w_expanded, target_centered)

    try:
        # SVD: H = U S Vh
        U, _, Vh = np.linalg.svd(H)

        # Fixed reflection removal (Kabsch algorithm)
        d = np.linalg.det(np.matmul(U, Vh))

        F = np.eye(3, dtype=dtype)
        if H.ndim > 2:
            batch_shape = H.shape[:-2]
            F = np.tile(F, (*batch_shape, 1, 1))

        if F.ndim == 2:
            F[2, 2] = np.sign(d)
        else:
            F[..., 2, 2] = np.sign(d)

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
    term2 = np.matmul(coords_center[..., np.newaxis, :], RT)
    t = target_center[..., np.newaxis, :] - term2
    t = t.squeeze(-2)

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
    Torch implementation of weighted rigid alignment.
    """
    original_dtype = coords.dtype

    if not mask.any():
        return coords

    # Create boolean mask for filling
    mask_bool = mask.bool().unsqueeze(-1)

    # Sanitize inputs: masked_fill handles NaNs correctly by replacing them with 0.0
    # where the mask is False (masked out). This is crucial before any math.
    coords = coords.masked_fill(~mask_bool, 0.0)
    target = target.masked_fill(~mask_bool, 0.0)

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

    # Inputs are already sanitized (0.0 in masked regions), so these sums are safe.
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
