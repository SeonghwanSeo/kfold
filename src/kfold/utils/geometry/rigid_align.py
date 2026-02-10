import warnings
from typing import overload

import numpy as np
import torch

__all__ = ["compute_rmsd", "rigid_align", "weighted_rigid_align"]


@overload
def compute_rmsd(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    align: bool = False,
    no_svd: bool = False,
) -> np.ndarray: ...


@overload
def compute_rmsd(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    align: bool = False,
    no_svd: bool = False,
) -> torch.Tensor: ...


def compute_rmsd(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    align: bool = False,
    no_svd: bool = False,
) -> np.ndarray | torch.Tensor:
    """
    computes the root mean square deviation (rmsd) between two sets of coordinates.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the target coordinates.
    mask : np.ndarray | torch.Tensor (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).
    align : bool, optional
        If True, perform rigid alignment before computing RMSD (default: False).
    no_svd : bool, optional
        If True, use non-SVD method for RMSD computation (default: False).

    Returns
    -------
    rmsd : np.ndarray | torch.Tensor
        RMSD values.
    """
    if no_svd:
        # Non-SVD based RMSD computation (float64 for numerical stability)
        if isinstance(coords, np.ndarray):
            assert isinstance(target, np.ndarray) and isinstance(mask, np.ndarray | None)
            return compute_rmsd_numpy(coords, target, mask, align)
        elif isinstance(coords, torch.Tensor):
            assert isinstance(target, torch.Tensor) and isinstance(
                mask, torch.Tensor | None
            )
            return compute_rmsd_torch(coords, target, mask, align)
        else:
            raise TypeError(
                f"Unsupported array type: {type(coords)}. "
                "Expected np.ndarray or torch.Tensor."
            )
    else:
        # Use SVD-based method via rigid alignment
        if align:
            coords = rigid_align(coords, target, mask)
        diff = coords - target
        if mask is None:
            n_points = coords.shape[-2]
        else:
            mask_expanded = mask[..., None]
            n_points = mask.sum(-1, dtype=target.dtype).clip(1)
            diff = (
                torch.where(mask_expanded, diff, 0.0)
                if isinstance(diff, torch.Tensor)
                else np.where(mask_expanded, diff, 0.0)
            )

        rmsd_sq = (diff**2).sum((-2, -1)) / n_points
        rmsd = (rmsd_sq.clip(0.0)) ** 0.5
        return rmsd


def compute_rmsd_numpy(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    align: bool = False,
) -> np.ndarray:
    """Compute minimal RMSD between two sets of coordinates without SVD.
    NOTE(SeonghwanSeo): This function replaces the SVD-based RMSD computation
    """
    original_dtype = coords.dtype
    # 1. Masking & Centering
    if mask is None:
        return compute_rmsd_small_fast(coords, target, align)

    mask = mask.astype(bool, copy=False)  # [*, N]
    mask_expanded = mask[..., np.newaxis]  # [*, N, 1]
    mask_weights = mask_expanded.astype(np.float64)  # [*, N, 1]
    n_points = mask.sum(axis=-1).clip(1)  # [*,]
    # Sanitize inputs: If there are NaNs in masked regions, they will propagate
    p = np.where(mask_expanded, coords, 0.0).astype(np.float64)  # [*, N, 3]
    q = np.where(mask_expanded, target, 0.0).astype(np.float64)  # [*, N, 3]
    del coords, target

    if not align:
        # Direct RMSD computation without alignment
        return np.sqrt(
            (((q - p) ** 2).sum(axis=(-2, -1)) / n_points).clip(0.0), dtype=original_dtype
        )

    # Center coordinates
    p_center = p.sum(-2, keepdims=True) / n_points[..., None, None]
    q_center = q.sum(-2, keepdims=True) / n_points[..., None, None]
    p_centered = (p - p_center) * mask_weights
    q_centered = (q - q_center) * mask_weights

    # 2. Compute E0 (Sum of squared norms)
    e0 = (p_centered**2).sum(axis=(-1, -2)) + (q_centered**2).sum(axis=(-1, -2))  # [*,]

    # 3. Compute Covariance Matrix H (P^T @ Q)
    h = np.matmul(p_centered.swapaxes(-1, -2), q_centered)

    # 4. Compute eigenvalues of H^T @ H
    s_sq_matrix = np.matmul(h.swapaxes(-1, -2), h)
    eigenvalues = np.linalg.eigvalsh(s_sq_matrix)
    singular_values = np.sqrt(eigenvalues.clip(0.0))  # [*, 3]

    # 5. Handle Reflection (Chirality check)
    det_h = np.linalg.det(h)
    sign = np.where(det_h < 0, -1.0, 1.0)
    singular_values[..., 0] *= sign

    # 6. Compute RMSD
    # Trace of Sigma (sum of singular values)
    trace_max = np.sum(singular_values, axis=-1)
    rmsd_sq = (e0 - 2 * trace_max) / n_points
    rmsd = np.sqrt(rmsd_sq.clip(0.0), dtype=original_dtype)
    return rmsd


def compute_rmsd_small_fast(
    coords: np.ndarray,
    target: np.ndarray,
    align: bool = True,
) -> np.ndarray:
    """
    Highly optimized RMSD for small matrices/batches.
    Assumes:
      1. No Mask (Inputs are dense or pre-masked).
      2. coords/target shape: [..., N, 3]
    """
    # 1. Direct Centering (Faster than expansion for small N due to cache locality)
    # keepdims=True avoids reshaping overhead
    original_dtype = coords.dtype
    p = coords.astype(np.float64, copy=False)
    q = target.astype(np.float64, copy=False)
    del coords, target

    if not align:
        # Direct RMSD without alignment
        diff = p - q
        rmsd_sq = (diff**2).sum(axis=(-2, -1)) / p.shape[-2]
        return np.sqrt(np.maximum(rmsd_sq, 0.0), dtype=original_dtype)

    p_centered = p - p.mean(axis=-2, keepdims=True)
    q_centered = q - q.mean(axis=-2, keepdims=True)

    # 2. Compute Covariance Matrix H (3x3)
    # Matmul is heavily optimized for batched small matrices
    # [..., 3, N] @ [..., N, 3] -> [..., 3, 3]
    h = np.matmul(p_centered.swapaxes(-1, -2), q_centered)

    # 3. Compute Eigenvalues of H^T @ H
    # This is the unavoidable bottleneck in NumPy.
    # For small batches, CPU overhead of calling LAPACK dominates.
    s_sq_matrix = np.matmul(h.swapaxes(-1, -2), h)

    # 3x3 Hermitian eigenvalues (Use eigvalsh for speed/stability)
    eigenvalues = np.linalg.eigvalsh(s_sq_matrix)

    # 4. Compute Singular Values (Sqrt of eigenvalues)
    # Clip to avoid negative zero or float error
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

    # 5. Reflection Check (Determinant)
    # [..., 3] vs [..., 3, 3]
    # Small optimization: We only need the sign of Det(H), not the value.
    # But np.linalg.det is the fastest way in NumPy.
    det_h = np.linalg.det(h)

    # Sign correction for the smallest singular value (index 0)
    # In-place update is faster
    mask_neg = det_h < 0
    if mask_neg.any():
        singular_values[mask_neg, 0] *= -1.0

    # 6. RMSD Calculation
    # E0 = Sum(P_c^2) + Sum(Q_c^2)
    # For small N, calculating norms directly is faster than statistical expansion
    e0 = (p_centered**2).sum(axis=(-2, -1)) + (q_centered**2).sum(axis=(-2, -1))

    # Trace Sum
    trace_sigma = singular_values.sum(axis=-1)

    # N for division
    rmsd_sq = (e0 - 2.0 * trace_sigma) / p.shape[-2]

    # Final Sqrt
    return np.sqrt(np.maximum(rmsd_sq, 0.0), dtype=original_dtype)


def compute_rmsd_torch(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    align: bool = False,
) -> torch.Tensor:
    """Compute minimal RMSD between two sets of coordinates using PyTorch."""
    original_dtype = coords.dtype

    # 1. Masking & Centering
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)

    mask = mask.bool()
    mask_expanded = mask.unsqueeze(-1)
    mask_weights = mask_expanded.double()

    # Sanitize inputs: If there are NaNs in masked regions, they will propagate
    p = torch.where(mask_expanded, coords, 0.0).double()
    q = torch.where(mask_expanded, target, 0.0).double()
    del coords, target

    # mask_weights: [*, N, 1]
    n_points = mask.sum(dim=-1).clamp(min=1.0)

    if not align:
        # Direct RMSD
        rmsd = ((q - p).pow(2).sum((-2, -1)) / n_points).clamp(0.0).sqrt()
        return rmsd.to(original_dtype)

    # Center coordinates
    p_center = p.sum(-2, keepdim=True) / n_points[..., None, None]
    q_center = q.sum(-2, keepdim=True) / n_points[..., None, None]
    p_centered = (p - p_center) * mask_weights
    q_centered = (q - q_center) * mask_weights

    # 2. Compute E0 (Sum of squared norms)
    e0 = q_centered.pow(2).sum(dim=(-1, -2)) + p_centered.pow(2).sum(dim=(-1, -2))

    # 3. Compute Covariance Matrix H (P^T @ Q)
    h = torch.einsum("...ni,...nj->...ij", p_centered, q_centered)  # [*, 3, 3]

    # 4. Compute eigenvalues of H^T @ H
    h_th = torch.matmul(h.transpose(-1, -2), h)
    eigenvalues = torch.linalg.eigvalsh(h_th)
    singular_values = torch.sqrt(eigenvalues.clamp(min=0.0))

    # 5. Handle Reflection (Chirality check)
    det_h = torch.linalg.det(h)
    sign = torch.where(det_h < 0, -1.0, 1.0)
    s_others = singular_values[..., 1:].sum(dim=-1)
    s_min = singular_values[..., 0] * sign

    # 6. Compute RMSD
    trace_max = s_others + s_min
    rmsd_sq = (e0 - 2.0 * trace_max) / n_points
    rmsd = rmsd_sq.clamp(min=0.0).sqrt()
    return rmsd.to(original_dtype)


@overload
def rigid_align(
    coords: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    anchor_index: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor: ...


def rigid_align(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    anchor_index: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Performs rigid alignment without weights

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the target coordinates.
    mask : np.ndarray | torch.Tensor
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | torch.Tensor | None, optional
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : np.ndarray | torch.Tensor
        Array or tensor of shape (*, N, 3) containing the aligned coordinates.
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
    mask: np.ndarray | None,
    anchor_index: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def weighted_rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    mask: torch.Tensor | None,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor: ...


def weighted_rigid_align(
    coords: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    mask: np.ndarray | torch.Tensor | None,
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
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) representing the target coordinates.
    weights : np.ndarray | torch.Tensor (optional)
        Array of shape (*, N) containing weights for each point.
    mask : np.ndarray | torch.Tensor (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | torch.Tensor (optional)
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : np.ndarray | torch.Tensor
        Array of shape (*, N, 3) containing the aligned coordinates.

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
    mask: np.ndarray | None,
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
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (*, N, 3) representing the target coordinates.
    weights : np.ndarray | None (optional)
        Array of shape (*, N) containing weights for each point.
    mask : np.ndarray | None (optional)
        Array of shape (*, N) indicating valid points (1 for valid, 0 for invalid).
    anchor_index : np.ndarray | None, optional
        Array of shape (N,) containing indices of anchors to be used for alignment.

    Returns
    -------
    aligned_coords : np.ndarray
        Array of shape (*, N, 3) containing the aligned coordinates.

    Notes
    -----
    - If the number of points N < 4, a warning is issued since the rotation may not be
      unique.
    - If SVD fails, the identity rotation is used and a warning is issued.
    """
    if mask is None:
        mask = np.ones(coords.shape[:-1], dtype=bool)

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

    if anchor_index is not None:
        # Select anchor points if provided
        _coords = coords[..., anchor_index, :]
        _target = target[..., anchor_index, :]
        weights = weights[..., anchor_index]
        RT, T = get_rigid_transform_numpy(_coords, _target, weights)
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
        Array of shape (*, N, 3) representing the coordinates to be aligned.
    target : np.ndarray
        Array of shape (*, N, 3) representing the target coordinates.
    weights : np.ndarray
        Array of shape (*, N) containing weights for each point.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : np.ndarray
        Array of shape (*, 3, 3) representing the rotation matrices.
    t : np.ndarray
        Array of shape (*, 3) representing the translation vectors.
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
    mask: torch.Tensor | None,
    anchor_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Torch implementation of weighted rigid alignment.
    """
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)

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

    # Compute rigid transformation under autocast for numerical stability
    with torch.autocast(device_type=coords.device.type, dtype=torch.float32):
        if anchor_index is not None:
            # Select anchor points if provided
            _coords = coords[..., anchor_index, :]
            _target = target[..., anchor_index, :]
            weights = weights[..., anchor_index]
            RT, T = get_rigid_transform_torch(_coords, _target, weights)
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
        Tensor of shape (*, N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Tensor of shape (*, N, 3) representing the target coordinates.
    weights : torch.Tensor
        Tensor of shape (*, N) containing weights for each point.
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    R : torch.Tensor
        Tensor of shape (*, 3, 3) representing the rotation matrices.
    t : torch.Tensor
        Tensor of shape (*, 3) representing the translation vectors.
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
