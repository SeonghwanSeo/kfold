# started from code from https://github.com/jwohlwend/boltz, MIT License

import warnings

import torch


def rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Performs rigid alignment without weights

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (..., N, 3) representing the coordinates to be aligned.
    target : torch.Tensor
        Tensor of shape (..., N, 3) representing the target coordinates.
    mask : torch.Tensor
        Tensor of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

    Returns
    -------
    aligned_coords : torch.Tensor
        Tensor of shape (..., N, 3) containing the aligned coordinates.
    """
    weights = mask.to(coords.dtype)
    return weighted_rigid_align(coords, target, weights, mask, eps)


def weighted_rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
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
    weights : torch.Tensor
        Tensor of shape (..., N) containing weights for each point.
    mask : torch.Tensor
        Tensor of shape (..., N) indicating valid points (1 for valid, 0 for invalid).
    eps : float, optional
        Small value added for numerical stability (default: 1e-8).

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

    L = coords.shape[-2]
    if L < 4:
        warnings.warn(
            f"Point cloud has only {L} points (< 4). "
            "Weighted rigid alignment may not produce a unique rotation.",
            stacklevel=2,
        )

    weights = weights * mask
    w_sum = weights.sum(dim=-1, keepdim=True) + eps

    coords_center = (coords * weights[..., None]).sum(dim=-2, keepdim=True) / w_sum[
        ..., None
    ]
    target_center = (target * weights[..., None]).sum(dim=-2, keepdim=True) / w_sum[
        ..., None
    ]

    coords = coords - coords_center
    target = target - target_center

    H = torch.einsum(
        "...ni, ...nj -> ...ij",
        coords * weights[..., None],
        target,
    )

    with torch.autocast(device_type="cuda", dtype=torch.float32):
        try:
            U, _, V = torch.linalg.svd(H)

            # Fixed reflection removal
            F = torch.eye(3, dtype=torch.float32, device=H.device)
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
            RT = torch.eye(3, dtype=torch.float32, device=coords.device)
            RT = RT.tile(*coords.shape[:-2], 1, 1)

        aligned_coords = coords @ RT + target_center

    return aligned_coords.to(original_dtype)
