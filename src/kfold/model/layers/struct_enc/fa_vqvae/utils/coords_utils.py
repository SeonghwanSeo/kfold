import torch


def normalize_last_dim(v):
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(norm, min=1e-8, max=None)


def bond_angles(a, b, c):
    """
    Computes bond angles for the 3 points a, b, c.
    Since torch.linalg.cross and torch.linalg.cross  support
    broadcasting, this supports broadcasting.

    Args:
        a, b, c: Each is a tensor of shape [*, 3]

    Returns:
        Angle between 0 and pi, shape [*]
    """
    b0 = b - a  # [*, 3]
    b1 = c - a  # [*, 3]
    b0, b1 = map(normalize_last_dim, (b0, b1))  # [*, 3] each
    cos_angle = torch.linalg.vecdot(b0, b1, dim=-1)  # [*]
    cross = torch.linalg.cross(b0, b1, dim=-1)  # [*, 3]
    sin_angle = torch.linalg.norm(cross, dim=-1)  # [*]
    return torch.atan2(sin_angle, cos_angle)  # [*]


def compute_residue_frames(
    coords: torch.Tensor,
    mask: torch.Tensor,
):
    """Compute per-residue local coordinate frames from backbone N, CA, C atoms.

    Frame convention:
        origin = CA
        x = normalize(N - CA)
        z = normalize(x cross (C - CA))
        y = z cross x

    Args:
        coords: [B, L, 37, 3] atom coordinates (invalid atoms should be 0).
        mask:   [B, L, 37] boolean mask of valid atoms.

    Returns:
        R:              [B, L, 3, 3] rotation matrices (rows = local basis vectors).
        origin:         [B, L, 3] frame origins (CA positions).
        backbone_valid: [B, L] boolean mask of residues with valid N, CA, C.
    """
    N = coords[:, :, 0, :]  # [B, L, 3]
    CA = coords[:, :, 1, :]  # [B, L, 3]
    C = coords[:, :, 2, :]  # [B, L, 3]

    backbone_valid = mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2]  # [B, L]

    v1 = N - CA
    v2 = C - CA

    e1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, v2, dim=-1)
    e3 = e3 / (torch.linalg.norm(e3, dim=-1, keepdim=True) + 1e-8)
    e2 = torch.cross(e3, e1, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-2)  # [B, L, 3, 3]

    return R, CA, backbone_valid


def canonicalize_residue_coords(
    coords: torch.Tensor,
    mask: torch.Tensor,
    frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Canonicalize atom coordinates into per-residue local frames (ESM3-style).

    For each residue, builds an orthonormal frame from backbone atoms N, CA, C
    and expresses all 37 atom positions in that local frame, making the
    representation SE(3)-invariant.

    For residues missing any backbone atom, falls back to simple CA-relative
    coordinates (translation only, no rotation).

    Args:
        coords: [B, L, 37, 3] atom coordinates (invalid atoms should be 0).
        mask:   [B, L, 37] boolean mask of valid atoms.
        frames: optional pre-computed (R, origin, backbone_valid) from
                compute_residue_frames.  When supplied the frame computation
                is skipped, avoiding redundant work and guaranteeing that the
                caller and this function use identical frames.

    Returns:
        canonical_coords: [B, L, 37, 3] coordinates in local residue frames.
    """
    if frames is not None:
        R, origin, backbone_valid = frames
    else:
        R, origin, backbone_valid = compute_residue_frames(coords, mask)

    coords_centered = coords - origin.unsqueeze(2)  # [B, L, 37, 3]
    canonical_coords = torch.einsum("blij,blaj->blai", R, coords_centered)

    fallback = coords_centered
    canonical_coords = torch.where(
        backbone_valid[:, :, None, None].expand_as(canonical_coords),
        canonical_coords,
        fallback,
    )
    canonical_coords = canonical_coords.masked_fill(~mask.unsqueeze(-1), 0.0)

    return canonical_coords


def uncanonicalize_residue_coords(
    canonical_coords: torch.Tensor,
    R: torch.Tensor,
    origin: torch.Tensor,
    backbone_valid: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Inverse of canonicalize: transform local-frame coords back to global space.

    global = R^T @ local + CA

    Args:
        canonical_coords: [B, L, 37, 3] coordinates in local residue frames.
        R:                [B, L, 3, 3] rotation matrices from compute_residue_frames.
        origin:           [B, L, 3] frame origins (CA positions).
        backbone_valid:   [B, L] boolean mask of residues with valid frames.
        mask:             [B, L, 37] boolean mask of valid atoms.

    Returns:
        global_coords: [B, L, 37, 3] coordinates in global space.
    """
    # R^T @ canonical_coords + origin
    global_coords = torch.einsum("blji,blaj->blai", R, canonical_coords)  # R^T
    global_coords = global_coords + origin.unsqueeze(2)

    # Fallback residues were only CA-translated, so just add origin back
    fallback = canonical_coords + origin.unsqueeze(2)
    global_coords = torch.where(
        backbone_valid[:, :, None, None].expand_as(global_coords),
        global_coords,
        fallback,
    )
    global_coords = global_coords.masked_fill(~mask.unsqueeze(-1), 0.0)

    return global_coords
