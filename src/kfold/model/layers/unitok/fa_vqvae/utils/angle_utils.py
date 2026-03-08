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
