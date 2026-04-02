"""Time-dependent COM / internal MSE loss coefficients."""

import torch


def linear_com_internal_weights(
    t_hat: torch.Tensor | None,
    *,
    com_weight: float = 1.0,
    internal_weight: float = 1.0,
    c_com: float = 1.0,
    c_internal: float = 1.0,
) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    """Time-dependent COM/internal coefficients.

    λ_com(t) = (1 + c_com * t) * com_weight.
    λ_int(t) = (1 - c_internal * t) * internal_weight.

    When ``t_hat`` is missing, uses ``t = 0`` (both λ reduce to the base weights).
    """
    if t_hat is None or not torch.is_tensor(t_hat):
        return com_weight, internal_weight
    lam_com = 1.0 + c_com * t_hat
    lam_int = 1.0 - c_internal * t_hat
    return lam_com * com_weight, lam_int * internal_weight
