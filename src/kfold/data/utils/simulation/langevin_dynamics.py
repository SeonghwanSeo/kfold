"""Langevin dynamics-based prior sampling for polymer structures introduced by
NeuralPLexer3 (Qiao et al., 2025, https://arxiv.org/abs/2412.10743)
Algorithm S3 "Sampling from the Globular Polymer Prior via Short Langevin Dynamics".

NOTE (SeonghwanSeo): Since bond information is not constructed in current data pipeline,
I modified the original algorithm to approximate bond forces using residue centers.
."""

import numpy as np


def run_langevin_dynamics(
    x_init: np.ndarray,
    mask: np.ndarray,
    num_steps: int = 64,
    dt: float = 0.25,
    bond_r: float = 2.0,
    res_r: float = 4.0,
    ent_r: float = 10.0,
    sphere_r: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Langevin dynamics simulation."""
    if mask.sum() == 0:
        # If no atoms are present, return the initial coordinates
        return x_init

    rng = rng or np.random.default_rng()
    x = np.where(mask[..., None], x_init, 0.0).astype(np.float32)

    L, Natoms = mask.shape
    w_all = mask.astype(np.float32)[..., None]

    # Pre-calculate inverse weights
    w_res = w_all.sum(axis=1).clip(min=1)
    inv_w_res = 1.0 / w_res

    denom_all = float(w_all.sum())
    inv_denom_all = 1.0 / denom_all if denom_all > 0 else 0.0

    # Pre-calculate scaling factors
    res_r2 = res_r**2
    ent_r2 = ent_r**2
    bond_r2 = bond_r**2
    sphere_r2 = sphere_r**2
    noise_scale = float(2.0 * np.sqrt(dt))

    for _ in range(num_steps):
        # 1. Calculate Centers
        # Global Center
        center_of_mass = (x * w_all).sum(axis=(0, 1)) * inv_denom_all
        d_ent = center_of_mass.reshape(1, 1, 3) - x

        # Residue Centers [L, 3]
        center_of_res = (x * w_all).sum(axis=1) * inv_w_res

        # 2. Calculate Chain Bond Drift (Residue Level)
        # Shift logic to compute (C_prev - C_curr) + (C_next - C_curr)
        # d_bond_res shape: [L, 3]
        d_bond_res = np.zeros_like(center_of_res)

        # Pull towards Next (i -> i+1)
        # center_of_res[1:] is C_{i+1}, center_of_res[:-1] is C_i
        d_bond_res[:-1] += center_of_res[1:] - center_of_res[:-1]

        # Pull towards Prev (i -> i-1)
        # center_of_res[:-1] is C_{i-1}, center_of_res[1:] is C_i
        d_bond_res[1:] += center_of_res[:-1] - center_of_res[1:]

        # Broadcast Residue bond force to atoms: [L, 3] -> [L, 1, 3] -> [L, Natoms, 3]
        # This force is applied uniformly to all atoms in the residue
        d_bond = d_bond_res.reshape(L, 1, 3)

        # 3. Residue Constraint (Atom Level)
        d_res = center_of_res.reshape(L, 1, 3) - x

        # 4. Total Drift
        drift = (d_ent / ent_r2) + (d_res / res_r2) + (d_bond / bond_r2) - (x / sphere_r2)

        # 5. Update
        eps = rng.standard_normal(size=x.shape, dtype=np.float32)
        x = x + (dt * drift) + (noise_scale * eps)
        x = x * w_all  # Apply mask

    if not np.isfinite(x).all():
        # If non-finite values are detected, return the initial coordinates
        return x_init

    final_center = (x * w_all).sum(axis=(0, 1)) * inv_denom_all
    x = x - final_center.reshape(1, 1, 3)
    return x
