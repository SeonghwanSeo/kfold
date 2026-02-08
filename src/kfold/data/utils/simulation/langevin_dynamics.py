"""Langevin dynamics-based prior sampling for polymer structures introduced by
NeuralPLexer3 (Qiao et al., 2025, https://arxiv.org/abs/2412.10743)
Algorithm S3 "Sampling from the Globular Polymer Prior via Short Langevin Dynamics".

NOTE (SeonghwanSeo): Since bond information is not constructed in current data pipeline,
I modified the original algorithm to approximate bond forces using residue centers.
."""

import dataclasses
from typing import Self

import numpy as np
from omegaconf import DictConfig, OmegaConf


@dataclasses.dataclass(kw_only=True)
class LangevinDynamicsConfig:
    """Configuration for Langevin dynamics simulator."""

    num_steps: int = 64
    dt: float = 0.25
    bond_r: float = 2.0
    res_r: float = 4.0
    ent_r: float = 10.0
    sphere_r: float = 10.0

    @classmethod
    def from_config(cls, config: DictConfig | Self) -> Self:
        """Create LangevinDynamicsConfig from DictConfig"""
        base_cfg = OmegaConf.structured(cls)
        merged_cfg = OmegaConf.merge(base_cfg, config)
        return OmegaConf.to_object(merged_cfg)


class LangevinDynamicsSimulator:
    def __init__(self, config: LangevinDynamicsConfig) -> None:
        """Initialize Langevin dynamics simulator."""
        config: LangevinDynamicsConfig = LangevinDynamicsConfig.from_config(config)
        self.num_steps: int = config.num_steps
        self.dt: float = config.dt
        self.bond_r: float = config.bond_r
        self.res_r: float = config.res_r
        self.ent_r: float = config.ent_r
        self.sphere_r: float = config.sphere_r

    def __call__(
        self,
        x_init: np.ndarray,
        residue_index: np.ndarray,
        is_constraint: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.simulate(
            x_init,
            residue_index,
            is_constraint=is_constraint,
            rng=rng,
        )

    def simulate(
        self,
        x_init: np.ndarray,
        residue_index: np.ndarray,
        is_constraint: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run Langevin dynamics simulation.

        Parameters
        ----------
        x_init : np.ndarray
            Initial coordinates of shape [Natoms, 3].
        residue_index : np.ndarray
            Residue membership for each atom, shape [Natoms].
            Assumes 0-indexed, contiguous integers (0 to L-1).
        is_constraint : np.ndarray
            Constraint mask of shape [Natoms], indicating fixed atoms.
        rng : np.random.Generator, optional
            Random number generator for stochastic operations.

        Return
        ------
        sampled_coords : np.ndarray
            Sampled coordinates after Langevin dynamics of shape [Natoms, 3].
        """
        return run_langevin_dynamics(
            x_init,
            residue_index,
            num_steps=self.num_steps,
            dt=self.dt,
            bond_r=self.bond_r,
            res_r=self.res_r,
            ent_r=self.ent_r,
            sphere_r=self.sphere_r,
            is_constraint=is_constraint,
            rng=rng,
        )


def scatter_mean(
    src: np.ndarray,
    index: np.ndarray,
    dim: int = 0,
    dim_size: int | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Computes the mean of values in `src` array into `out` array at indices specified by
    `index`.

    Equivalent to:
        out[index[i]] = mean(src[i])

    Args:
        src: The source array of values.
        index: The indices of elements to scatter. Must be broadcastable to src.shape.
        dim: The axis along which to index.
        dim_size: The size of the output along `dim`. If None, inferred from max(index).
        fill_value: Value to fill in output where no index points (empty bins).

    Returns:
        The output array with averaged values.
    """
    # 1. Handle dimensionality and broadcasting
    # Move the target dimension to the front (axis 0) for consistent handling
    src = np.moveaxis(src, dim, 0)
    index = np.moveaxis(index, dim, 0)

    # If index is 1D (common case), broadcast it to match src shape for the scatter
    if index.ndim < src.ndim:
        index = np.expand_dims(index, axis=tuple(range(1, src.ndim)))
        index = np.broadcast_to(index, src.shape)

    # 2. Determine Output Size
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Output shape: [dim_size, ...other_dims...]
    out_shape = list(src.shape)
    out_shape[0] = dim_size

    # 3. Scatter Sum (Numerator)
    # using np.zeros_like to preserve dtype
    out_sum = np.zeros(out_shape, dtype=src.dtype)
    np.add.at(out_sum, index, src)

    # 4. Scatter Count (Denominator)
    # We scatter '1's into the same indices to count occurrences
    out_count = np.zeros(out_shape, dtype=src.dtype)
    np.add.at(out_count, index, 1)

    # 5. Compute Mean (Sum / Count)
    # Handle division by zero safely (where count is 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = out_sum / out_count

    # Replace NaNs (from 0/0 division) with the fill_value
    out[np.isnan(out)] = fill_value

    # 6. Restore original dimensionality
    return np.moveaxis(out, 0, dim)


def run_langevin_dynamics(
    x_init: np.ndarray,
    residue_index: np.ndarray,
    num_steps: int = 64,
    dt: float = 0.25,
    bond_r: float = 2.0,
    res_r: float = 4.0,
    ent_r: float = 10.0,
    sphere_r: float = 10.0,
    is_constraint: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Langevin dynamics simulation for flat coordinate arrays.

    Parameters
    ----------
    x_init : np.ndarray
        Initial coordinates of shape [Natoms, 3].
    residue_index : np.ndarray
        Residue membership for each atom, shape [Natoms].
        Assumes 0-indexed, contiguous integers (0 to L-1).
    num_steps : int
        Number of Langevin dynamics steps to perform.
    dt : float
        Time step for Langevin dynamics.
    bond_r : float
        Bond constraint scaling factor.
    res_r : float
        Residue constraint scaling factor.
    ent_r : float
        Global entropic constraint scaling factor.
    sphere_r : float
        Spherical constraint scaling factor.
    is_constraint : np.ndarray
        Constraint mask of shape [Natoms], indicating fixed atoms.
    rng : np.random.Generator, optional
        Random number generator for stochastic operations.

    Return
    ------
    sampled_coords : np.ndarray
        Sampled coordinates after Langevin dynamics of shape [Natoms, 3].
    """
    # 0. Validate and Setup
    if x_init.size == 0:
        return x_init

    if is_constraint is not None:
        if not is_constraint.any():
            is_constraint = None
        else:
            is_constraint = is_constraint[..., None]

    rng = rng or np.random.default_rng()
    dtype = x_init.dtype

    # Determine number of residues (L)
    L = int(residue_index.max()) + 1

    # Pre-calculate counts per residue for mean computation
    # shape: [L, 1]
    res_counts = np.bincount(residue_index, minlength=L).astype(dtype)
    res_counts = np.maximum(res_counts, 1.0)[..., None]  # Avoid div/0

    # Pre-calculate scaling factors
    res_r2 = res_r**2
    ent_r2 = ent_r**2
    bond_r2 = bond_r**2
    sphere_r2 = sphere_r**2
    noise_scale = float(2.0 * np.sqrt(dt))

    x = x_init
    for _ in range(num_steps):
        # --- 1. Calculate Centers ---

        # A. Global Center of Mass
        # Simple mean over all atoms
        center_of_mass = x.mean(axis=0)  # [3]
        d_ent = center_of_mass - x  # [Natoms, 3]

        # B. Residue Centers
        # Scatter sum: Sum atom coords into residue bins
        res_sum = np.zeros((L, 3), dtype=dtype)
        np.add.at(res_sum, residue_index, x)

        # [L, 3]
        center_of_res = res_sum / res_counts

        # --- 2. Calculate Chain Bond Drift (Residue Level) ---
        # Logic: Residues are connected linearly (0-1-2-...)
        # Force = (Neighbor_Center - Current_Center)
        d_bond_res = np.zeros_like(center_of_res)

        # Pull towards Next (i -> i+1)
        if L > 1:
            d_bond_res[:-1] += center_of_res[1:] - center_of_res[:-1]

        # Pull towards Prev (i -> i-1)
        if L > 1:
            d_bond_res[1:] += center_of_res[:-1] - center_of_res[1:]

        # Broadcast residue forces back to atoms
        # Gather: [L, 3] -> [Natoms, 3] using index
        d_bond = d_bond_res[residue_index]

        # --- 3. Residue Constraint (Atom Level) ---
        # Pull atoms towards their own residue center
        # Gather center_of_res to match atom shape
        d_res = center_of_res[residue_index] - x

        # --- 4. Total Drift ---
        drift = (d_ent / ent_r2) + (d_res / res_r2) + (d_bond / bond_r2) - (x / sphere_r2)

        # --- 5. Update ---
        eps = rng.standard_normal(size=x.shape, dtype=dtype)
        x = x + (dt * drift) + (noise_scale * eps)

        # --- 6. Apply Constraints ---
        if is_constraint is not None:
            x = np.where(is_constraint, x_init, x)

    if is_constraint is None:
        # Final Centering if no constraints
        final_center = x.mean(axis=0)
        return x - final_center
    return x


def run_langevin_dynamics_polymer(
    x_init: np.ndarray,
    mask: np.ndarray,
    num_steps: int = 64,
    dt: float = 0.25,
    bond_r: float = 2.0,
    res_r: float = 4.0,
    ent_r: float = 10.0,
    sphere_r: float = 10.0,
    is_constraint: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Langevin dynamics simulation for polymer

    Parameters
    ----------
    x_init : np.ndarray
        Initial coordinates of shape [L, Natoms, 3].
    mask : np.ndarray
        Existence mask of shape [L, Natoms].
    num_steps : int
        Number of Langevin dynamics steps to perform.
    dt : float
        Time step for Langevin dynamics.
    bond_r : float
        Bond constraint scaling factor.
    res_r : float
        Residue constraint scaling factor.
    ent_r : float
        Global entropic constraint scaling factor.
    sphere_r : float
        Spherical constraint scaling factor.
    is_constraint : np.ndarray
        Constraint mask of shape [L, Natoms], indicating fixed atoms.
    rng : np.random.Generator, optional
        Random number generator for stochastic operations.

    Return
    ------
    sampled_coords : np.ndarray
        Sampled coordinates after Langevin dynamics of shape [L, Natoms, 3].
    """
    if not mask.any():
        # If no atoms are present, return the initial coordinates
        return x_init

    x_flat = x_init[mask]

    residue_index = np.repeat(np.arange(x_init.shape[0]), mask.sum(axis=1))

    if is_constraint is not None:
        is_constraint = is_constraint[mask]

    x_sampled_flat = run_langevin_dynamics(
        x_flat,
        residue_index,
        num_steps,
        dt,
        bond_r,
        res_r,
        ent_r,
        sphere_r,
        is_constraint=is_constraint,
        rng=rng,
    )
    x_sampled = np.zeros_like(x_init)
    x_sampled[mask] = x_sampled_flat
    return x_sampled
