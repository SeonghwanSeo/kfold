"""Langevin dynamics-based prior sampling for polymer structures introduced by
NeuralPLexer3 (Qiao et al., 2025, https://arxiv.org/abs/2412.10743)
Algorithm S3 "Sampling from the Globular Polymer Prior via Short Langevin Dynamics".

NOTE (SeonghwanSeo): Since bond information is not constructed in current data pipeline,
I modified the original algorithm to approximate bond forces using residue centers.

The ligand simulator below is a separate K-Fold prior perturbation path. It
restores the historical bond-length and bond-angle harmonic potential around a
supplied CCD/ETKDG conformer; it is not NeuralPLexer3 Algorithm S3.
"""

import dataclasses
import functools
from typing import Any, Self

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

    @classmethod
    def default(cls) -> Self:
        """Create a LangevinDynamicsSimulator with default configuration."""
        return cls(LangevinDynamicsConfig())

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


@dataclasses.dataclass(kw_only=True)
class LigandLangevinDynamicsConfig:
    """Configuration for bond-angle harmonic ligand prior perturbation."""

    enabled: bool = False
    backend: str = "vectorized"
    initial_noise_scale: float = 0.0
    num_steps: int = 100
    dt: float = 0.01
    temperature: float = 1.0
    bond_strength: float = 50.0
    angle_strength: float = 10.0
    relaxation_steps: int = 2
    relaxation_dt: float = 0.00125
    max_bond_deviation: float | None = None
    max_aligned_rmsd: float | None = None

    @classmethod
    def from_config(cls, config: DictConfig | Self) -> Self:
        """Create a validated config from a structured or OmegaConf config."""
        base_cfg = OmegaConf.structured(cls)
        merged_cfg = OmegaConf.merge(base_cfg, config)
        return OmegaConf.to_object(merged_cfg)


class LigandLangevinDynamicsSimulator:
    """Sample local ligand conformers around a supplied initial structure."""

    def __init__(self, config: LigandLangevinDynamicsConfig) -> None:
        config = LigandLangevinDynamicsConfig.from_config(config)
        self._validate_config(config)
        self.config = config

    @staticmethod
    def _validate_config(config: LigandLangevinDynamicsConfig) -> None:
        if config.backend not in {"vectorized", "numba"}:
            raise ValueError(
                "Ligand Langevin backend must be 'vectorized' or 'numba'."
            )
        if config.initial_noise_scale < 0.0:
            raise ValueError("Initial ligand noise scale must be non-negative.")
        if config.num_steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if config.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if config.bond_strength < 0.0 or config.angle_strength < 0.0:
            raise ValueError("Harmonic restraint strengths must be non-negative.")
        if config.relaxation_steps < 0:
            raise ValueError("relaxation_steps must be non-negative.")
        if config.relaxation_dt <= 0.0:
            raise ValueError("relaxation_dt must be positive.")
        if (
            config.max_bond_deviation is not None
            and config.max_bond_deviation <= 0.0
        ):
            raise ValueError("max_bond_deviation must be positive.")
        if config.max_aligned_rmsd is not None and config.max_aligned_rmsd <= 0.0:
            raise ValueError("max_aligned_rmsd must be positive.")

    def __call__(
        self,
        x_init: np.ndarray,
        bond_indices: np.ndarray,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return self.simulate(x_init, bond_indices, num_samples, rng)

    def simulate(
        self,
        x_init: np.ndarray,
        bond_indices: np.ndarray,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run restrained Langevin dynamics with the configured backend.

        Parameters
        ----------
        x_init : np.ndarray
            Complete initial ligand coordinates, shape ``[Natom, 3]``.
        bond_indices : np.ndarray
            Chain-local bonded atom pairs, shape ``[Nbond, 2]``.
        num_samples : int
            Number of independent internal conformers to sample.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Perturbed coordinates with shape ``[Nsample, Natom, 3]``.
        """
        if x_init.ndim != 2 or x_init.shape[-1] != 3:
            raise ValueError(f"x_init must have shape [Natom, 3], got {x_init.shape}.")
        if not np.isfinite(x_init).all():
            raise ValueError("x_init must contain only finite coordinates.")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_samples == 0:
            return np.empty((0, *x_init.shape), dtype=x_init.dtype)

        num_atoms = len(x_init)
        bond_indices = _normalize_pair_indices(bond_indices, num_atoms)
        angle_indices = _build_angle_indices(bond_indices, num_atoms)
        rest_bond_lengths = _compute_bond_lengths(x_init, bond_indices)
        rest_angles = _compute_angles(x_init, angle_indices)
        x = np.repeat(x_init[None, ...], num_samples, axis=0)
        if self.config.num_steps == 0 and self.config.relaxation_steps == 0:
            return x

        rng = rng or np.random.default_rng()
        dtype = x_init.dtype
        initial_center = x_init.mean(axis=0, keepdims=True)
        if self.config.initial_noise_scale > 0.0:
            initial_noise = rng.standard_normal(size=x.shape, dtype=dtype)
            initial_noise *= np.asarray(
                self.config.initial_noise_scale,
                dtype=dtype,
            )
            x += initial_noise
            x -= x.mean(axis=1, keepdims=True) - initial_center[None, ...]
        rest_bond_lengths = rest_bond_lengths.astype(dtype, copy=False)
        rest_angles = rest_angles.astype(dtype, copy=False)
        # Historical ligand SDE: dx = -grad(U) dt + sqrt(2 T dt) dW.
        # The NP3 polymer sampler below intentionally uses 2 sqrt(dt).
        noise_scale = np.sqrt(
            np.asarray(2, dtype=dtype)
            * np.asarray(self.config.temperature, dtype=dtype)
            * np.asarray(self.config.dt, dtype=dtype)
        )
        stochastic_noise = np.empty(
            (self.config.num_steps, *x.shape),
            dtype=dtype,
        )
        for step_index in range(self.config.num_steps):
            stochastic_noise[step_index] = rng.standard_normal(
                size=x.shape,
                dtype=dtype,
            )
        stochastic_noise *= noise_scale

        if self.config.backend == "numba":
            kernel = _get_numba_ligand_kernel()
            return kernel(
                x,
                bond_indices,
                angle_indices,
                rest_bond_lengths,
                rest_angles,
                stochastic_noise,
                np.asarray(self.config.dt, dtype=dtype),
                np.asarray(self.config.bond_strength, dtype=dtype),
                np.asarray(self.config.angle_strength, dtype=dtype),
                self.config.relaxation_steps,
                np.asarray(self.config.relaxation_dt, dtype=dtype),
                initial_center,
            )

        bond_incidence = _make_incidence_matrix(bond_indices, num_atoms, dtype)
        bond_i = bond_indices[:, 0]
        bond_j = bond_indices[:, 1]
        if len(angle_indices) > 0:
            angle_i, angle_j, angle_k = angle_indices.T
            angle_i_incidence = _make_incidence_matrix(
                np.stack([angle_i, angle_j], axis=-1), num_atoms, dtype
            )
            angle_k_incidence = _make_incidence_matrix(
                np.stack([angle_k, angle_j], axis=-1), num_atoms, dtype
            )
        eps = np.asarray(1e-8, dtype=dtype)
        cos_limit = np.asarray(1.0 - 1e-7, dtype=dtype)
        total_steps = self.config.num_steps + self.config.relaxation_steps
        for step_index in range(total_steps):
            is_stochastic_step = step_index < self.config.num_steps
            drift = np.zeros_like(x)
            if len(bond_indices) > 0:
                displacement = x[:, bond_i] - x[:, bond_j]
                distance = np.sqrt(
                    np.einsum("...d,...d->...", displacement, displacement)
                )
                is_valid_bond = distance >= eps
                extension = distance - rest_bond_lengths[None, :]
                safe_distance = np.maximum(distance, eps)
                coefficient = -self.config.bond_strength * extension / safe_distance
                coefficient *= is_valid_bond
                bond_drift = displacement * coefficient[..., None]
                drift += np.matmul(bond_incidence, bond_drift)

            if len(angle_indices) > 0:
                vector_i = x[:, angle_i] - x[:, angle_j]
                vector_k = x[:, angle_k] - x[:, angle_j]
                norm_i = np.sqrt(
                    np.einsum("...d,...d->...", vector_i, vector_i)
                )
                norm_k = np.sqrt(
                    np.einsum("...d,...d->...", vector_k, vector_k)
                )
                is_valid_angle = (norm_i >= eps) & (norm_k >= eps)
                safe_norm_i = np.maximum(norm_i, eps)
                safe_norm_k = np.maximum(norm_k, eps)
                unit_i = vector_i / safe_norm_i[..., None]
                unit_k = vector_k / safe_norm_k[..., None]
                cos_angle = np.einsum("...d,...d->...", unit_i, unit_k)
                np.clip(cos_angle, -cos_limit, cos_limit, out=cos_angle)
                angle = np.arccos(cos_angle)
                sin_angle = np.sqrt(np.maximum(1.0 - cos_angle**2, eps))
                angle_extension = angle - rest_angles[None, :]
                scale = self.config.angle_strength * angle_extension / sin_angle
                scale *= is_valid_angle
                angle_gradient_i = (
                    cos_angle[..., None] * unit_i - unit_k
                ) * (scale / safe_norm_i)[..., None]
                angle_gradient_k = (
                    cos_angle[..., None] * unit_k - unit_i
                ) * (scale / safe_norm_k)[..., None]
                drift -= np.matmul(angle_i_incidence, angle_gradient_i)
                drift -= np.matmul(angle_k_incidence, angle_gradient_k)

            step_dt = (
                self.config.dt
                if is_stochastic_step
                else self.config.relaxation_dt
            )
            drift *= step_dt
            if is_stochastic_step:
                drift += stochastic_noise[step_index]
            x += drift

        # Remove the translational zero mode without constraining rotation.
        x -= x.mean(axis=1, keepdims=True) - initial_center[None, ...]

        return x


def _run_ligand_langevin_numba_kernel(
    x: np.ndarray,
    bond_indices: np.ndarray,
    angle_indices: np.ndarray,
    rest_bond_lengths: np.ndarray,
    rest_angles: np.ndarray,
    stochastic_noise: np.ndarray,
    dt: np.floating,
    bond_strength: np.floating,
    angle_strength: np.floating,
    num_relaxation_steps: int,
    relaxation_dt: np.floating,
    initial_center: np.ndarray,
) -> np.ndarray:
    """Sparse bond-angle integration implementation for Numba."""
    num_samples, num_atoms, _ = x.shape
    num_stochastic_steps = len(stochastic_noise)
    drift = np.empty_like(x)
    zero = np.float32(0)
    one = np.float32(1)
    eps = one / np.float32(100_000_000)
    cos_limit = np.float32(9_999_999) / np.float32(10_000_000)

    for step_index in range(num_stochastic_steps + num_relaxation_steps):
        drift.fill(zero)
        for sample_index in range(num_samples):
            for bond_index in range(len(bond_indices)):
                atom_i = bond_indices[bond_index, 0]
                atom_j = bond_indices[bond_index, 1]
                dx = x[sample_index, atom_i, 0] - x[sample_index, atom_j, 0]
                dy = x[sample_index, atom_i, 1] - x[sample_index, atom_j, 1]
                dz = x[sample_index, atom_i, 2] - x[sample_index, atom_j, 2]
                distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                if distance < eps:
                    continue
                extension = distance - rest_bond_lengths[bond_index]
                coefficient = -bond_strength * extension / distance
                for coordinate, displacement in enumerate((dx, dy, dz)):
                    force = coefficient * displacement
                    drift[sample_index, atom_i, coordinate] += force
                    drift[sample_index, atom_j, coordinate] -= force

            for angle_index in range(len(angle_indices)):
                atom_i = angle_indices[angle_index, 0]
                atom_j = angle_indices[angle_index, 1]
                atom_k = angle_indices[angle_index, 2]
                vector_i = (
                    x[sample_index, atom_i] - x[sample_index, atom_j]
                )
                vector_k = (
                    x[sample_index, atom_k] - x[sample_index, atom_j]
                )
                norm_i = np.sqrt(
                    vector_i[0] ** 2
                    + vector_i[1] ** 2
                    + vector_i[2] ** 2
                )
                norm_k = np.sqrt(
                    vector_k[0] ** 2
                    + vector_k[1] ** 2
                    + vector_k[2] ** 2
                )
                if norm_i < eps or norm_k < eps:
                    continue
                unit_i = vector_i / norm_i
                unit_k = vector_k / norm_k
                cos_angle = (
                    unit_i[0] * unit_k[0]
                    + unit_i[1] * unit_k[1]
                    + unit_i[2] * unit_k[2]
                )
                cos_angle = min(cos_limit, max(-cos_limit, cos_angle))
                angle = np.arccos(cos_angle)
                sin_angle = np.sqrt(
                    max(one - cos_angle * cos_angle, eps)
                )
                extension = angle - rest_angles[angle_index]
                scale = angle_strength * extension / sin_angle
                gradient_i = (cos_angle * unit_i - unit_k) * (
                    scale / norm_i
                )
                gradient_k = (cos_angle * unit_k - unit_i) * (
                    scale / norm_k
                )
                for coordinate in range(3):
                    drift[sample_index, atom_i, coordinate] -= gradient_i[
                        coordinate
                    ]
                    drift[sample_index, atom_k, coordinate] -= gradient_k[
                        coordinate
                    ]
                    drift[sample_index, atom_j, coordinate] += (
                        gradient_i[coordinate] + gradient_k[coordinate]
                    )

        step_dt = (
            dt if step_index < num_stochastic_steps else relaxation_dt
        )
        for sample_index in range(num_samples):
            for atom_index in range(num_atoms):
                for coordinate in range(3):
                    update = step_dt * drift[
                        sample_index,
                        atom_index,
                        coordinate,
                    ]
                    if step_index < num_stochastic_steps:
                        update += stochastic_noise[
                            step_index,
                            sample_index,
                            atom_index,
                            coordinate,
                        ]
                    x[sample_index, atom_index, coordinate] += update

    for sample_index in range(num_samples):
        center = np.zeros(3, dtype=x.dtype)
        for atom_index in range(num_atoms):
            center += x[sample_index, atom_index]
        center /= num_atoms
        for atom_index in range(num_atoms):
            x[sample_index, atom_index] -= center - initial_center[0]
    return x


@functools.cache
def _get_numba_ligand_kernel() -> Any:
    """Compile the sparse ligand kernel only when the backend is requested."""
    from numba import njit

    return njit(cache=True)(_run_ligand_langevin_numba_kernel)


def _build_angle_indices(bond_indices: np.ndarray, num_atoms: int) -> np.ndarray:
    """Return all unique graph angles ``i-j-k`` induced by ligand bonds."""
    neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    for atom_i, atom_j in bond_indices:
        neighbors[int(atom_i)].append(int(atom_j))
        neighbors[int(atom_j)].append(int(atom_i))

    angle_indices: list[tuple[int, int, int]] = []
    for center, center_neighbors in enumerate(neighbors):
        for first_index in range(len(center_neighbors) - 1):
            for second_index in range(first_index + 1, len(center_neighbors)):
                angle_indices.append(
                    (
                        center_neighbors[first_index],
                        center,
                        center_neighbors[second_index],
                    )
                )
    return np.asarray(angle_indices, dtype=np.int64).reshape(-1, 3)


def _compute_bond_lengths(
    coords: np.ndarray, bond_indices: np.ndarray
) -> np.ndarray:
    """Compute bond lengths for a fixed ligand topology."""
    if len(bond_indices) == 0:
        return np.empty((0,), dtype=coords.dtype)
    displacement = coords[bond_indices[:, 0]] - coords[bond_indices[:, 1]]
    return np.linalg.norm(displacement, axis=-1)


def _compute_angles(coords: np.ndarray, angle_indices: np.ndarray) -> np.ndarray:
    """Compute bond angles in radians for a fixed ligand topology."""
    if len(angle_indices) == 0:
        return np.empty((0,), dtype=coords.dtype)
    atom_i, atom_j, atom_k = angle_indices.T
    vector_i = coords[atom_i] - coords[atom_j]
    vector_k = coords[atom_k] - coords[atom_j]
    norm_i = np.linalg.norm(vector_i, axis=-1)
    norm_k = np.linalg.norm(vector_k, axis=-1)
    denominator = np.maximum(norm_i * norm_k, 1e-8)
    cos_angle = np.einsum("...d,...d->...", vector_i, vector_k) / denominator
    return np.arccos(np.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))


def _normalize_pair_indices(pair_indices: np.ndarray, num_atoms: int) -> np.ndarray:
    """Validate, canonicalize, and deduplicate undirected atom pairs."""
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    if pair_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError(
            f"pair_indices must have shape [Npair, 2], got {pair_indices.shape}."
        )
    if pair_indices.min() < 0 or pair_indices.max() >= num_atoms:
        raise ValueError("pair_indices contains an out-of-range atom index.")
    if np.any(pair_indices[:, 0] == pair_indices[:, 1]):
        raise ValueError("pair_indices cannot contain self edges.")
    pair_indices = np.sort(pair_indices, axis=1)
    return np.unique(pair_indices, axis=0)


def _make_incidence_matrix(
    pair_indices: np.ndarray,
    num_atoms: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Return an atom-by-edge incidence matrix for vectorized force scatter."""
    incidence = np.zeros((num_atoms, len(pair_indices)), dtype=dtype)
    edge_index = np.arange(len(pair_indices))
    incidence[pair_indices[:, 0], edge_index] = 1.0
    incidence[pair_indices[:, 1], edge_index] = -1.0
    return incidence


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

    rng = rng or np.random.default_rng()
    dtype = x_init.dtype

    # Assert the residue_index is ascending ordered
    if not np.all(residue_index[:-1] <= residue_index[1:]):
        raise ValueError("residue_index must be sorted in ascending order.")
    unique_res_ids, start_indices = np.unique(residue_index, return_index=True)
    L = len(unique_res_ids)

    # Pre-calculate counts per residue for mean computation
    # shape: [L, 1]
    res_counts = np.diff(np.append(start_indices, len(residue_index))).astype(dtype)
    res_counts = np.maximum(res_counts, 1.0)[..., None]  # Avoid div/0

    # Pre-calculate scaling factors
    res_r2 = res_r**2
    ent_r2 = ent_r**2
    bond_r2 = bond_r**2
    sphere_r2 = sphere_r**2
    noise_scale = float(2.0 * np.sqrt(dt))

    # Pre-allocate
    center_of_res = np.zeros((L, 3), dtype=dtype)
    d_bond_res = np.zeros((L, 3), dtype=dtype)

    x = x_init.copy()
    if is_constraint is not None:
        x_fixed = x_init[is_constraint]
    for _ in range(num_steps):
        # --- 1. Calculate Centers ---

        # A. Global Center of Mass
        # Simple mean over all atoms
        center_of_mass = x.mean(axis=0)  # [3]
        d_ent = center_of_mass - x  # [Natoms, 3]

        # B. Residue Centers
        # Scatter sum: Sum atom coords into residue bins
        res_sum = np.add.reduceat(x, start_indices, axis=0)
        np.divide(res_sum, res_counts, out=center_of_res)

        # --- 2. Calculate Chain Bond Drift (Residue Level) ---
        # Logic: Residues are connected linearly (0-1-2-...)
        # Force = (Neighbor_Center - Current_Center)
        d_bond_res.fill(0.0)

        if L > 1:
            # Pull towards Next (i -> i+1)
            d_bond_res[:-1] += center_of_res[1:] - center_of_res[:-1]

            # Pull towards Prev (i -> i-1)
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
        x += (dt * drift) + (noise_scale * eps)

        # --- 6. Apply Constraints ---
        if is_constraint is not None:
            x[is_constraint] = x_fixed

    if is_constraint is None:
        # Final Centering if no constraints
        x -= x.mean(axis=0)
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
