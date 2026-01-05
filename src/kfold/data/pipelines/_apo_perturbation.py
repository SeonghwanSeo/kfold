"""Apo structure perturbation module using RieProDy and Langevin dynamics."""

import dataclasses

import numpy as np

import kfold.constants as C
from kfold.data.utils.simulation.langevin_dynamics import run_langevin_dynamics
from kfold.data.utils.simulation.rieprody import RiePrody, RieProdyConfig

ATOM37_ORDER: dict[str, int] = C.atom.protein_atom37_order


@dataclasses.dataclass(kw_only=True)
class LangevinConfig:
    # Langevin dynamics parameters (fallback)
    min_steps: int = 1
    max_steps: int = 3
    dt: float = 0.25
    res_r: float = 4.0
    bond_r: float = 4.0
    ent_r: float = 10.0
    sphere_r: float = 10.0


@dataclasses.dataclass(kw_only=True)
class ApoPerturbationConfig:
    prob_rieprody: float = 1.0
    rieprody: RieProdyConfig | None = None
    langevin: LangevinConfig = dataclasses.field(default_factory=LangevinConfig)


class ApoPerturbation:
    """Class to handle apo structure perturbation with RieProDy."""

    def __init__(self, config: ApoPerturbationConfig) -> None:
        """Initialize ApoPerturbation."""
        self.config = config
        self.prob_rieprody: float = config.prob_rieprody
        self.langevin: LangevinConfig = config.langevin
        if config.rieprody is not None:
            self.rieprody = RiePrody(config.rieprody)
        else:
            self.rieprody = None

    def __call__(
        self,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates."""
        return self.run(coords, mask, rng, name)

    def run(
        self,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 14, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 14].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        key : str | None
            Key for lmdb lookup / logging for rieprody perturbation.

        Returns
        -------
        perturbed_coords : np.ndarray
            Perturbed apo structure coordinates of shape [L, 37, 3].
        """
        rng = rng or np.random.default_rng()

        assert coords.ndim == 3 and coords.shape[1] == 37, (
            f"Expected apo_coords shape [L, 37, 3], got {coords.shape}"
        )
        if mask is None:
            # Create mask based on finite coordinates
            # WARN: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(coords).all(axis=-1)

        if self.rieprody is not None and rng.uniform() < self.prob_rieprody:
            # Apply RieProDy perturbation and fallback to Langevin
            perturbed_coords = self.rieprody_perturbation(coords, mask, rng, key)
            if perturbed_coords is None:
                # Fallback to Langevin dynamics perturbation
                perturbed_coords = self.langevin_dynamics_perturbation(coords, mask, rng)
        else:
            # Directly apply Langevin dynamics perturbation
            perturbed_coords = self.langevin_dynamics_perturbation(coords, mask, rng)
        return perturbed_coords

    def rieprody_perturbation(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        key: str | None,
    ) -> np.ndarray | None:
        """Metric-based perturbation using pre-computed LMDB metric + RieProDy RBM.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        metric_data : dict
            Pre-computed metric data from LMDB.
        rng : np.random.Generator
            Random number generator for stochastic operations.
        key : str | None
            Key for lmdb lookup / logging for rieprody perturbation.
        """
        assert self.rieprody is not None, "RieProDy module is not initialized."
        return self.rieprody.run(coords, mask, rng=rng, key=key)

    def langevin_dynamics_perturbation(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturbation using Langevin dynamics as a fallback.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        # Randomly select number of steps for Langevin dynamics
        num_steps = rng.integers(self.langevin.min_steps, self.langevin.max_steps + 1)
        perturbed_coords = run_langevin_dynamics(
            x_init=coords,
            mask=mask,
            num_steps=int(num_steps),
            dt=self.langevin.dt,
            res_r=self.langevin.res_r,
            bond_r=self.langevin.bond_r,
            ent_r=self.langevin.ent_r,
            sphere_r=self.langevin.sphere_r,
            rng=rng,
        )
        return perturbed_coords
