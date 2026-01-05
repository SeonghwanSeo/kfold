"""Polymer prior sampler including globular prior via Langevin dynamics.

Globular prior:
    Algorithm S3 "Sampling from the Globular Polymer Prior via Short Langevin Dynamics".
    of NeuralPLexer3 (Qiao et al., 2025, https://arxiv.org/abs/2412.10743)
."""

import dataclasses

import numpy as np

from kfold.data.utils.simulation.langevin_dynamics import run_langevin_dynamics


@dataclasses.dataclass(kw_only=True)
class PolymerPriorConfig:
    """Configuration for PolymerPriorSampler."""

    type: str = "globular"  # Type of prior: "null", "zero", "normal", "globular".
    # Langevin dynamics parameters.
    num_steps: int = 64
    dt: float = 0.25
    res_r: float = 4.0
    ent_r: float = 10.0
    bond_r: float = 2.0
    sphere_r: float = 10.0


class PolymerPriorSampler:
    def __init__(self, config: PolymerPriorConfig) -> None:
        """Polymer prior sampler using Langevin dynamics."""
        self.config: PolymerPriorConfig = config
        self.prior_type: str = config.type
        self.num_steps: int = config.num_steps
        self.dt: float = config.dt
        self.res_r: float = config.res_r
        self.ent_r: float = config.ent_r
        self.bond_r: float = config.bond_r
        self.sphere_r: float = config.sphere_r

        assert self.prior_type in ["null", "zero", "normal", "globular"], (
            f"Unknown prior type: {self.prior_type}"
        )

    def sample(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior using Langevin dynamics.

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        x_init : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        if self.prior_type == "null":
            return self.sample_null(mask)
        elif self.prior_type == "zero":
            return self.sample_zero(mask)
        elif self.prior_type == "normal":
            return self.sample_normal(mask, rng)
        elif self.prior_type == "globular":
            return self.sample_globular(mask, rng)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def sample_null(self, mask: np.ndarray) -> np.ndarray:
        """Sample null prior (fully-masked).

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].

        Returns
        -------
        x_init : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        L, Natoms = mask.shape
        x_init = np.full((L, Natoms, 3), np.nan, dtype=np.float32)
        return x_init

    def sample_zero(self, mask: np.ndarray) -> np.ndarray:
        """Sample zero prior.

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].

        Returns
        -------
        x_init : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        L, Natoms = mask.shape
        x_init = np.zeros((L, Natoms, 3), dtype=np.float32)
        # Set masked positions to NaN.
        x_init[~mask] = np.nan
        return x_init

    def sample_normal(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample normal prior.

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        x_init : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        L, Natoms = mask.shape
        x_init = (
            rng.standard_normal(size=(L, Natoms, 3), dtype=np.float32) * self.sphere_r
        )
        # Set masked positions to NaN.
        x_init[~mask] = np.nan
        return x_init

    def sample_globular(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior using Langevin dynamics.

        Parameters
        ----------
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        x_init : np.ndarray
            Sampled coordinates of shape [L, Natom, 3].
        """
        L, Natoms = mask.shape
        x_init = (
            rng.standard_normal(size=(L, Natoms, 3), dtype=np.float32) * self.sphere_r
        )
        x_init[~mask] = 0.0  # Initialize masked positions to zero for dynamics.
        x_init = run_langevin_dynamics(
            x_init,
            mask,
            num_steps=self.num_steps,
            dt=self.dt,
            res_r=self.res_r,
            ent_r=self.ent_r,
            bond_r=self.bond_r,
            sphere_r=self.sphere_r,
            rng=rng,
        )
        # Set masked positions back to NaN.
        x_init[~mask] = np.nan
        return x_init
