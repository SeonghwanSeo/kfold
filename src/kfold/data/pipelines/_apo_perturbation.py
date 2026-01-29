"""Apo structure perturbation module using RieProDy and Langevin dynamics."""

import dataclasses

import numpy as np

import kfold.constants as C
from kfold.data.utils.simulation.bioprior import BioPriorConfig, BioPriorPerturbation
from kfold.data.utils.simulation.rieprody import RieProdyConfig, RieProdyPerturbation

ATOM37_ORDER: dict[str, int] = C.atom.protein_atom37_order


@dataclasses.dataclass(kw_only=True)
class ApoPerturbationConfig:
    prob_rieprody: float = 1.0
    rieprody: RieProdyConfig | None = None
    bioprior: BioPriorConfig = dataclasses.field(default_factory=BioPriorConfig)


class ApoPerturbation:
    """Class to handle apo structure perturbation with RieProDy."""

    def __init__(self, config: ApoPerturbationConfig) -> None:
        """Initialize ApoPerturbation."""
        self.config: ApoPerturbationConfig = config
        self.prob_rieprody: float = config.prob_rieprody
        if config.rieprody is not None:
            self.rieprody = RieProdyPerturbation(config.rieprody)
        else:
            self.rieprody = None
        self.bioprior = BioPriorPerturbation(config.bioprior)

    def __call__(
        self,
        sequence: str,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates."""
        return self.run(sequence, coords, mask, rng, key)

    def run(
        self,
        sequence: str,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        key: str | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
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
            # HACK: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(coords).all(axis=-1)

        if self.rieprody is not None and rng.uniform() < self.prob_rieprody:
            # Apply RieProDy perturbation and fallback to bioPrior
            perturbed_coords = self.rieprody_perturbation(coords, mask, rng, key)
            if perturbed_coords is None:
                perturbed_coords = self.bioprior_perturbation(sequence, coords, rng)
        else:
            # Directly apply BioPrior perturbation
            perturbed_coords = self.bioprior_perturbation(sequence, coords, rng)

        if perturbed_coords is None:
            # Fallback to original coordinates if both perturbations fail
            return coords
        else:
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

        Returns
        -------
        perturbed_coords : np.ndarray | None
            Perturbed coordinates or None if perturbation failed.
        """
        assert self.rieprody is not None, "RieProDy module is not initialized."
        return self.rieprody.run(coords, mask, rng=rng, key=key)

    def bioprior_perturbation(
        self,
        sequence: str,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Perturbation using BioPrior.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        coords : np.ndarray
            Coordinates of shape [L, 37, 3].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray | None
            Perturbed coordinates or None if perturbation failed.
        """
        return self.bioprior.run(sequence, coords, rng=rng)
