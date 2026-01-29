"""Module for apo structure perturbation using BioPrior."""

import dataclasses
import logging
from typing import Self

import numpy as np
from omegaconf import DictConfig, OmegaConf

import kfold.constants as C
from kfold.utils.geometry.rigid_align import compute_rmsd


@dataclasses.dataclass(kw_only=True)
class BioPriorConfig:
    """Configuration for BioPrior perturbation.

    Attributes
    ----------
    noise_scale : float
        Scale of the noise applied during perturbation.
    min_steps : int
        Minimum number of perturbation steps.
    max_steps : int
        Maximum number of perturbation steps.
    scale_length : bool
        Whether to scale noise based on protein length.
        Default is False, since structure will be cropped after perturbation.
    max_rmsd : float | None
        Maximum allowed RMSD after perturbation. If exceeded, perturbation fails.
        Default is None, since structure will be cropped after perturbation.
        Global RMSD threshold is too strict for large proteins.
    """

    noise_scale: float = 1.0
    min_steps: int = 1
    max_steps: int = 30
    scale_length: bool = False
    max_rmsd: float | None = None
    log_level: int | str = "INFO"

    @classmethod
    def from_config(cls, config: DictConfig | Self) -> Self:
        """Create BioPriorConfig using omegaconf merge"""
        base_cfg = OmegaConf.structured(cls)
        merged_cfg = OmegaConf.merge(base_cfg, config)
        return OmegaConf.to_object(merged_cfg)


class BioPriorPerturbation:
    """Class to handle apo structure perturbation with BioPrior."""

    def __init__(self, config: BioPriorConfig) -> None:
        """Initialize Bio-Prior perturbation module."""
        from bioprior.protein import ProteinPerturbation

        config = BioPriorConfig.from_config(config)
        self.config: BioPriorConfig = config

        self._module = ProteinPerturbation(
            ProteinPerturbation.Config(
                global_noise_scale=config.noise_scale,
                length_scale_exponent=1.0 if config.scale_length else 0.0,
            )
        )

        self.logger = logging.getLogger("BioPriorPerturbation")
        self.logger.setLevel(config.log_level)

    # === Main perturbation methods === #
    def run(
        self,
        sequence: str,
        coords: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | None:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 37, 3].
        rng : np.random.Generator, optional
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray | None
            Perturbed apo structure coordinates of shape [L, 37, 3].
            Returns None if perturbation failed.
        """
        # Validate input shapes
        if coords.ndim != 3 or coords.shape[1:] != (37, 3):
            raise ValueError(
                f"Input coords must have shape [L, 37, 3], got {coords.shape}"
            )
        rng = rng or np.random.default_rng()

        try:
            perturbed = self._run_perturbation(sequence, coords, rng)
        except Exception as e:
            self.logger.warning(f"Perturbation failed: {e}")
            return None

        return perturbed

    def _run_perturbation(
        self,
        sequence: str,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        coords : np.ndarray
            Apo protein structure coordinates of shape [L, 37, 3].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray | None
            Perturbed apo structure coordinates of shape [L, 37, 3].
            Returns None if perturbation failed.
        """
        from bioprior.protein import Protein

        # Sanitize sequence: replace non-standard amino acids with 'X'
        sequence = "".join(
            aa if aa in C.residue.PROTEIN_AMINO_ACIDS_SET else "X" for aa in sequence
        )

        seed = int(rng.integers(0, 1_000_000))
        num_steps = int(rng.integers(self.config.min_steps, self.config.max_steps + 1))
        if num_steps == 0:
            return coords

        # Create Protein object and apply perturbation
        obj = Protein(sequence, ref_coords=coords)
        perturbed_coords = self._module.run(
            obj,
            coords,
            num_steps=num_steps,
            seed=seed,
            kabsch_align=True,
        )
        if self.config.max_rmsd is not None:
            rmsd = compute_rmsd(
                coords.reshape(-1, 3),
                perturbed_coords.reshape(-1, 3),
                mask=np.isfinite(perturbed_coords).all(axis=-1).reshape(-1),
            )
            if rmsd > self.config.max_rmsd:
                # Perturbation failed, return None
                return None

        assert perturbed_coords.shape == coords.shape, (
            f"Perturbed coords must have shape {coords.shape}, "
            f"got {perturbed_coords.shape}",
        )
        return perturbed_coords
