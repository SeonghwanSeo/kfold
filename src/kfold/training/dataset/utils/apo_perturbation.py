# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein apo structure perturbation using BioPrior during training."""

import dataclasses

import numpy as np

import kfold.constants as C
from kfold.data.utils.simulation.bioprior import BioPriorConfig, BioPriorPerturbation
from kfold.utils.misc import spawn_rng


@dataclasses.dataclass(kw_only=True)
class ApoPerturbationConfig:
    bioprior: BioPriorConfig = dataclasses.field(default_factory=BioPriorConfig)


class ApoPerturbation:
    """Class to handle protein apo structure perturbation"""

    def __init__(self, config: ApoPerturbationConfig) -> None:
        """Initialize ApoPerturbation."""
        self.config: ApoPerturbationConfig = config
        self.bioprior = BioPriorPerturbation(config.bioprior)

    def __call__(
        self,
        chain_type: C.ChainType,
        sequence: str,
        coords: np.ndarray,
        mask: np.ndarray | None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        chain_type : C.ChainType
            Chain type; only protein perturbation is supported.
        sequence : str
            Sequence of the chain.
        coords : np.ndarray
            Apo protein coordinates of shape [L, 37, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, 37].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        perturbed_coords : np.ndarray
            Perturbed apo structure coordinates of shape [L, 37, 3].
        """
        return self.run(chain_type, sequence, coords, mask, rng)

    def run(
        self,
        chain_type: C.ChainType,
        sequence: str,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = spawn_rng(rng)
        if chain_type.is_protein:
            return self.run_protein_perturbation(sequence, coords, mask, rng)
        else:
            raise NotImplementedError(
                "Perturbation is only implemented for protein chains."
            )

    def run_protein_perturbation(
        self,
        sequence: str,
        coords: np.ndarray,
        mask: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        assert coords.ndim == 3 and coords.shape[1:] == (37, 3), (
            f"Expected apo_coords shape [L, 37, 3], got {coords.shape}"
        )
        if mask is None:
            # Create mask based on finite coordinates
            # HACK: assumes that missing atoms are represented by NaN/Inf
            mask: np.ndarray = np.isfinite(coords).all(axis=-1)

        perturbed_coords = self.bioprior.run(sequence, coords, rng=rng)
        if perturbed_coords is None:
            # Keep the original coordinates if perturbation fails.
            return coords

        # Keep the original apo availability contract after perturbation.
        perturbed_coords[~mask] = np.nan
        return perturbed_coords
