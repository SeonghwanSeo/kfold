import dataclasses
import logging
from typing import Self

import numpy as np

from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.geometry.rigid_align import rigid_align
from kfold.utils.misc import spawn_rng

from ._protein_perturbation import ProteinPerturbation, ProteinPerturbationConfig


# === Helper functions === #
def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example:
        "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, apo_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    apo_st, apo_end = map(int, apo_range.split(":"))
    if (res_end - res_st) != (apo_end - apo_st):
        raise ValueError(f"Residue range length mismatch: {residue_map}")
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    return res_st - 1, res_end, apo_st - 1, apo_end


@dataclasses.dataclass(kw_only=True)
class ApoInitializerConfig:
    """Configuration for ApoInitializer.

    Attributes
    ----------
    prob_perturbation : float
        Probability of applying perturbation to apo structures.
    use_cached_conformer_only : bool
        Whether to use cached conformers only for small molecules.
    use_holo_if_apo_unavailable : bool
        Whether to fallback to holo coordinates if apo structures are
        unavailable.
    perturbation : ProteinPerturbationConfig | None
        Configuration for protein apo perturbation.
    """

    prob_perturbation: float = 0.0
    use_cached_conformer_only: bool = False
    use_holo_if_apo_unavailable: bool = False
    perturbation: ProteinPerturbationConfig | None = None

    @classmethod
    def inference_mode(cls, use_perturbation: bool = False) -> Self:
        """Get ApoSampler instance for inference mode."""
        prob_perturbation = 1.0 if use_perturbation else 0.0
        return cls(prob_perturbation=prob_perturbation)


class ApoInitializer:
    """Class to sample apo structure coordinates for protein/RNA/DNA chains.

    NOTE: Each apo structures (shape: [Napo, L, 37, 3]) is ordered according to
    following priority:
      - No perturbed apo structure from file
      - Perturbed apo structure(s) with RieProDy if available and enabled
      - Perturbed apo structure(s) with BioPrior if available

    In particular, we hard-code that the first apo structure (apo index 0) is always
    the raw apo structure without perturbation (if available), which is important for
    model input consistency between training and inference.
    """

    def __init__(self, config: ApoInitializerConfig) -> None:
        self.config: ApoInitializerConfig = config
        self.prob_perturbation: float = config.prob_perturbation
        self.use_holo_if_apo_unavailable: bool = config.use_holo_if_apo_unavailable

        # Apo protein perturbation module
        if self.prob_perturbation > 0.0 and config.perturbation is None:
            raise ValueError(
                "Protein perturbation config must be provided if prob_perturbation > 0.0"
            )
        if config.perturbation is not None:
            self.perturbation = ProteinPerturbation(config.perturbation)
        else:
            self.perturbation = None

        # Logger
        self.logger = logging.getLogger("ApoInitializer")

    @classmethod
    def inference_mode(cls, use_perturbation: bool = False) -> Self:
        """Get ApoInitializer instance for inference mode."""
        return cls(ApoInitializerConfig.inference_mode(use_perturbation))

    def __call__(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
        apply_perturbation: bool = True,
    ) -> dict[int, np.ndarray]:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from asym_id to structure file paths and residue indices
            - input types:
                - seq: str
                - coords: np.ndarray (L, 37, 3)
            - optional keys:
                - key: str
                    Optional key for using pre-computed perturbation with rieprody.
                - residue_map: residue index mapping between holo and apo
                    e.g., "11:100->66:155"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        apply_perturbation : bool
            Whether to apply protein apo perturbation before sequence alignment.

        Returns
        -------
        apo_coords_dict : dict[int, np.ndarray]
            Mapping from asym_id to apo coordinates of shape [Natoms, 3].
        """
        return self.sample_apo_structure(
            struct, lookup, rng, apply_perturbation=apply_perturbation
        )

    def sample_apo_structure(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
        apply_perturbation: bool = True,
    ) -> dict[int, np.ndarray]:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from asym_id to structure file paths and residue indices
            - input types:
                - seq: str
                - coords: np.ndarray (L, 37, 3)
            - optional keys:
                - key: str
                    Optional key for using pre-computed perturbation with
                    rieprody.
                - residue_map: residue index mapping between holo and apo
                    e.g., "11:100->66:155"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        apply_perturbation : bool
            Whether to apply protein apo perturbation before sequence alignment.

        Returns
        -------
        apo_coords_dict : dict[int, np.ndarray]
            Mapping from asym_id to apo coordinates of shape [Natoms, 3].
        """
        # Create new rng for this sampling to avoid affecting global state
        rng = spawn_rng(rng)

        # === Collect the apo coordinates for supported polymer chains === #
        apo_coords_dict: dict[int, np.ndarray] = {}
        entity_cache: dict[int, np.ndarray] = {}
        polymer_asym_ids = {c.asym_id for c in struct.chains if c.is_polymer}
        unknown_keys = set(lookup) - polymer_asym_ids
        if unknown_keys:
            raise KeyError(
                "Apo/Prior lookup must be keyed by polymer asym_id. "
                f"Unknown keys for structure {struct.id}: {sorted(unknown_keys)}"
            )
        for c in struct.chains:
            if not c.is_polymer:
                # Only protein/nucleic-acid chains have apo structures.
                continue

            apo_info = lookup.get(c.asym_id)
            is_multimer_record = apo_info is not None and (
                bool(apo_info.get("is_multimer_apo", False))
                or bool(apo_info.get("is_multimer_prior", False))
            )
            if not is_multimer_record and c.entity_id in entity_cache:
                apo_coords_dict[c.asym_id] = entity_cache[c.entity_id].copy()
                continue

            chain_key = f"{struct.id}_{c.asym_id}"
            apo_coords = self._sample_apo_structure_for_chain(
                c, apo_info, chain_key, rng, apply_perturbation=apply_perturbation
            )
            apo_coords_dict[c.asym_id] = apo_coords
            if not is_multimer_record:
                entity_cache[c.entity_id] = apo_coords.copy()

        return apo_coords_dict

    def _sample_apo_structure_for_chain(
        self,
        chain: Chain,
        apo_info: dict | None,
        chain_key: str,
        rng: np.random.Generator,
        apply_perturbation: bool,
    ) -> np.ndarray:
        """Sample apo structure coordinates for a single polymer chain."""
        assert chain.is_polymer, (
            "Apo structure sampling is only applicable to protein/RNA/DNA chains."
        )
        if apo_info is None:
            if self.use_holo_if_apo_unavailable:
                self.logger.warning(
                    f"Apo structure not found for chain {chain_key} in lookup. "
                    "Falling back to holo coordinates."
                )
                apo_coords = chain.atom.coords.copy()
            else:
                apo_coords = np.full_like(chain.atom.coords, np.nan)
            return apo_coords

        apo_seq: str = apo_info["seq"]
        apo_coords: np.ndarray = apo_info["coords"].copy()
        expected_num_atoms = chain.get_polymer_residue_coord_width()
        assert apo_coords.shape == (len(apo_seq), expected_num_atoms, 3), (
            f"Apo coordinates shape mismatch: expected "
            f"({len(apo_seq)}, {expected_num_atoms}, 3), got {apo_coords.shape}"
        )
        res_map = apo_info.get("residue_map", None)
        assert not chain.is_nucleic_acid or res_map is None, (
            "Nucleic-acid apo structures must be full-length records without "
            "residue_map/apo_range."
        )
        rieprody_key = apo_info.get("key", None)

        # === Random perturbation === #
        if (
            apply_perturbation
            and chain.is_protein
            and self.perturbation is not None
            and rng.random() < self.prob_perturbation
        ):
            apo_coords = self._apply_perturbation(
                apo_seq, apo_coords, rng, rieprody_key=rieprody_key
            )

        # === Align apo coordinates to sequence === #
        length = chain.num_residues
        if res_map is not None:
            res_st, res_end, apo_st, apo_end = parse_residue_map(res_map)
            if res_st == 0 and res_end == length:
                # Simply crop apo_coords without padding
                apo_coords = apo_coords[apo_st:apo_end]
            else:
                # Need to pad apo_coords to match the full sequence length
                padded_apo_coords = np.full(
                    (length, apo_coords.shape[1], 3), np.nan, dtype=np.float32
                )
                padded_apo_coords[res_st:res_end] = apo_coords[apo_st:apo_end]
                apo_coords = padded_apo_coords
        else:
            # No residue map provided, assume apo_coords is already aligned.
            if apo_coords.shape[0] != length:
                self.logger.warning(
                    f"Apo coordinates length {apo_coords.shape[0]} does not match "
                    f"sequence length {length} for chain {chain_key}, and no "
                    "residue_map was provided. Filling apo coordinates with NaN."
                )
                return np.full_like(chain.atom.coords, np.nan)

        return chain.map_polymer_residue_coords_to_atom_coords(
            apo_coords, context="Apo coordinates"
        )

    def _apply_perturbation(
        self,
        sequence: str,
        apo_coords: np.ndarray,
        rng: np.random.Generator,
        backend: str = "auto",
        rieprody_key: str | None = None,
    ) -> np.ndarray:
        """Augment apo coordinates, then rigid-align them back to the input frame.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        apo_coords : np.ndarray
            Apo structure coordinates of shape [L, Natom, 3].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        rieprody_key : str | None
            Optional key for using pre-computed perturbation metrics.

        Returns
        -------
        augmented_coords : np.ndarray
            Augmented structure coordinates of shape [L, Natom, 3].
        """
        assert self.perturbation is not None, (
            "Protein perturbation module is not initialized."
        )
        apo_mask = np.isfinite(apo_coords).all(axis=-1)
        if not apo_mask.any():
            # No valid apo coordinates, return as is
            return apo_coords
        original_coords = apo_coords.copy()
        aug_coords = self.perturbation(
            sequence, apo_coords, rng=rng, backend=backend, rieprody_key=rieprody_key
        )
        aug_mask = np.isfinite(aug_coords).all(axis=-1)
        align_mask = apo_mask & aug_mask
        if align_mask.any():
            aug_coords = rigid_align(
                aug_coords.reshape(-1, 3),
                original_coords.reshape(-1, 3),
                align_mask.reshape(-1),
            ).reshape(aug_coords.shape)
        aug_coords[~align_mask] = np.nan
        return aug_coords
