import dataclasses
import logging
from functools import lru_cache
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.types.structure import Chain, RefStructure
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


@lru_cache(maxsize=21)
def get_atom_order_in_residue(res_name: str) -> tuple[int, ...]:
    """Get the indices of ambiguous atoms for a given residue type."""
    res_name: C.ResidueName = C.ResidueName[res_name]
    assert res_name in C.residue.PROTEIN_RESIDUES, f"Unsupported residue name: {res_name}"
    atom_order = C.atom.protein_atom37_order
    return tuple(atom_order[an.value] for an in C.atom.RESIDUE_ATOMS[res_name])


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
        """Get ApoSampler instance for inference mode.

        Accepts and ignores any additional positional or keyword arguments
        for backward/forward compatibility.
        """
        prob_perturbation = 1.0 if use_perturbation else 0.0
        return cls(prob_perturbation=prob_perturbation)


class ApoInitializer:
    """Class to sample apo structure coordinates for protein chains.

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
        """Get ApoInitializer instance for inference mode.

        Accepts and ignores any additional positional or keyword arguments
        for backward/forward compatibility.
        """
        return cls(ApoInitializerConfig.inference_mode(use_perturbation))

    def __call__(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
    ) -> dict[int, np.ndarray]:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from entity_id to structure file paths and residue indices
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

        Returns
        -------
        apo_coords_dict : dict[int, np.ndarray]
            Mapping from entity_id to apo coordinates of shape [Napo, L, 37, 3].
        """
        return self.sample_apo_structure(struct, lookup, rng)

    def sample_apo_structure(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
    ) -> dict[int, np.ndarray]:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from entity_id to structure file paths and residue indices
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

        Returns
        -------
        apo_coords_dict : dict[int, np.ndarray]
            Mapping from entity_id to apo coordinates of shape [Napo, L, 37, 3].
        """
        # Create new rng for this sampling to avoid affecting global state
        rng = spawn_rng(rng)

        # === Collect the apo coordinates for all protein chains === #
        apo_coords_dict: dict[int, np.ndarray] = {}
        for c in struct.chains:
            if not c.ctype.is_protein:
                # Only protein chains have apo structures.
                continue
            if c.entity_id in apo_coords_dict:
                # Apo coordinates already collected for this entity_id;
                # copy them to this chain too.
                continue

            chain_key = f"{struct.id}_{c.entity_id}"
            apo_coords = self._sample_apo_structure_for_chain(c, lookup, chain_key, rng)
            apo_coords_dict[c.entity_id] = apo_coords

        return apo_coords_dict

    def _sample_apo_structure_for_chain(
        self,
        chain: Chain,
        lookup: dict[int, dict],
        chain_key: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample apo structure coordinates for a single protein chain."""
        assert chain.is_protein, (
            "Apo structure sampling is only applicable to protein chains."
        )
        eid = chain.entity_id
        if eid not in lookup:
            if self.use_holo_if_apo_unavailable:
                self.logger.warning(
                    f"Apo structure not found for entity {chain_key} in lookup. "
                    "Falling back to holo coordinates."
                )
                apo_coords = chain.atom.coords.copy()
            else:
                apo_coords = np.full_like(chain.atom.coords, np.nan)
            return apo_coords

        apo_info = lookup[eid]
        if "seq" not in apo_info or "coords" not in apo_info:
            raise ValueError(
                f"Apo info must contain 'seq' and 'coords' keys: {apo_info.keys()}"
            )
        apo_seq: str = apo_info["seq"]
        apo_coords: np.ndarray = apo_info["coords"]
        if apo_coords.shape != (len(apo_seq), 37, 3):
            raise ValueError(
                f"Apo coordinates shape mismatch: expected ({len(apo_seq)}, 37, 3), "
                f"got {apo_coords.shape}"
            )
        res_map = apo_info.get("residue_map", None)
        rieprody_key = apo_info.get("key", None)

        # === Random perturbation === #
        if self.perturbation is not None and rng.random() < self.prob_perturbation:
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
            assert apo_coords.shape[0] == length, (
                f"Apo coordinates length {apo_coords.shape[0]} does not match "
                f"sequence length {length} and no residue_map provided."
            )

        # === Map apo coordinates to full atom set === #
        atom37_order = C.atom.protein_atom37_order
        # Insert apo coordinates into chain according to atom order
        # [L, Natom, 3] -> [Nallatoms, 3]
        src_res_indices: list[int] = []
        src_atom_indices: list[int] = []
        dst_atom_indices: list[int] = []
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        atom_names: list[str] = chain.atom.name.tolist()  # pre-converted to list
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based residue index
            if chain.residue.is_standard[res_i]:
                # Standard residues are assumed to have complete atom sets.
                atom_st = chain.residue.atom_starts[res_i]
                atom_orders = get_atom_order_in_residue(ccd_sequence[res_i])
                natoms = len(atom_orders)
                src_res_indices.extend([res_i] * natoms)
                src_atom_indices.extend(atom_orders)
                dst_atom_indices.extend(range(atom_st, atom_st + natoms))

            else:
                for atom_i in chain.residue.iter_residue_atoms(residue_index):
                    an = atom_names[atom_i]
                    if an in atom37_order:
                        src_res_indices.append(res_i)
                        src_atom_indices.append(atom37_order[an])
                        dst_atom_indices.append(atom_i)

        apo_flat = np.full((chain.num_atoms, 3), np.nan, dtype=np.float32)
        apo_flat[dst_atom_indices] = apo_coords[src_res_indices, src_atom_indices]
        return apo_flat

    def _apply_perturbation(
        self,
        sequence: str,
        apo_coords: np.ndarray,
        rng: np.random.Generator,
        backend: str = "bioprior",
        rieprody_key: str | None = None,
    ) -> np.ndarray:
        """Augment apo structure coordinates with perturbation.

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
        apo_mask = np.isfinite(apo_coords).all(axis=-1)
        if not apo_mask.any():
            # No valid apo coordinates, return as is
            return apo_coords
        aug_coords = self.perturbation(
            sequence, apo_coords, rng=rng, backend=backend, rieprody_key=rieprody_key
        )
        aug_coords[~apo_mask] = np.nan
        return aug_coords
