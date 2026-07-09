import dataclasses
import logging
from typing import Self

import numpy as np

from kfold.data.types.ccd import CCD, Component
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

    def __init__(self, config: ApoInitializerConfig, ccd: CCD | None = None) -> None:
        self.config: ApoInitializerConfig = config
        self.ccd = ccd
        self.prob_perturbation: float = config.prob_perturbation
        self.use_cached_conformer_only: bool = config.use_cached_conformer_only
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
    def inference_mode(
        cls, use_perturbation: bool = False, ccd: CCD | None = None
    ) -> Self:
        """Get ApoInitializer instance for inference mode."""
        return cls(ApoInitializerConfig.inference_mode(use_perturbation), ccd=ccd)

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

        # === Collect apo coordinates for polymers and ligand conformers === #
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
            if c.is_ligand:
                apo_coords_dict[c.asym_id] = self._sample_ligand_conformer(c, rng)
                continue
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

    def _sample_ligand_conformer(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a ligand apo conformer in chain atom order.

        Atoms that cannot be matched to the generated/reference conformer are
        intentionally left as NaN; PriorSampler fills them by Langevin relaxation.
        """
        assert chain.is_ligand, "Ligand conformer sampling only applies to ligands."

        coords = np.full_like(chain.atom.coords, np.nan, dtype=np.float32)
        ccd_sequence = chain.get_ccd_sequence()
        if chain.smiles is not None:
            assert chain.num_residues == 1, "Multiple residues with SMILES not supported."
            comp = Component.from_smiles(ccd_sequence[0], chain.smiles)
            ref_pos = self._get_ligand_ref_conformer(comp, rng)
            return self._insert_component_conformer(
                coords,
                chain,
                residue_index=1,
                component=comp,
                ref_pos=ref_pos,
                allow_order_fallback=True,
            )

        if self.ccd is None:
            self.logger.warning(
                f"CCD is unavailable for ligand chain {chain.asym_id}. "
                "Filling apo coordinates with NaN."
            )
            return coords

        for res_i, code in enumerate(ccd_sequence):
            residue_index = res_i + 1
            if code not in self.ccd:
                self.logger.warning(
                    f"CCD code {code} not found for ligand chain {chain.asym_id}. "
                    "Filling missing atoms with NaN."
                )
                continue
            comp = self.ccd[code]
            ref_pos = self._get_ligand_ref_conformer(comp, rng)
            coords = self._insert_component_conformer(
                coords,
                chain,
                residue_index=residue_index,
                component=comp,
                ref_pos=ref_pos,
                allow_order_fallback=False,
            )
        return coords

    def _get_ligand_ref_conformer(
        self,
        component: Component,
        rng: np.random.Generator,
    ) -> np.ndarray:
        conformer_type = "train" if self.use_cached_conformer_only else "auto"
        timeout = 5 if self.use_cached_conformer_only else 30
        coords = component.get_conformer(conformer_type, rng=rng, timeout=timeout)
        if coords is None:
            coords = component.get_conformer("nan", rng=rng)
        assert coords is not None
        return coords.astype(np.float32)

    @staticmethod
    def _insert_component_conformer(
        coords: np.ndarray,
        chain: Chain,
        residue_index: int,
        component: Component,
        ref_pos: np.ndarray,
        allow_order_fallback: bool,
    ) -> np.ndarray:
        ref_atom_order = component.get_atom_index_map()
        atom_names = chain.atom.name.tolist()
        residue_atom_indices = list(chain.residue.iter_residue_atoms(residue_index))

        src_atom_indices: list[int] = []
        dst_atom_indices: list[int] = []
        for atom_i in residue_atom_indices:
            atom_name = atom_names[atom_i]
            src_i = ref_atom_order.get(atom_name)
            if src_i is None:
                continue
            src_atom_indices.append(src_i)
            dst_atom_indices.append(atom_i)

        if src_atom_indices:
            coords[dst_atom_indices] = ref_pos[src_atom_indices]
            return coords

        if allow_order_fallback and len(residue_atom_indices) == ref_pos.shape[0]:
            coords[residue_atom_indices] = ref_pos
        return coords

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
