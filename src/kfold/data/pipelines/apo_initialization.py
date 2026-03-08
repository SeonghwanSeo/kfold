import dataclasses
import logging
from functools import lru_cache
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.utils.io.structure import read_protein_structure
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import compute_rmsd

from ._protein_perturbation import ProteinPerturbation, ProteinPerturbationConfig
from ._small_mol_perturbation import SmallMolPerturbation, SmallMolPerturbationConfig


# === Helper functions === #
@lru_cache(64)
def get_atom_order_in_residue(res_name: str) -> tuple[int, ...]:
    """Get the indices of ambiguous atoms for a given residue type."""
    res_name: C.ResidueName = C.ResidueName[res_name]
    # determine the chain type based on residue name
    if res_name in C.residue.PROTEIN_RESIDUES:
        atom_order = C.atom.protein_atom37_order
    elif res_name in C.residue.DNA_RESIDUES:
        atom_order = C.atom.nucleic_acid_atom29_order
    elif res_name in C.residue.RNA_RESIDUES:
        atom_order = C.atom.nucleic_acid_atom29_order
    else:
        raise ValueError(f"Unsupported residue name: {res_name}")
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]
    return tuple(atom_order[an.value] for an in residue_atoms)


def get_ref_comp(
    res_name: str, ccd: CCD, cache: dict[str, Component] | None
) -> Component:
    cache = cache if cache is not None else {}
    if res_name not in cache:
        assert res_name in ccd, f"Residue name {res_name} not found in CCD."
        cache[res_name] = ccd[res_name]
    return cache[res_name]


@lru_cache(64)
def get_ambiguous_atoms_in_residue(
    res_name: str,
    extended: bool = False,
) -> list[list[int]] | None:
    """Get the indices of ambiguous atoms for a given residue type."""
    res_name: C.ResidueName = C.ResidueName[res_name]
    if extended:
        ambiguous_atoms_dict = C.atom.RESIDUE_AMBIGUOUS_ATOMS_EXTENDED
    else:
        ambiguous_atoms_dict = C.atom.RESIDUE_AMBIGUOUS_ATOMS
    if res_name not in ambiguous_atoms_dict:
        # If there is no ambiguous atoms, return empty list
        return None
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]
    src_atoms, dst_atoms = C.atom.RESIDUE_AMBIGUOUS_ATOMS_EXTENDED[res_name]
    src_indices = [residue_atoms.index(atom) for atom in src_atoms]
    dst_indices = [residue_atoms.index(atom) for atom in dst_atoms]
    assert set(src_indices).isdisjoint(set(dst_indices)), (
        "Overlapping ambiguous atom indices."
    )
    original_perm = list(range(len(residue_atoms)))
    swap_perm = original_perm.copy()
    for s_idx, d_idx in zip(src_indices, dst_indices, strict=True):
        swap_perm[s_idx] = d_idx
        swap_perm[d_idx] = s_idx
    return [original_perm, swap_perm]  # up to 2 permutations


def get_molecule_symmetries(
    ref_comp: Component, atom_names: list[str]
) -> list[list[int]] | None:
    """Get molecule's symmetries from ccd."""
    symmetries = ref_comp.symmetries
    if symmetries is None or len(symmetries) <= 1:
        # No symmetries
        return None

    if len(atom_names) == ref_comp.num_atoms:
        # All atoms are present, no need to filter
        return list(symmetries)

    # Some atoms are missing (drop_leaving_atoms=True; e.g., covalent ligands)
    valid_atoms: set[str] = set(atom_names)

    name_to_index = ref_comp.get_atom_index_map()
    mol_to_ref_i_map: dict[int, int] = {
        name_to_index[name]: i for i, name in enumerate(atom_names)
    }

    all_perms: list[list[int]] = []  # identity
    for perm in symmetries:
        # Example perm for 4-atom molecules: [0, 2, 1, 3] (swapping atom 1 and 2)
        sym_dict: dict[int, int] = {}
        for i, j in enumerate(perm):
            a_i, a_j = ref_comp.names[i], ref_comp.names[j]
            if a_i not in valid_atoms:
                # atom i is not in the molecule
                continue
            if a_j in valid_atoms:
                # both atoms are in the molecule
                i_true = mol_to_ref_i_map[i]
                j_true = mol_to_ref_i_map[j]
                sym_dict[i_true] = j_true
            else:
                # atom j is not in the molecule
                # skip this symmetry
                break
        else:
            # Completed without break
            # NOTE: This is bijective mapping within valid atoms (see above)
            perm = [sym_dict[i] for i in range(len(valid_atoms))]
            all_perms.append(perm)

    if len(all_perms) <= 1:
        # No symmetries found
        return None

    return all_perms


@dataclasses.dataclass(kw_only=True)
class ApoInitializerConfig:
    """Configuration for ApoInitializer.

    Attributes
    ----------
    use_residue_permutation : bool
        Whether to find optimal residue permutation for symmetry correction.
        NOTE: Training only.
    prob_perturbation : float
        Probability of applying perturbation to apo structures.
    use_cached_conformer_only : bool
        Whether to use cached conformers only for small molecules.
    protein_perturbation : ProteinPerturbationConfig | None
        Configuration for protein apo perturbation.
    ligand_perturbation : SmallMolPerturbationConfig
        Configuration for ligand perturbation.
    """

    use_residue_permutation: bool = False
    prob_perturbation: float = 1.0
    use_cached_conformer_only: bool = False
    use_holo_if_apo_unavailable: bool = False
    protein_perturbation: ProteinPerturbationConfig | None
    ligand_perturbation: SmallMolPerturbationConfig | None

    @classmethod
    def inference_mode(cls) -> Self:
        """Get ApoInitializer instance for inference mode."""
        return cls(
            prob_perturbation=0.0,
            use_residue_permutation=False,
            use_cached_conformer_only=False,
            use_holo_if_apo_unavailable=False,
            protein_perturbation=None,
            ligand_perturbation=None,
        )


class ApoInitializer:
    """Class to populate and augment apo structures."""

    def __init__(
        self,
        config: ApoInitializerConfig,
        ccd: CCD,
        is_protein_monomer_distillation: bool = False,
    ):
        self.config: ApoInitializerConfig = config
        self.use_residue_permutation: bool = config.use_residue_permutation
        self.use_holo_if_apo_unavailable: bool = config.use_holo_if_apo_unavailable

        self.prob_perturbation: float = config.prob_perturbation
        self.ccd: CCD = ccd

        # Protein monomer distillation mode: directly copy holo coordinates to apo.
        # This is for protein monomer synthetic data, such as AFDB or ESM Atlas.
        self.is_protein_monomer_distillation: bool = is_protein_monomer_distillation

        # Apo perturbation module
        if config.protein_perturbation is not None:
            self.protein_perturbation = ProteinPerturbation(config.protein_perturbation)
        else:
            self.protein_perturbation = None

        if config.ligand_perturbation is not None:
            self.ligand_perturbation = SmallMolPerturbation(config.ligand_perturbation)
        else:
            self.ligand_perturbation = None

        # Training mode
        # During train/val, disable ETKDG generation for efficiency,
        # i.e., only the cached ETKDG and CCD conformers (ideal, mode) are used.
        self.conformer_mode: str = "train" if config.use_cached_conformer_only else "auto"

        # Logger
        self.logger = logging.getLogger("ApoInitializer")

    @classmethod
    def inference_mode(cls, ccd: CCD) -> Self:
        """Get ApoInitializer instance for inference mode."""
        return cls(ApoInitializerConfig.inference_mode(), ccd)

    def __call__(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from entity_id to structure file paths and residue indices
            - input types:
                - case1: apo structure file path
                    - path: PathLike
                - case2: sequence and atom37 coordinates
                    - seq: str
                    - coords: np.ndarray (L, 37, 3)
            - optional keys:
                - key: str
                    Optional key for using pre-computed perturbation with rieprody.
                - residue_map: residue index mapping between holo and apo
                    e.g., "11:100->66:155"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        self.populate_apo_structure(struct, lookup, rng)

    def populate_apo_structure(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Populate apo structure coordinates into the reference structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates and masks.
        lookup : dict[int, dict]
            Mapping from entity_id to structure file paths and residue indices
            - input types:
                - case1: apo structure file path
                    - path: PathLike
                - case2: sequence and atom37 coordinates
                    - seq: str
                    - coords: np.ndarray (L, 37, 3)
            - optional keys:
                - key: str
                    Optional key for using pre-computed perturbation with rieprody.
                - residue_map: residue index mapping between holo and apo
                    e.g., "11:100->66:155"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        rng = rng or np.random.default_rng()

        # Insert apo coordinates
        if self.is_protein_monomer_distillation:
            assert struct.num_chains == 1, (
                "Protein monomer distillation only supports single-chain structures."
            )
            self.copy_chain_holo_coords_to_apo(struct.chains[0], rng)
        else:
            self.insert_apo_coordinates(struct, lookup, rng)

        # If symmetry correction is enabled, align apo to holo
        if self.use_residue_permutation:
            self.find_best_residue_permutation(struct)

    def insert_apo_coordinates(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Insert apo structure coordinates for each chain in the structure."""
        self._insert_apo_coordinates_for_protein_chains(struct, lookup, rng)
        self._insert_apo_coordinates_for_ligand_chains(struct, rng)

    def _insert_apo_coordinates_for_protein_chains(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        # cache apo coordinates per entity to avoid redundant loading/sampling
        apo_coords_dict: dict[int, np.ndarray] = {}
        for i in range(struct.num_chains):
            chain = struct.chains[i]
            if not chain.ctype.is_protein:
                # Only protein chains have apo structures in the current implementation.
                continue

            eid: int = chain.entity_id
            ek: str = f"{struct.id}:{eid}"  # For logging purposes
            if eid in apo_coords_dict:
                # Reuse cached apo coordinates
                apo_coords = apo_coords_dict[eid]
            else:
                if eid not in lookup:
                    if self.use_holo_if_apo_unavailable:
                        self.logger.warning(
                            f"Apo structure not found for entity "
                            f"{ek} in lookup. "
                            "Falling back to holo coordinates."
                        )
                        self.copy_chain_holo_coords_to_apo(chain, rng)
                    continue

                apo_info = lookup[eid]
                ccd_sequence = chain.get_ccd_sequence()
                try:
                    apo_coords = self.get_protein_apo_structure(
                        ccd_sequence, apo_info, rng
                    )
                except Exception as e:
                    # NOTE: Apo structure loading can fail for various reasons,
                    # such as too large sequence length, mismatched residue
                    # mapping, or file reading errors.
                    apo_info.pop("seq", None)
                    apo_info.pop("coords", None)
                    if self.use_holo_if_apo_unavailable:
                        # Falling back to holo coordinates.
                        self.logger.warning(
                            "Failed to load apo structure for entity "
                            f"{ek} from {apo_info}: {e}. "
                            "Falling back to holo coordinates."
                        )
                        self.copy_chain_holo_coords_to_apo(chain, rng)
                    else:
                        self.logger.error(
                            "Failed to load apo structure for entity "
                            f"{ek} from {apo_info}: {e}."
                        )
                        continue
                # Store apo coordinates
                apo_coords_dict[eid] = apo_coords

            # Sanity check
            assert apo_coords.shape == (chain.num_residues, 37, 3), (
                f"Apo coordinates shape mismatch for entity {ek}: "
                f"expected ({chain.num_residues}, {37}, 3), got {apo_coords.shape}"
            )

            protein_atom_order = C.atom.protein_atom37_order

            # Apply random rotation/translation augmentation
            apo_coords = self.apply_random_augmentation(apo_coords, rng)

            # Insert apo coordinates into chain according to atom order
            # [L, Natom, 3] -> [Nallatoms, 3]
            src_res_indices: list[int] = []
            src_atom_indices: list[int] = []
            dst_atom_indices: list[int] = []
            atom_names: list[str] = chain.atom.name.tolist()  # pre-converted to list
            ccd_sequence = chain.get_ccd_sequence()
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
                        if an in protein_atom_order:
                            src_res_indices.append(res_i)
                            src_atom_indices.append(protein_atom_order[an])
                            dst_atom_indices.append(atom_i)

            chain.atom.apo_coords[dst_atom_indices] = apo_coords[
                src_res_indices, src_atom_indices
            ]

    def _insert_apo_coordinates_for_ligand_chains(
        self,
        struct: RefStructure,
        rng: np.random.Generator,
    ) -> None:
        """Use ETKDG conformers or CCD reference conformers as apo coordinates"""
        _ref_comp_cache: dict[str, Component] = {}
        _ref_comp_smi_cache: dict[str, Component] = {}
        for chain_i in range(struct.num_chains):
            chain = struct.chains[chain_i]
            if chain.ctype.is_polymer:
                continue  # polymer chains handled above
            chain_meta = struct.metadata.chains[chain_i]

            apo_coords = np.full_like(chain.atom.coords, np.nan)
            for res_i in range(chain.num_residues):
                residue_index = res_i + 1  # 1-based index
                ccd_name = str(chain.residue.name[res_i])

                # Load reference molecule from CCD
                if ccd_name.startswith("LIG"):
                    # This residue is from a smiles string, load smiles from metadata
                    smiles = chain_meta.smiles
                    assert smiles is not None, (
                        "Smiles string not found in metadata for LIG residue."
                    )
                    assert chain.num_residues == 1, (
                        "Residue with LIG found in chain with multiple residues."
                    )
                    if smiles in _ref_comp_smi_cache:
                        ref_comp = _ref_comp_smi_cache[smiles]
                    else:
                        # Use a shorter timeout (5.0s) for training,
                        # and longer timeout (30.0s) for inference.
                        timeout = 5 if self.conformer_mode == "train" else 30
                        ref_comp: Component = Component.from_smiles(
                            ccd_name, smiles, timeout=timeout, rng=rng
                        )
                        _ref_comp_smi_cache[smiles] = ref_comp
                else:
                    ref_comp = get_ref_comp(ccd_name, self.ccd, _ref_comp_cache)

                ref_atom_order: dict[str, int] = ref_comp.get_atom_index_map()
                ref_pos = ref_comp.get_conformer(self.conformer_mode, rng=rng)
                assert ref_pos is not None, "Auto mode always provides a conformer."

                # Map reference conformer to chain's atom order
                src_atom_indices: list[int] = []
                dst_atom_indices: list[int] = []
                for atom_i in chain.residue.iter_residue_atoms(residue_index):
                    an = chain.atom.name[atom_i]
                    if an in ref_atom_order:
                        ref_at_idx = ref_atom_order[an]
                        src_atom_indices.append(ref_at_idx)
                        dst_atom_indices.append(atom_i)
                apo_coords[dst_atom_indices] = ref_pos[src_atom_indices]

            if (
                self.ligand_perturbation is not None
                and rng.random() < self.prob_perturbation
            ):
                apo_coords = self.ligand_perturbation(apo_coords, chain, rng)

            # Apply random rotation augmentation
            apo_coords = self.apply_random_augmentation(apo_coords[None, ...], rng)[0]

            # Feed apo coordinates
            chain.atom.apo_coords[:, :] = apo_coords

    def copy_chain_holo_coords_to_apo(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ):
        """Get apo structure coordinates from label monomer structure.
        This is for protein monomer synthetic data, such as AFDB or ESM Atlas.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing label coordinates.
        perturb : bool
            Whether to apply perturbation to the copied coordinates.
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        # Cache atom order mapping
        protein_atom_order = C.atom.protein_atom37_order

        assert chain.ctype.is_protein, (
            "copy_chain_holo_coords_to_apo only supports protein chains."
        )
        apo_coords = np.full((chain.num_residues, 37, 3), np.nan, dtype=np.float32)
        atom_names: list[str] = chain.atom.name.tolist()  # pre-converted to list
        atom_coords = chain.atom.coords  # [Nallatoms, 3]
        res_indices: list[int] = []
        atom_indices: list[int] = []
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based residue index
            for atom_i in chain.residue.iter_residue_atoms(residue_index):
                an = atom_names[atom_i]
                # Assume all atoms are standard protein atoms
                a_i = protein_atom_order[an]
                res_indices.append(res_i)
                atom_indices.append(a_i)
        assert len(res_indices) == atom_coords.shape[0], (
            "Mismatch in number of atoms between apo and holo structures."
        )
        apo_coords[res_indices, atom_indices] = atom_coords

        # Apply perturbation
        if (
            self.protein_perturbation is not None
            and rng.random() < self.prob_perturbation
        ):
            sequence: str = chain.get_sequence()
            apo_coords = self.apply_perturbation(sequence, apo_coords, rng)

        # Apply random augmentation
        apo_coords = self.apply_random_augmentation(apo_coords, rng)

        # Feed apo coordinates
        chain.atom.apo_coords[:] = apo_coords[res_indices, atom_indices]

    def get_protein_apo_structure(
        self,
        ccd_sequence: list[str],
        apo_info: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Load apo structure from file for protein chains.

        Parameters
        ----------
        apo_info : dict
            Information about the apo structure file and residue indices.
            - input types:
                - case1: apo structure file path
                    - path: PathLike
                - case2: sequence and atom37 coordinates
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
        apo_coords : np.ndarray
            Apo structure coordinates of shape [L, 37, 3].
        """

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

        # Load apo structure
        if "seq" in apo_info and "coords" in apo_info:
            # Apo info includes pre-loaded sequence and coordinates.
            sequence: str = apo_info["seq"]
            apo_coords: np.ndarray = apo_info["coords"]
            assert apo_coords.shape == (len(sequence), 37, 3), (
                f"Apo coordinates shape mismatch: expected ({len(sequence)}, 37, 3), "
                f"got {apo_coords.shape}"
            )
        elif "path" in apo_info:
            # Load sequence and apo coordinates from structure file.
            path = apo_info["path"]
            sequence, apo_coords = read_protein_structure(path)
        else:
            raise ValueError(
                "Apo info must contain either 'seq' and 'coords', "
                "or 'path' to the structure file."
            )

        # Apply perturbation if enabled
        if (
            self.protein_perturbation is not None
            and rng.random() < self.prob_perturbation
        ):
            # get optional key for pre-computed perturbation with rieprody
            rieprody_key = apo_info.get("key", None)
            apo_coords = self.apply_perturbation(
                sequence, apo_coords, rng=rng, rieprody_key=rieprody_key
            )

        # Crop apo_coords based on residue_map
        length = len(ccd_sequence)
        if "residue_map" in apo_info:
            residue_map = apo_info["residue_map"]
            res_st, res_end, apo_st, apo_end = parse_residue_map(residue_map)
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
        return apo_coords

    def apply_perturbation(
        self,
        sequence: str,
        apo_coords: np.ndarray,
        rng: np.random.Generator,
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
        assert self.protein_perturbation is not None, (
            "Protein perturbation module not initialized."
        )
        apo_mask = np.isfinite(apo_coords).all(axis=-1)
        aug_coords = self.protein_perturbation(
            sequence, apo_coords, rng=rng, rieprody_key=rieprody_key
        )
        aug_coords[~apo_mask] = np.nan
        return aug_coords

    def apply_random_augmentation(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Augment coordinates with random rotation/translation.

        Parameters
        ----------
        coords : np.ndarray
            Structure coordinates of shape [L, Natom, 3].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        augmented_coords : np.ndarray
            Augmented structure coordinates of shape [L, Natom, 3].
        """
        assert coords.ndim == 3, "Apo coordinates must be of shape [L, Natom, 3]."
        # Flatten
        L, Natom = coords.shape[:2]
        mask = np.isfinite(coords).all(axis=-1)
        # Apply random rotation
        augmented_coords = center_random_augmentation(
            coords.reshape(L * Natom, 3),
            mask.reshape(L * Natom),
            augmentation=True,
            rng=rng,
        ).reshape(L, Natom, 3)
        augmented_coords[~mask] = np.nan
        return augmented_coords

    def find_best_residue_permutation(self, struct: RefStructure) -> None:
        """Find the best residue permutation for symmetry correction.
        Use intra-residue structure comparison to find the best permutation.
        """
        _ref_comp_cache: dict[str, Component] = {}

        for chain in struct.chains:
            ctype = chain.ctype

            if chain.is_ion:
                # Skip ions (single atom)
                continue

            ccd_sequence: list[str] = chain.get_ccd_sequence()
            all_atom_names: list[str] = chain.atom.name.tolist()

            for res_i in range(chain.num_residues):
                res_name: str = ccd_sequence[res_i]
                num_atoms: int = chain.residue.num_atoms[res_i]
                atom_st: int = chain.residue.atom_starts[res_i]
                atom_end: int = atom_st + num_atoms

                if chain.residue.is_standard[res_i]:
                    # Get ambiguous atom permutations for this standard residue
                    assert ctype.is_polymer, "Only polymer chains have standard residues."
                    perms = get_ambiguous_atoms_in_residue(res_name, extended=False)
                elif res_name.startswith("LIG"):
                    # For custom ligands, skip.
                    perms = None
                elif res_name in self.ccd:
                    ref_mol = get_ref_comp(res_name, self.ccd, _ref_comp_cache)
                    atom_names: list[str] = all_atom_names[atom_st:atom_end]
                    perms = get_molecule_symmetries(ref_mol, atom_names)
                else:
                    perms = None

                if perms is None or len(perms) <= 1:
                    # No ambiguous atoms for this residue
                    continue

                # Find the best permutation
                res_holo = chain.atom.coords[atom_st:atom_end]  # [num_atoms, 3]
                res_apo = chain.atom.apo_coords[atom_st:atom_end]  # [num_atoms, 3]
                res_holo_mask = np.isfinite(res_holo).all(-1)  # [num_atoms,]
                res_apo_mask = np.isfinite(res_apo).all(-1)  # [num_atoms,]

                if not res_holo_mask.any() or not res_apo_mask.any():
                    # No valid atoms to align
                    continue

                best_perm = None
                min_rmsd = float("inf")
                for perm in perms[:100]:
                    permuted_apo = res_apo[perm, :]
                    m = res_holo_mask & res_apo_mask[perm]
                    if not m.any():
                        continue
                    rmsd = compute_rmsd(
                        permuted_apo[m], res_holo[m], mask=None, align=True, no_svd=True
                    )
                    if rmsd < min_rmsd:
                        min_rmsd, best_perm = rmsd, perm

                if best_perm is not None:
                    # Apply best permutation
                    res_apo[:, :] = res_apo[best_perm, :]
