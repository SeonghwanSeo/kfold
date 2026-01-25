import dataclasses
import itertools
import logging
from collections import defaultdict
from functools import lru_cache

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.structure import read_protein_structure
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import compute_rmsd

from ._apo_perturbation import ApoPerturbation, ApoPerturbationConfig
from ._apo_prior import PolymerPriorConfig, PolymerPriorSampler

NUM_ATOMS_PER_RESIDUE: dict[C.ChainType, int] = {
    C.ChainType.PROTEIN: 37,
    C.ChainType.DNA: 29,
    C.ChainType.RNA: 29,
}


# === Helper functions === #
@lru_cache(32)
def get_ambiguous_atoms_in_residue(
    res_name: str,
) -> list[list[int]] | None:
    """Get the indices of ambiguous atoms for a given residue type."""
    res_name: C.ResidueName = C.ResidueName[res_name]
    if res_name not in C.atom.RESIDUE_AMBIGUOUS_ATOMS:
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
    ccd_id: str,
    mol_atom_names: list[str],
    ref_mol: Component,
) -> list[list[int]] | None:
    """Get molecule's symmetries from ccd."""
    symmetries = ref_mol.symmetries
    if symmetries is None or len(symmetries) <= 1:
        # No symmetries
        return None

    if len(mol_atom_names) == ref_mol.num_atoms:
        # All atoms are present, no need to filter
        return list(symmetries)

    # Some atoms are missing (drop_leaving_atoms=True; e.g., covalent ligands)
    valid_atoms: set[str] = set(mol_atom_names)

    name_to_index = ref_mol.get_atom_index_map()
    mol_to_ref_i_map: dict[int, int] = {
        name_to_index[name]: i for i, name in enumerate(mol_atom_names)
    }

    all_perms: list[list[int]] = []  # identity
    for perm in symmetries:
        # Example perm for 4-atom molecules: [0, 2, 1, 3] (swapping atom 1 and 2)
        sym_dict: dict[int, int] = {}
        for i, j in enumerate(perm):
            a_i, a_j = ref_mol.names[i], ref_mol.names[j]
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


def get_valid_atom_mask(ctype: C.ChainType, ccd_sequence: list[str]) -> np.ndarray:
    """Generate empty coordinates for a given chain type and sequence."""
    assert ctype.is_polymer, "Only polymer chains are supported."
    if ctype.is_protein:
        num_atoms = 37
        atom_order = C.atom.protein_atom37_order
    else:
        num_atoms = 29
        atom_order = C.atom.nucleic_acid_atom29_order
    L = len(ccd_sequence)
    mask = np.zeros((L, num_atoms), dtype=bool)
    for i, code in enumerate(ccd_sequence):
        res_name: C.ResidueName = C.residue.get_residue_name_with_unk(code, ctype)
        atom_names = C.atom.RESIDUE_ATOMS[res_name]
        for at in atom_names:
            at_idx = atom_order[at]
            mask[i, at_idx] = True
    return mask


def get_zero_coordinates(ctype: C.ChainType, ccd_sequence: list[str]) -> np.ndarray:
    """Generate zero coordinates for a given chain type and sequence."""
    L = len(ccd_sequence)
    Natom = NUM_ATOMS_PER_RESIDUE[ctype]
    coords = np.full((L, Natom, 3), np.nan, dtype=np.float32)
    mask = get_valid_atom_mask(ctype, ccd_sequence)
    coords[mask] = 0.0
    return coords


@dataclasses.dataclass(kw_only=True)
class ApoInitializerConfig:
    """Configuration for ApoInitializer.

    Attributes
    ----------
    use_perturbation : bool
        Whether to apply perturbation to protein apo structures.
    use_random_augmentation : bool
        Whether to apply random rotation/translation augmentation
        to apo structures.
    use_ot_permutation : bool
        Whether to apply optimal transport-based permutation
    translation_scale : float
        Scale of random translation augmentation (in Angstrom).
    prob_perturbation : float
        Probability of applying perturbation to apo structures.
    prob_replace_to_holo : float
        Probability of replacing apo structure with holo structure.
        The apo perturbation is skipped if replaced. This is motivated
        by the fact that most holo structures is one of the apo states.
        NOTE: This should be used for training only.
    apo_perturbation : ApoPerturbationConfig | None
        Configuration for protein apo perturbation.
    prior_sampler : PolymerPriorConfig
        Configuration for polymer prior sampler.
    training : bool
        Whether in training mode.
        NOTE: Recommended to set True for both training and validation datasets
        to disable ETKDG generation for efficiency.
    fill_missing_atom: bool
        Whether to fill missing atoms to neighboring atoms in apo structures
        for better interpolation.
    """

    use_perturbation: bool = False
    use_random_augmentation: bool = True
    use_ot_permutation: bool = False
    translation_scale: float = 100.0  # Angstrom
    prob_perturbation: float = 1.0
    prob_replace_to_holo: float = 0.0
    apo_perturbation: ApoPerturbationConfig | None = dataclasses.field(
        default_factory=ApoPerturbationConfig
    )
    prior_sampler: PolymerPriorConfig = dataclasses.field(
        default_factory=PolymerPriorConfig
    )
    training: bool = False
    fill_missing_atom: bool = False


class ApoInitializer:
    """Class to populate and augment apo structures."""

    def __init__(
        self,
        config: ApoInitializerConfig,
        ccd: CCD | None = None,
    ):
        self.config: ApoInitializerConfig = config
        self.use_perturbation: bool = config.use_perturbation
        self.use_random_augmentation: bool = config.use_random_augmentation
        self.use_ot_permutation: bool = config.use_ot_permutation
        self.translation_scale: float = config.translation_scale
        self.fill_missing_atom: bool = config.fill_missing_atom

        self.prob_perturbation: float = config.prob_perturbation
        self.prob_replace_to_holo: float = config.prob_replace_to_holo
        self.ccd: CCD = ccd

        if self.prob_replace_to_holo > 0.0:
            raise NotImplementedError(
                "ApoInitializer with prob_replace_to_holo > 0.0 is not implemented yet."
            )

        # Apo perturbation module
        if self.use_perturbation:
            assert config.apo_perturbation is not None, (
                "ApoPerturbationConfig must be provided when use_perturbation is True."
            )
            self.apo_perturbation: ApoPerturbation = ApoPerturbation(
                config.apo_perturbation
            )

        # Polymer prior sampler module
        self.prior_sampler: PolymerPriorSampler = PolymerPriorSampler(
            config.prior_sampler
        )

        # Training mode
        self.training: bool = config.training
        # During training, disable ETKDG generation for efficiency,
        # i.e., only the cached ETKDG and CCD conformers (ideal, mode) are used.
        self.conformer_mode: str = "train" if self.training else "auto"

        # Logger
        self.logger = logging.getLogger("ApoInitializer")

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
            - name: str
                e.g., "AF-P012345-F1-model_v1"
            - path: Path
                e.g., "AF-P012345-F1-model_v1.cif"
            - residue_map: str
                e.g., "11:100->66:155"
            - source: str
                e.g., "AFDB", "PDB"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        return self.populate_apo_structure(struct, lookup, rng)

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
            - name: str
                e.g., "AF-P012345-F1-model_v1"
            - path: Path
                e.g., "AF-P012345-F1-model_v1.cif"
            - residue_map: str
                e.g., "11:100->66:155"
            - source: str
                e.g., "AFDB", "PDB"
        rng : np.random.Generator
            Random number generator for stochastic operations.
        """
        rng = rng or np.random.default_rng()

        # Insert apo coordinates
        self.insert_apo_coordinates(struct, lookup, rng)

        # If symmetry correction is enabled, align apo to holo
        if self.use_ot_permutation:
            self.match_optimal_transport_permutation(struct, rng)

        # Fill missing atoms for better interpolation
        if self.config.fill_missing_atom:
            self.fill_missing_atoms_to_neighbors(struct)

    def insert_apo_coordinates(
        self,
        struct: RefStructure,
        lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Insert apo structure coordinates for each chain in the structure."""
        # === 1. Insert apo coordinates for polymer chains === #
        # cache apo coordinates per entity to avoid redundant loading/sampling
        apo_coords_dict: dict[int, np.ndarray] = {}
        for i in range(struct.num_chains):
            chain = struct.chains[i]
            if not chain.ctype.is_polymer:
                continue  # non-polymer chains handled later

            # Determine number of atoms and atom order
            if chain.ctype.is_protein:
                Natom = 37
                atom_order = C.atom.protein_atom37_order
            else:
                Natom = 29
                atom_order = C.atom.nucleic_acid_atom29_order

            entity_id = chain.entity_id
            if entity_id not in apo_coords_dict:
                ctype = chain.ctype
                ccd_sequence = chain.get_ccd_sequence()

                if entity_id in lookup:
                    # Load apo structure from file
                    assert ctype.is_protein, "Only protein chains have apo structures."
                    try:
                        apo_coords = self.get_protein_apo_structure(
                            ccd_sequence, lookup[entity_id], rng
                        )
                    except Exception as e:
                        # NOTE: There are some errors in rcsb-to-uniprot mapping file.
                        # For robustness, we fall back to prior sampling if loading fails.
                        self.logger.error(
                            "Failed to load apo structure for entity "
                            f"{entity_id}: {e}. Sampling from prior instead."
                        )
                        apo_coords = self.sample_apo_structure_from_prior(
                            ccd_sequence, ctype, rng
                        )
                else:
                    # No apo structure available, sample from prior
                    apo_coords = self.sample_apo_structure_from_prior(
                        ccd_sequence, ctype, rng
                    )
                # Store apo coordinates
                apo_coords_dict[entity_id] = apo_coords
            else:
                # Reuse cached apo coordinates
                apo_coords = apo_coords_dict[entity_id]

            # Sanity check
            assert apo_coords.shape == (chain.num_residues, Natom, 3), (
                f"Apo coordinates shape mismatch for entity {entity_id}: "
                f"expected ({chain.num_residues}, {Natom}, 3), got {apo_coords.shape}"
            )

            if self.use_random_augmentation:
                # Apply random rotation/translation augmentation
                apo_coords = self.apply_random_augmentation(apo_coords, rng)

            # Insert apo coordinates into chain according to atom order
            # [L, Natom, 3] -> [Nallatoms, 3]
            src_res_indices: list[int] = []
            src_atom_indices: list[int] = []
            dst_atom_indices: list[int] = []
            atom_names: list[str] = chain.atom.name.tolist()  # pre-converted to list
            for res_i in range(chain.num_residues):
                residue_index = res_i + 1  # 1-based residue index
                for atom_i in chain.residue.iter_residue_atoms(residue_index):
                    an = atom_names[atom_i]
                    if an in atom_order:
                        src_res_indices.append(res_i)
                        src_atom_indices.append(atom_order[an])
                        dst_atom_indices.append(atom_i)

            chain.atom.apo_coords[dst_atom_indices] = apo_coords[
                src_res_indices, src_atom_indices
            ]

        # === 2. Insert apo coordinates for non-polymer chains === #
        # Use CCD reference conformers or ETKDG-generated.
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
                    assert chain_meta.smiles is not None, (
                        "Smiles string not found in metadata for LIG residue."
                    )
                    assert chain.num_residues == 1, (
                        "Residue with LIG found in chain with multiple residues."
                    )
                    ref_mol = Component.from_smiles(
                        ccd_name, chain_meta.smiles, num_confs=1
                    )
                else:
                    assert ccd_name in self.ccd, (
                        f"Residue name {ccd_name} not found in CCD."
                    )
                    ref_mol = self.ccd[ccd_name]

                ref_atom_order: dict[str, int] = ref_mol.get_atom_index_map()
                ref_pos = ref_mol.get_conformer(self.conformer_mode, rng=rng)
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

            if self.use_random_augmentation:
                # Apply random rotation augmentation
                apo_coords = self.apply_random_augmentation(apo_coords[None, ...], rng)[0]

            # Feed apo coordinates
            chain.atom.apo_coords[:, :] = apo_coords

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
            - name: str
                e.g., "AF-P012345-F1-model_v1"
            - path: Path
                e.g., "AF-P012345-F1-model_v1.cif.gz"
            - residue_map: str
                e.g., "11:100->66:155"
            - source: str
                e.g., "AF2", "PDB"
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
            st, end = map(int, res_range.split(":"))
            apo_st, apo_end = map(int, apo_range.split(":"))
            if (end - st) != (apo_end - apo_st):
                raise ValueError(f"Residue range length mismatch: {residue_map}")
            # Convert to 0-based indexing
            # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
            return st - 1, end, apo_st - 1, apo_end

        path = apo_info["path"]
        residue_map = apo_info["residue_map"]
        lmdb_key = apo_info["rieprody_key"]

        # Load apo structure
        sequence, apo_coords = read_protein_structure(path)

        # Apply perturbation if enabled
        if self.use_perturbation and rng.random() < self.prob_perturbation:
            apo_mask = np.isfinite(apo_coords).all(axis=-1)
            apo_coords = self.apply_perturbation(
                sequence, apo_coords, apo_mask, rng, key=lmdb_key
            )

        # Crop apo_coords based on residue_map
        length = len(ccd_sequence)
        st, end, apo_st, apo_end = parse_residue_map(residue_map)

        if st == 0 and end == length:
            # Use full apo_coords
            return apo_coords[apo_st:apo_end].copy()
        else:
            padded_apo_coords = np.full(
                (length, apo_coords.shape[1], 3), np.nan, dtype=np.float32
            )
            padded_apo_coords[st:end] = apo_coords[apo_st:apo_end]
            return padded_apo_coords

    def sample_apo_structure_from_prior(
        self,
        ccd_sequence: list[str],
        ctype: C.ChainType,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample apo structure from prior when apo structure is not available.

        Parameters
        ----------
        ccd_sequence : list[str]
            CCD sequence of the chain.
        ctype : C.ChainType
            Chain type of the chain.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        apo_coords : np.ndarray
            Apo structure coordinates of shape [L, Natom, 3],
            where Natom is 37 for protein and 29 for nucleic acid.
        """
        assert ctype.is_polymer, "Only polymer chains are supported."
        mask = get_valid_atom_mask(ctype, ccd_sequence)
        apo_coords = self.prior_sampler.sample(mask, rng)
        return apo_coords

    def apply_perturbation(
        self,
        sequence: str,
        apo_coords: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        key: str | None = None,
    ) -> np.ndarray:
        """Augment apo structure coordinates with perturbation.

        Parameters
        ----------
        sequence : str
            Amino acid sequence of the protein.
        apo_coords : np.ndarray
            Apo structure coordinates of shape [L, Natom, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [L, Natom].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        key : str | None
            Optional key for using pre-computed perturbation metrics.

        Returns
        -------
        augmented_coords : np.ndarray
            Augmented structure coordinates of shape [L, Natom, 3].
        """
        return self.apo_perturbation.run(sequence, apo_coords, mask, rng=rng, key=key)

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
            s_trans=-0.0,
            rng=rng,
        ).reshape(L, Natom, 3)

        # Apply random translation
        # Random unit vector
        rand_dir = rng.normal(size=(3,))
        rand_dir /= np.linalg.norm(rand_dir)
        translation = rand_dir * self.translation_scale
        augmented_coords += translation

        augmented_coords[~mask] = np.nan
        return augmented_coords

    def match_optimal_transport_permutation(
        self,
        struct: RefStructure,
        rng: np.random.Generator,
    ):
        """Align apo to holo to minimize transport cost (RMSD).

        This method solves the discrete optimal transport problem over the
        structure's symmetry group (chain permutations and residue flips).
        It ensures that the flow matching target (holo) is aligned to the
        source (apo) with the minimal displacement, constructing the
        optimal straight-line trajectory.

        Parameters
        ----------
        struct : RefStructure
            Reference structure. Apo coords will be permuted in-place.
        rng : np.random.Generator
            Random number generator for stochastic sampling.
        """
        # Validate that there is at least one resolved holo atoms
        for chain in struct.chains:
            if np.isfinite(chain.atom.coords).all(-1).any():
                break
        else:
            raise ValueError("No resolved holo atoms found for ot permutation.")

        # Skip permutation if there is no resolved apo atoms
        for chain in struct.chains:
            if np.isfinite(chain.atom.apo_coords).all(-1).any():
                break
        else:
            # No resolved apo atoms found; return without permutation
            return

        # First, chain permutation
        # RNG state is used for sampling permutations when too many exist
        try:
            self.find_best_chain_permutation(struct, max_permutations=2_000, rng=rng)
        except Exception as e:
            self.logger.error(f"Failed to find best chain permutation: {e}.")

        # Second, residue-level permutation (e.g., flipping)
        try:
            self.find_best_residue_permutation(struct)
        except Exception as e:
            self.logger.error(f"Failed to find best residue permutation: {e}.")

    def find_best_chain_permutation(
        self,
        struct: RefStructure,
        max_permutations: int,
        rng: np.random.Generator,
    ) -> None:
        # === 1. Prepare coordinates === #
        asym_id_to_entity_id: dict[int, int] = {
            chain.asym_id: chain.entity_id for chain in struct.chains
        }
        entity_ids: list[int] = sorted(set(chain.entity_id for chain in struct.chains))
        entity_apo_dict: dict[int, list[np.ndarray]] = defaultdict(list)
        entity_ctypes: dict[int, C.ChainType] = {}
        entity_sizes: dict[int, int] = {}
        for chain in struct.chains:
            entity_apo_dict[chain.entity_id].append(chain.atom.apo_coords.copy())
            entity_ctypes[chain.entity_id] = chain.ctype
            entity_sizes[chain.entity_id] = chain.num_residues

        # Sort entities by size (largest first)
        entity_ids.sort(key=lambda x: entity_sizes[x], reverse=True)

        # Remove all covalent ligands from permutation candidates
        for conn in struct.connections:
            for asym_id in conn.asym_id:
                eid = asym_id_to_entity_id[asym_id]
                if entity_ctypes[eid].is_ligand:
                    if eid in entity_ids:
                        entity_ids.remove(eid)

        for eid in list(entity_ids):
            apo_coords_list = entity_apo_dict[eid]
            # Check the chain apo coordinates are provided
            if any(np.isnan(coords).all() for coords in apo_coords_list):
                # If any chain has no apo coordinates, remove from permutation
                entity_ids.remove(eid)

        if len(entity_ids) == 0:
            # No entities to permute
            return

        if all(len(entity_apo_dict[eid]) == 1 for eid in entity_ids):
            # Only one chain per entity, no permutation needed
            return

        perm_chains = [chain for chain in struct.chains if chain.entity_id in entity_ids]

        # === 2. Anchor Selection for Alignment === #
        entity_anchors: dict[int, np.ndarray] = {}
        entity_anchor_coords: dict[int, list[np.ndarray]] = {}

        for chain in perm_chains:
            eid = chain.entity_id
            if eid in entity_anchors:
                continue
            if chain.ctype.is_protein:
                idx = np.where(chain.atom.name == "CA")[0]
                num_anchors = 10
            elif chain.ctype.is_nucleic_acid:
                idx = np.where(chain.atom.name == "C1'")[0]
                num_anchors = 10
            else:
                idx = chain.atom.is_apo_resolved.nonzero()[0]
                num_anchors = 2

            if len(idx) == 0:
                # Fallback to any resolved atoms
                idx = chain.atom.is_apo_resolved.nonzero()[0]
                num_anchors = 2

            # Take evenly spaced anchors up to num_anchors
            stride = max(1, len(idx) // num_anchors)
            idx = idx[::stride]
            entity_anchors[eid] = idx
            entity_anchor_coords[eid] = [coords[idx] for coords in entity_apo_dict[eid]]

        # === 3. Generate candidate permutations === #
        entity_to_perms = {}
        for eid in entity_ids:
            num_chains = len(entity_apo_dict[eid])
            # Limit per-entity permutations to avoid memory explosion
            perms = list(
                itertools.islice(
                    itertools.permutations(range(num_chains)), max_permutations * 5
                )
            )
            entity_to_perms[eid] = perms

        final_permutations: list[dict[int, list[int]]] = [
            {eid: perm for eid, perm in zip(entity_ids, perms, strict=True)}
            for perms in itertools.islice(
                itertools.product(*(entity_to_perms[eid] for eid in entity_ids)),
                max_permutations * 5,
            )
        ]
        if len(final_permutations) > max_permutations:
            rng.shuffle(final_permutations)
            final_permutations = final_permutations[:max_permutations]

        # === 4. Prepare label centers and masks === #
        label_centers = np.concatenate(
            [chain.atom.coords[entity_anchors[chain.entity_id]] for chain in perm_chains],
            axis=0,
        )
        label_mask = np.isfinite(label_centers).all(-1)
        entity_order = [chain.entity_id for chain in perm_chains]

        # === 5. Evaluate permutations === #
        best_perm = None
        best_rmsd = float("inf")
        apo_centers = np.empty_like(label_centers)
        for perm in final_permutations:
            # Use a simpler way to track which index to take for each entity
            st = 0
            _perm = {eid: list(perm[eid]) for eid in entity_ids}
            for eid in entity_order:
                swap_idx = _perm[eid].pop(0)
                chain_coords = entity_anchor_coords[eid][swap_idx]
                apo_centers[st : st + len(chain_coords)] = chain_coords
                st += len(chain_coords)
            apo_mask = np.isfinite(apo_centers).all(-1)
            m = label_mask & apo_mask

            rmsd = compute_rmsd(
                apo_centers[m], label_centers[m], mask=None, align=True, no_svd=False
            )
            if rmsd < best_rmsd:
                best_perm, best_rmsd = perm, rmsd

        # === 6. Apply permutations === #
        counts = defaultdict(int)
        if best_perm is not None:
            # Reorder apo coordinates according to best permutation
            for chain in perm_chains:
                eid = chain.entity_id
                if eid not in entity_ids:
                    continue
                orig_idx = counts[eid]  # index in the entity
                perm_idx = best_perm[eid][orig_idx]
                if orig_idx != perm_idx:
                    # Apply permutation
                    chain.atom.apo_coords[:] = entity_apo_dict[eid][perm_idx]
                counts[eid] += 1
        del counts

    def find_best_residue_permutation(self, struct: RefStructure) -> None:
        """Find the best residue permutation for symmetry correction.
        Use intra-residue structure comparison to find the best permutation.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing holo coordinates.
        """
        component_cache: dict[str, Component] = {}
        for chain in struct.chains:
            ctype = chain.ctype

            if chain.is_ion:
                # Skip ions (single atom)
                continue

            for res_i in range(chain.num_residues):
                res_name: str = chain.residue.name[res_i].item()
                num_atoms: int = chain.residue.num_atoms[res_i]
                atom_st: int = chain.residue.atom_starts[res_i]
                atom_end: int = atom_st + num_atoms

                if ctype.is_polymer and chain.residue.is_standard[res_i]:
                    # Get ambiguous atom permutations for this standard residue
                    perms = get_ambiguous_atoms_in_residue(res_name)
                elif res_name in self.ccd:
                    if res_name not in component_cache:
                        # Load reference molecule from CCD
                        ref_mol = self.ccd[res_name]
                        # Cache the component
                        component_cache[res_name] = ref_mol
                    else:
                        ref_mol = component_cache[res_name]
                    atom_names: list[str] = chain.atom.name[atom_st:atom_end].tolist()
                    perms = get_molecule_symmetries(res_name, atom_names, ref_mol)
                else:
                    perms = None

                if perms is None:
                    # No ambiguous atoms for this residue
                    continue

                # Find the best permutation
                res_holo = chain.atom.coords[atom_st:atom_end]  # [num_atoms, 3]
                res_apo = chain.atom.apo_coords[atom_st:atom_end]  # [num_atoms, 3]
                res_holo_mask = np.isfinite(res_holo).all(-1)  # [num_atoms,]
                res_apo_mask = np.isfinite(res_apo).all(-1)  # [num_atoms,]

                best_perm = None
                min_rmsd = float("inf")
                for perm in perms:
                    permuted_apo_mask = res_apo_mask[perm]
                    align_mask = res_holo_mask & permuted_apo_mask
                    permuted_apo = res_apo[perm, :]
                    rmsd = compute_rmsd(
                        permuted_apo, res_holo, align_mask, align=True, no_svd=True
                    )
                    if rmsd < min_rmsd:
                        min_rmsd, best_perm = rmsd, perm

                if best_perm is not None:
                    # Apply best permutation
                    res_apo[:, :] = res_apo[best_perm, :]
                else:
                    pass

    def fill_missing_atoms_to_neighbors(self, struct: RefStructure) -> None:
        """Fill missing atoms to neighboring atoms in apo structures
        for better interpolation.

        Strategy:
        - Intra-residue: Fill missing atoms with the residue center.
        - Inter-residue: Fill unresolved residues with the center of the
          nearest resolved residue (by sequence index).

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates.
        """
        for chain in struct.chains:
            apo_coords: np.ndarray = chain.atom.apo_coords

            # Mask of valid atoms: [N_atoms]
            atom_mask: np.ndarray = chain.atom.is_apo_resolved

            # Check if any atoms are missing
            if atom_mask.all():
                continue

            # Check if all atoms are missing
            if not atom_mask.any():
                continue  # Cannot fill anything

            num_residues = chain.num_residues
            residue_centers = np.full((num_residues, 3), np.nan, dtype=np.float32)
            is_residue_resolved = np.zeros(num_residues, dtype=bool)

            # --- Step 1: Compute Centers & Fill Intra-residue gaps ---
            # Note: Keeping loop due to ragged atom_starts/ends
            for res_i in range(num_residues):
                atom_st = chain.residue.atom_starts[res_i]
                atom_end = atom_st + chain.residue.num_atoms[res_i].item()

                # Slice views (modifications affect apo_coords)
                res_coords_view = apo_coords[atom_st:atom_end]
                res_mask_view = atom_mask[atom_st:atom_end]

                if res_mask_view.any():
                    is_residue_resolved[res_i] = True
                    # Compute center using only valid atoms
                    center = np.mean(res_coords_view[res_mask_view], axis=0)
                    residue_centers[res_i] = center

                    # Fill missing atoms within this resolved residue
                    if not res_mask_view.all():
                        res_coords_view[~res_mask_view] = center

            # --- Step 2: Fill Unresolved Residues (Vectorized) ---
            resolved_indices = np.where(is_residue_resolved)[0]

            # Safety check: If no residues are resolved, we cannot fill anything.
            if len(resolved_indices) == 0:
                # Optional: logging warning here
                continue

            missing_res_indices = np.where(~is_residue_resolved)[0]

            if len(missing_res_indices) > 0:
                # Find nearest resolved index for every missing index
                # np.searchsorted finds insertion points to keep order
                idx_insertion = np.searchsorted(resolved_indices, missing_res_indices)

                # Clamp indices to valid range for neighbor checking
                idx_left = np.clip(idx_insertion - 1, 0, len(resolved_indices) - 1)
                idx_right = np.clip(idx_insertion, 0, len(resolved_indices) - 1)

                # Calculate distances to left and right neighbors
                dist_left = np.abs(missing_res_indices - resolved_indices[idx_left])
                dist_right = np.abs(missing_res_indices - resolved_indices[idx_right])

                # Choose the closer neighbor
                # use_right is a boolean mask
                use_right = dist_right < dist_left
                closest_indices_map = np.where(use_right, idx_right, idx_left)

                # Get the actual residue indices
                closest_res_indices = resolved_indices[closest_indices_map]

                # Gather centers for all missing residues at once
                fill_centers = residue_centers[closest_res_indices]

                # Apply filling
                for i, res_i in enumerate(missing_res_indices):
                    atom_st = chain.residue.atom_starts[res_i]
                    atom_end = atom_st + chain.residue.num_atoms[res_i].item()
                    apo_coords[atom_st:atom_end] = fill_centers[i]
