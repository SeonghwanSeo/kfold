import itertools
import pickle
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch
from rieprody.proteins.protein_perturbation import ProteinPerturbationModule

import kfold.constants as C
from kfold.data.structure import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import (
    compute_rmsd,
    rigid_align,
    weighted_rigid_align,
)

ResUID = tuple[int, int]  # (asym_id, residue_index)


@lru_cache(100)
def get_ambiguous_atoms_in_residue(res_name: C.ResidueName) -> list[list[int]]:
    """Get the indices of ambiguous atoms for a given residue type.

    Parameters
    ----------
    res_name : C.ResidueName
        The residue type.

    Returns
    -------
    perms: list[list[int]]
        A list of atom permutations
    """
    if res_name not in C.atom.RESIDUE_AMBIGUOUS_ATOMS:
        # If there is no ambiguous atoms, return empty list
        return []
    residue_atoms = C.atom.RESIDUE_ATOMS[res_name]

    src_atoms, dst_atoms = C.atom.RESIDUE_AMBIGUOUS_ATOMS_EXTENDED[res_name]
    src_indices = [residue_atoms.index(atom) for atom in src_atoms]
    dst_indices = [residue_atoms.index(atom) for atom in dst_atoms]
    # Ensure no overlapping indices
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
    ccd_symmetry_dict: dict,
) -> list[list[int]]:
    """Get molecule's symmetries from ccd"""
    ccd_syms, ccd_atom_names = ccd_symmetry_dict[ccd_id]
    atom_id_in_ccd: dict[int, int] = {
        ccd_atom_names.index(name): i for i, name in enumerate(mol_atom_names)
    }
    valid_atoms: set[int] = set(atom_id_in_ccd.keys())

    all_syms: list[list[int]] = []
    # Get symmetries
    for sym in ccd_syms:
        # Example sym for 4-atom molecules: [0, 2, 1, 3] (swapping atom 1 and 2)
        sym_dict: dict[int, int] = {}
        for i, j in enumerate(sym):
            if i not in valid_atoms:
                # atom i is not in the molecule
                continue
            if j in valid_atoms:
                # both atoms are in the molecule
                i_true = atom_id_in_ccd[i]
                j_true = atom_id_in_ccd[j]
                sym_dict[i_true] = j_true
            else:
                # atom j is not in the molecule
                # skip this symmetry
                break
        else:
            # Completed without break
            # NOTE: This is bijective mapping within valid atoms (see above)
            all_syms.append([sym_dict[i] for i in range(len(valid_atoms))])
    return all_syms


class ApoPerturbation:
    """Class to handle apo structure perturbation."""

    def __init__(
        self,
        use_perturbation: bool = False,
        use_random_rotation: bool = False,
        use_symmetry_correction: bool = False,
        prob_perturbation: float = 0.5,
        prob_replace_to_holo: float = 0.0,
        mask_nucleic_acids: bool = False,
        ccd_symmetry_dict: dict | None = None,
        seed: int | None = 42,
        metric_lmdb_path: Path | str | None = None,
        metric_comp: dict | None = None,
        random_walk: dict | None = None,
    ):
        """Initialize ApoPerturbation.
        Parameters
        ----------
        use_perturbation : bool, optional
            Whether to apply perturbation to apo structures.
        use_random_rotation : bool, optional
            Whether to apply random rotation to apo structures.
        use_symmetry_correction : bool, optional
            Whether to correct for symmetry to holo structures.
            NOTE: This is used during training only.
        prob_perturbation : float, optional
            Probability of applying perturbation to apo structures.
        prob_replace_to_holo : float, optional
            Probability of replacing apo structure with holo structure. The apo
            perturbation is skipped if replaced. This is motivated by the fact that
            most holo structures is one of the apo states.
            NOTE: This is used during training only.
        mask_nucleic_acids : bool, optional
            Whether to mask nucleic acid chains during perturbation.
            TODO: (SeonghwanSeo) Remove this argument after DNA/RNA apo coordinates
            is prepared.
        ccd_symmetry_dict: dict | None
            Dictionary containing symmetry information for CCD entries.
        seed : int | None, optional
            Random seed for stochastic operations.
        metric_lmdb_path : Path | str | None, optional
            Path to LMDB file containing pre-computed metric information.
        metric_comp : dict | None, optional
            Configuration for metric computation. Should contain:
            - apo_internal_coord_metric_calculation_device
            - apo_internal_coord_metric_calculation_precision
            - apo_internal_coord_metric_save_precision
        random_walk : dict | None, optional
            Configuration for random walk. Should contain:
            - apo_internal_coord_random_walk_device
            - apo_internal_coord_random_walk_precision
            - consider_side_chain_in_metric
            - fixed_bb_angles
            - total_time
            - num_steps
            - metric_calculation_period
            - metric (optional, default: "lagrangian")
            - with_christoffel_term (optional, default: False)
            - use_precomputed_metric (optional, default: True)
            - kabsch_aligned_traj (optional, default: True)
        """
        self.use_perturbation: bool = use_perturbation
        self.use_random_rotation: bool = use_random_rotation
        self.use_symmetry_correction: bool = use_symmetry_correction
        self.prob_perturbation: float = prob_perturbation
        self.prob_replace_to_holo: float = prob_replace_to_holo
        self.mask_nucleic_acids: bool = mask_nucleic_acids
        self.seed: int | None = seed

        if self.use_symmetry_correction:
            assert ccd_symmetry_dict is not None, (
                "CCD symmetry dictionary must be provided for symmetry correction."
            )
            self.ccd_symmetry_dict: dict = ccd_symmetry_dict

        # Apo perturbation setup
        self.metric_lmdb_path: Path | None = None
        if metric_lmdb_path is not None:
            self.metric_lmdb_path = Path(metric_lmdb_path)
            if not self.metric_lmdb_path.exists():
                raise FileNotFoundError(
                    f"Metric LMDB path does not exist: {self.metric_lmdb_path}"
                )

        self.metric_comp: dict | None = metric_comp
        self.random_walk: dict | None = random_walk

        # Initialize RieProDy module if perturbation is enabled
        self._rieprody_module: ProteinPerturbationModule | None = None
        if self.use_perturbation and metric_comp is not None and random_walk is not None:
            # Create a simple config-like object from dict
            class SimpleConfig:
                def __init__(self, config_dict: dict):
                    for key, value in config_dict.items():
                        if isinstance(value, dict):
                            setattr(self, key, SimpleConfig(value))
                        else:
                            setattr(self, key, value)

            # Combine metric_comp and random_walk into a config structure
            # Note: dataset fields are only used in preprocess() which we don't use
            # They are set to minimal values to avoid AttributeError in __init__
            from pathlib import Path as PathLib

            config_dict = {
                "metric_comp": metric_comp,
                "random_walk": random_walk,
                "dataset": {
                    "data_dir": PathLib(
                        "."
                    ),  # Not used in read_computed_data/solve_riemannian_brownian_motion
                    "cache_path": PathLib(
                        "."
                    ),  # Not used in read_computed_data/solve_riemannian_brownian_motion
                    "preprocess_num_workers": 1,  # Not used in preprocess() here
                },
            }
            config = SimpleConfig(config_dict)
            self._rieprody_module = ProteinPerturbationModule(config=config, records=None)

        # Lazy initialization of LMDB
        self._lmdb_env: lmdb.Environment | None = None

    @property
    def lmdb_env(self) -> lmdb.Environment | None:
        """Lazy initialization of LMDB environment."""
        if self._lmdb_env is None and self.metric_lmdb_path is not None:
            self._lmdb_env = lmdb.open(
                str(self.metric_lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_env

    def _load_metric_from_lmdb(self, record_id: str, entity_id: int) -> dict | None:
        """Load pre-computed metric data from LMDB.

        Parameters
        ----------
        record_id : str
            Record ID (PDB ID).
        entity_id : int
            Entity ID (1-based).

        Returns
        -------
        dict | None
            Metric data dictionary or None if not found.
        """
        if self.lmdb_env is None:
            return None

        # Convert entity_id (1-based) to entity_index (0-based)
        entity_index = entity_id - 1
        key = f"{record_id}-{entity_index}".encode()

        try:
            with self.lmdb_env.begin(write=False) as txn:
                value_bytes = txn.get(key)
                if value_bytes is None:
                    return None

                data = pickle.loads(value_bytes)
                return data
        except Exception as e:
            # Log error but don't raise - return None to skip perturbation
            import warnings

            warnings.warn(
                f"Failed to load metric data for {record_id}-{entity_index}: {e}",
                UserWarning,
            )
            return None

    def run(
        self, struct: TokenizedStructure, rng: np.random.Generator | None = None
    ) -> TokenizedStructure:
        """Sample the apo structure and apply augmentation if needed.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing apo coordinates and masks.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        augmented_struct : TokenizedStructure
            Tokenized structure with augmented apo coordinates.
            NOTE: if there is multiple apo structures, one of them is sampled randomly
            and augmented.
        """
        rng = rng or np.random.default_rng(self.seed)

        # Sample an apo structure for each chain and apply augmentation if needed
        apo_coords, apo_mask = self.sample_and_augment_apo_structure(struct, rng)

        # If symmetry correction is enabled, align apo to holo
        if self.use_symmetry_correction:
            apo_coords, apo_mask = self.align_apo_to_holo(
                struct, apo_coords, apo_mask, rng
            )

        # Create new atom structure
        num_tokens = struct.num_tokens
        atom_struct = struct.atom.copy_with(
            apo_coords=apo_coords.reshape(num_tokens, 24, 1, 3),
            apo_mask=apo_mask.reshape(num_tokens, 24, 1),
        )
        return struct.copy_with(atom=atom_struct)

    def sample_and_augment_apo_structure(
        self, struct: TokenizedStructure, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the apo structure and apply augmentation if needed.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing apo coordinates and masks.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        apo_coords : np.ndarray
            Sampled and augmented apo structure coordinates of shape [Ntoken, 24, 3].
        apo_mask : np.ndarray
            Sampled and augmented apo structure masks of shape [Ntoken, 24].
        """
        # Get coordinates and masks for all apo structures
        all_apo_coords = struct.atom.apo_coords  # [Ntoken, 24, Napo, 3]
        all_apo_mask = struct.atom.apo_mask  # [Ntoken, 24, Napo]

        num_apos = all_apo_coords.shape[2]
        assert num_apos >= 1, "Apo structure must have at least one apo conformation."

        # Synchronized apo structures for entities
        # HACK: Since some chains with the same entity id have different number of
        # tokens or atoms due to PTM, therefore, we need to distinguish this.
        entity_apo_coords: dict[tuple[int, int, int], np.ndarray] = {}
        entity_apo_masks: dict[tuple[int, int, int], np.ndarray] = {}

        # Process each chain
        chain_apo_coords_list: list[np.ndarray] = []
        chain_apo_mask_list: list[np.ndarray] = []
        for chain_i in range(struct.num_chains):
            entity_id = int(struct.chain.entity_id[chain_i])
            num_tokens: int = int(struct.chain.num_tokens[chain_i])
            num_atoms: int = int(struct.chain.num_atoms[chain_i])
            entity_key = (entity_id, num_tokens, num_atoms)

            # Slice indices for the current chain
            st: int = int(struct.chain.token_start[chain_i])
            end: int = st + num_tokens

            # synchronized apo structures for the same entity
            if entity_key in entity_apo_coords:
                chain_apo_coords_list.append(entity_apo_coords[entity_key])
                chain_apo_mask_list.append(entity_apo_masks[entity_key])
                continue

            if rng.random() < self.prob_replace_to_holo:
                # Decide to replace apo with the first bioassembly holo structure
                chain_holo_coords = struct.atom.coords[st:end]  # [L', 24, Nholo, 3]
                chain_holo_mask = struct.atom.resolved_mask[st:end]  # [L', 24]
                chain_apo_coords = chain_holo_coords[..., 0, :]  # [L', 24, 3]
                chain_apo_mask = chain_holo_mask  # [L', 24]
            else:
                # Sample one apo structure
                all_chain_apo_coords = all_apo_coords[st:end]  # [L', 24, Napo, 3]
                all_chain_apo_mask = all_apo_mask[st:end]  # [L', 24, Napo]
                chain_apo_coords, chain_apo_mask = self.sample_apo_structure(
                    all_chain_apo_coords, all_chain_apo_mask, rng
                )  # [Ntoken', 24, 3], [Ntoken', 24], bool

                # Apply apo perturbation (skip if it is replaced to holo coords)
                chain_type = C.ChainType(struct.chain.chain_type[chain_i])
                chain_apo_coords = self.apply_perturbation(
                    chain_apo_coords, chain_apo_mask, chain_type, rng, struct, chain_i
                )

            # TODO: (SeonghwanSeo) Remove this after NA apo prediction is prepared.
            if self.mask_nucleic_acids:
                # Mask out nucleic acid apo structures
                chain_type = C.ChainType(struct.chain.chain_type[chain_i])
                if chain_type in (C.ChainType.DNA, C.ChainType.RNA):
                    chain_apo_coords = np.zeros_like(chain_apo_coords)
                    chain_apo_mask = np.zeros_like(chain_apo_mask)

            # Store for synchronized apo structures
            entity_apo_coords[entity_key] = chain_apo_coords
            entity_apo_masks[entity_key] = chain_apo_mask

            # Append to the chain list
            chain_apo_coords_list.append(chain_apo_coords)
            chain_apo_mask_list.append(chain_apo_mask)

        # Apply random rotation augmentation (not in-place operation)
        chain_apo_coords_list = [
            self.apply_random_rotation(
                chain_apo_coords_list[chain_i], chain_apo_mask_list[chain_i], rng
            )
            for chain_i in range(struct.num_chains)
        ]

        apo_coords: np.ndarray = np.concatenate(chain_apo_coords_list, axis=0)
        apo_mask: np.ndarray = np.concatenate(chain_apo_mask_list, axis=0)
        return apo_coords, apo_mask

    # === Apo sampling === #

    def sample_apo_structure(
        self,
        all_chain_coords: np.ndarray,
        all_chain_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample one apo conformation randomly.
        If no valid apo structures, returns zeros.

        Parameters
        ----------
        all_chain_coords : np.ndarray
            Apo structure coordinates for a chain of shape [Ntoken', 24, Napo, 3].
        all_chain_mask : np.ndarray
            Apo structure mask for a chain of shape [Ntoken', 24, Napo].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        sampled_chain_coords : np.ndarray
            Sampled apo structure coordinates of shape [Ntoken', 24, 3].
        sampled_chain_mask : np.ndarray
            Sampled apo structure mask of shape [Ntoken', 24].
        """
        num_apos = all_chain_coords.shape[2]
        assert num_apos >= 1, "Apo structure must have at least one apo conformation."

        is_apo_valid = all_chain_mask.any(axis=(0, 1))  # [Napo]

        if not np.any(is_apo_valid):
            # If no valid apo structures, return zeros
            token_num = all_chain_coords.shape[0]
            return (
                np.zeros((token_num, 24, 3), dtype=np.float32),
                np.zeros((token_num, 24), dtype=bool),
            )

        if num_apos == 1:
            sampled_idx = 0
        else:
            valid_indices = np.where(is_apo_valid)[0]
            sampled_idx = int(rng.choice(valid_indices))

        sampled_chain_coords = all_chain_coords[:, :, sampled_idx, :]  # [Ntoken', 24, 3]
        sampled_chain_mask = all_chain_mask[:, :, sampled_idx]  # [Ntoken', 24]

        return sampled_chain_coords, sampled_chain_mask

    # Apo perturbation
    def _convert_to_rieprody_data(
        self,
        struct: TokenizedStructure,
        chain_i: int,
        metric_data: dict,
        apo_coords: np.ndarray,
    ) -> dict:
        """Convert TokenizedStructure data to RieProDy format.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure.
        chain_i : int
            Chain index.
        metric_data : dict
            Pre-computed metric data from LMDB.
        apo_coords : np.ndarray
            Apo coordinates for the chain [Ntoken, 24, 3].

        Returns
        -------
        data : dict
            Data dictionary compatible with RieProDy's read_computed_data.
        """

        # Convert numpy arrays to torch tensors
        # Note: RieProDy expects torch tensors
        def to_torch(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr
            return torch.from_numpy(arr).to(dtype)

        # Create a Data-like object compatible with RieProDy
        # The metric_data already contains most required fields
        receptor_data = type("Data", (), {})()

        # Set sequence from metric_data (pre-computed)
        receptor_data.sequence = metric_data["sequence"]

        # Use metric data fields (already in correct format from LMDB)
        receptor_data.metric_inv_cholesky = to_torch(metric_data["metric_inv_cholesky"])
        receptor_data.internal_coords_mask_R8_to_q = to_torch(
            metric_data["internal_coords_mask_R8_to_q"], dtype=torch.bool
        )
        receptor_data.mp_nerf_image_R = to_torch(metric_data["mp_nerf_image_R"])
        receptor_data.initial_q_R8 = to_torch(metric_data["initial_q_R8"])
        receptor_data.mp_nerf_non_empty_atom_mask_R14 = to_torch(
            metric_data["mp_nerf_non_empty_atom_mask_R14"], dtype=torch.bool
        )
        receptor_data.rot_atom_mask = to_torch(
            metric_data["rot_atom_mask"], dtype=torch.bool
        )

        # Additional required fields from metric_data
        if "cloud_mask" in metric_data:
            receptor_data.cloud_mask = to_torch(
                metric_data["cloud_mask"], dtype=torch.bool
            )
        if "cloud_mask_R3" in metric_data and metric_data["cloud_mask_R3"] is not None:
            receptor_data.cloud_mask_R3 = to_torch(
                metric_data["cloud_mask_R3"], dtype=torch.bool
            )
        if "reverse_cloud_mask" in metric_data:
            receptor_data.reverse_cloud_mask = to_torch(
                metric_data["reverse_cloud_mask"], dtype=torch.long
            )
        if (
            "reverse_cloud_mask_R3" in metric_data
            and metric_data["reverse_cloud_mask_R3"] is not None
        ):
            receptor_data.reverse_cloud_mask_R3 = to_torch(
                metric_data["reverse_cloud_mask_R3"], dtype=torch.long
            )
        if "point_ref_mask" in metric_data:
            receptor_data.point_ref_mask = to_torch(
                metric_data["point_ref_mask"], dtype=torch.long
            )
        if "bond_mask" in metric_data:
            receptor_data.bond_mask = to_torch(metric_data["bond_mask"])
        if "apo_angles_mask" in metric_data:
            receptor_data.apo_angles_mask = to_torch(metric_data["apo_angles_mask"])
        elif "angles_mask" in metric_data:
            receptor_data.apo_angles_mask = to_torch(metric_data["angles_mask"])
        if "chi2rot_atom_mask" in metric_data:
            receptor_data.chi2rot_atom_mask = to_torch(metric_data["chi2rot_atom_mask"])
        if "chi2ref_atom_mask" in metric_data:
            receptor_data.chi2ref_atom_mask = to_torch(metric_data["chi2ref_atom_mask"])

        return {"receptor": receptor_data}

    def apply_perturbation(
        self,
        apo_coords: np.ndarray,
        mask: np.ndarray,
        chain_type: C.ChainType,
        rng: np.random.Generator,
        struct: TokenizedStructure | None = None,
        chain_i: int | None = None,
    ) -> np.ndarray:
        """Apply perturbation to apo structure coordinates.

        Parameters
        ----------
        apo_coords : np.ndarray
            Apo structure coordinates of shape [Ntoken, 24, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [Ntoken, 24].
        chain_type : C.ChainType
            Type of the chain (e.g., PROTEIN, NUCLEIC_ACID, LIGAND),
        rng : np.random.Generator
            Random number generator for stochastic operations.
        struct : TokenizedStructure | None, optional
            Tokenized structure. Required for RieProDy perturbation.
        chain_i : int | None, optional
            Chain index. Required for RieProDy perturbation.

        Returns
        -------
        perturbed_apo_coords : np.ndarray
            Perturbed apo structure coordinates of shape [Ntoken, 24, 3].
        """
        if not self.use_perturbation:
            return apo_coords
        if rng.random() > self.prob_perturbation:
            return apo_coords

        # Only apply RieProDy perturbation to proteins
        if chain_type != C.ChainType.PROTEIN:
            return apo_coords

        if struct is None or chain_i is None:
            import warnings

            warnings.warn(
                "struct and chain_i are required for RieProDy perturbation. "
                "Skipping perturbation.",
                UserWarning,
            )
            return apo_coords

        if self.metric_lmdb_path is None:
            import warnings

            warnings.warn(
                "metric_lmdb_path is not provided. Skipping perturbation.",
                UserWarning,
            )
            return apo_coords

        try:
            # Get record ID from metadata
            record_id = struct.metadata.id if struct.metadata else None
            if record_id is None:
                import warnings

                warnings.warn(
                    "Record ID not found in metadata. Skipping perturbation.",
                    UserWarning,
                )
                return apo_coords

            # Get entity_id for this chain
            entity_id = int(struct.chain.entity_id[chain_i])

            # Load metric data from LMDB
            metric_data = self._load_metric_from_lmdb(record_id, entity_id)
            if metric_data is None:
                import warnings

                warnings.warn(
                    f"Metric data not found for {record_id}-{entity_id - 1}. "
                    "Skipping perturbation.",
                    UserWarning,
                )
                return apo_coords

            # Check if RieProDy module is initialized
            if self._rieprody_module is None:
                import warnings

                warnings.warn(
                    "RieProDy module is not initialized. Skipping perturbation.",
                    UserWarning,
                )
                return apo_coords

            # Convert data to RieProDy format
            rieprody_data = self._convert_to_rieprody_data(
                struct, chain_i, metric_data, apo_coords
            )

            # Read computed data into RieProDy module
            self._rieprody_module.read_computed_data(rieprody_data)

            # Apply perturbation using Riemannian Brownian Motion
            # Extract parameters from random_walk config
            params = {
                "total_time": self.random_walk.get("total_time", 0.1),
                "num_steps": self.random_walk.get("num_steps", 10),
                "metric": self.random_walk.get("metric", "lagrangian"),
                "with_christoffel_term": self.random_walk.get(
                    "with_christoffel_term", False
                ),
                "metric_calculation_period": self.random_walk.get(
                    "metric_calculation_period", 1
                ),
                "use_precomputed_metric": self.random_walk.get(
                    "use_precomputed_metric", True
                ),
                "kabsch_aligned_traj": self.random_walk.get("kabsch_aligned_traj", True),
                "return_as_N3": False,
                "return_q": False,
            }
            perturbed_coords = self._rieprody_module.solve_riemannian_brownian_motion(
                **params
            )  # Returns [num_steps+1, R, 14, 3]

            # Extract the final perturbed structure (last step)
            if perturbed_coords.ndim == 4:  # [num_steps+1, R, 14, 3]
                perturbed_r14 = perturbed_coords[-1].detach().cpu().numpy()  # [R, 14, 3]

                # RieProDy uses its own atom14 ordering (NeRF order). Convert per-residue
                # to kfold residue atom order, then pack into kfold's [Ntoken, 24, 3].
                num_tokens = int(apo_coords.shape[0])
                num_residues = int(perturbed_r14.shape[0])

                token_st: int = int(struct.chain.token_start[chain_i])
                token_end: int = token_st + num_tokens

                # cloud_mask indicates which atom14 entries are present per residue.
                cloud_mask = metric_data.get("cloud_mask", None)
                if cloud_mask is None:
                    import warnings

                    warnings.warn(
                        "metric_data is missing 'cloud_mask'. "
                        "Skipping RieProDy->kfold atom-order conversion.",
                        UserWarning,
                    )
                    return apo_coords
                cloud_mask = np.asarray(cloud_mask).astype(bool)  # [R, 14]

                # Build residue->token mapping (handles potential non-1:1 tokenization).
                first_residue_idx = int(struct.token.residue_index[token_st])  # 1-based
                residue_to_token_map: dict[int, int] = {}
                for token_local_idx, token_idx in enumerate(range(token_st, token_end)):
                    residue_idx = int(struct.token.residue_index[token_idx])
                    local_residue_idx = (
                        residue_idx - first_residue_idx
                    )  # 0-based within chain
                    if 0 <= local_residue_idx < num_residues:
                        residue_to_token_map[local_residue_idx] = token_local_idx

                # Start from original coords; overwrite only residues we can map.
                output_coords = apo_coords.copy()
                overwritten_mask = np.zeros(mask.shape, dtype=bool)  # [Ntoken, 24]

                for local_res_idx, token_local_idx in residue_to_token_map.items():
                    token_idx = (
                        token_st + token_local_idx
                    )  # global token index for residue info
                    res_type = int(struct.token.res_type[token_idx])
                    res_name = C.residue.residue_index_to_name[res_type]

                    # kfold per-residue atom list (variable length, packed into 24 slots)
                    k_atoms = C.atom.residue_atoms.get(res_name, None)
                    if k_atoms is None:
                        continue
                    na = len(k_atoms)
                    if na == 0 or na > 24:
                        continue

                    # Extract valid atoms in RieProDy atom14 order
                    # (blanks removed via cloud_mask)
                    cm = cloud_mask[local_res_idx]
                    if cm.shape[0] != 14:
                        # Unexpected shape; fall back to first `na` atoms.
                        coords_rie_valid = perturbed_r14[local_res_idx, :na, :]
                    else:
                        coords_rie_valid = perturbed_r14[local_res_idx, cm, :]

                    if coords_rie_valid.shape[0] < na:
                        # Not enough atoms to fill this residue; skip it.
                        continue

                    perm_rie_to_k = C.atom.RIEPRODY_TO_KFOLD_ATOM_ORDER.get(
                        res_name, tuple(range(na))
                    )
                    if len(perm_rie_to_k) != na:
                        perm_rie_to_k = tuple(range(na))

                    coords_k = coords_rie_valid[np.asarray(perm_rie_to_k, dtype=np.int32)]

                    output_coords[token_local_idx, :na, :] = coords_k.astype(np.float32)
                    overwritten_mask[token_local_idx, :na] = True

                # Kabsch-align the perturbed coords to the original apo_coords using only
                # atoms we actually overwrote and that are valid in the input mask.
                align_mask = overwritten_mask & mask  # [Ntoken, 24]
                if int(np.sum(align_mask)) >= 4:
                    flat = output_coords.reshape(num_tokens * 24, 3)
                    target = apo_coords.reshape(num_tokens * 24, 3)
                    flat_mask = align_mask.reshape(num_tokens * 24).astype(np.float32)
                    aligned = rigid_align(flat, target, flat_mask)
                    output_coords = aligned.reshape(num_tokens, 24, 3).astype(np.float32)

                return output_coords
            else:
                import warnings

                warnings.warn(
                    f"Unexpected perturbed_coords shape: {perturbed_coords.shape}. "
                    "Using original coordinates.",
                    UserWarning,
                )
                return apo_coords

        except Exception as e:
            import warnings

            warnings.warn(
                f"Error during RieProDy perturbation: {e}. Using original coordinates.",
                UserWarning,
            )
            return apo_coords

    def apply_random_rotation(
        self, apo_coords: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Augment apo structure coordinates with random rotation.

        Parameters
        ----------
        apo_coords : np.ndarray
            Apo structure coordinates of shape [Ntoken, 24, 3].
        mask : np.ndarray
            Mask indicating valid atoms of shape [Ntoken, 24].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        augmented_coords : np.ndarray
            Augmented structure coordinates of shape [Ntoken, 24, 3].
        """
        # Flatten
        Ntoken = apo_coords.shape[0]
        coords = apo_coords.reshape(Ntoken * 24, 3)
        mask = mask.reshape(Ntoken * 24)
        # Apply random rotation or simple centering(no rotation)
        augmented_coords = center_random_augmentation(
            coords, mask, augmentation=self.use_random_rotation
        )
        return augmented_coords.reshape(Ntoken, 24, 3)

    # === Symmetry correction === #

    def align_apo_to_holo(
        self,
        struct: TokenizedStructure,
        apo_coords: np.ndarray,
        apo_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply symmetry correction to apo structure coordinates.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        apo_coords : np.ndarray
            Apo structure coordinates of shape [Ntoken, 24, 3].
        apo_mask : np.ndarray
            Apo structure masks of shape [Ntoken, 24].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        permuted_apo_coords : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        permuted_apo_mask : np.ndarray
            Permuted apo structure masks of shape [Ntoken, 24].
        """
        # Prepare holo coordinates and masks
        holo_coords: np.ndarray = struct.atom.coords  # [Ntoken, 24, Nholo, 3]
        holo_mask: np.ndarray = struct.atom.resolved_mask  # [Ntoken, 24]
        n_holo = holo_coords.shape[-2]
        if n_holo != 1:
            # Currently only support single bioassembly for symmetry correction
            raise NotImplementedError(
                "Symmetry correction for multiple bioassemblies is not implemented."
            )
        holo_coords = holo_coords[..., 0, :]  # [Ntoken, 24, 3]
        holo_mask = holo_mask  # [Ntoken, 24]

        if not np.any(apo_mask & holo_mask):
            # If there are no valid apo/holo atoms to align, skip permutation.
            return apo_coords, apo_mask

        # Safe in-place modification
        apo_coords = apo_coords.copy()
        apo_mask = apo_mask.copy()

        # First, correct chain-level symmetry
        # NOTE: apo_mask is synchronized across permuting chains
        apo_coords = self.get_best_chain_permutation(
            struct,
            holo_coords,
            apo_coords,
            holo_mask,
            apo_mask,
            max_permutations=100,
            rng=rng,
        )
        # Second, correct residue-level symmetry
        apo_coords, apo_mask = self.get_best_residue_permutation(
            struct, holo_coords, apo_coords, holo_mask, apo_mask
        )
        # Third, correct molecule-level symmetry
        apo_coords, apo_mask = self.get_best_mol_permutation(
            struct, holo_coords, apo_coords, holo_mask, apo_mask
        )
        return apo_coords, apo_mask

    def get_best_chain_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        holo_mask: np.ndarray,
        apo_mask: np.ndarray,
        max_permutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Find the best chain permutation for symmetry correction.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        holo_coords : np.ndarray
            Holo structure coordinates. (Ntoken, 24, 3)
        apo_coords : np.ndarray
            Apo structure coordinates. (Ntoken, 24, 3)
        holo_mask: np.ndarray
            Holo structure masks. (Ntoken, 24)
        apo_mask: np.ndarray
            Apo structure masks. (Ntoken, 24)
        max_permutations : int
            Maximum number of permutations to consider.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        permuted_apo_coords : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        """
        # === Quick check to skip permutation === #
        num_chains: int = int(struct.num_chains)
        chain_entity_ids: list[int] = struct.chain.entity_id.tolist()
        if len(set(chain_entity_ids)) == num_chains:
            # If all chains have unique entity IDs, no permutation is needed.
            return apo_coords

        # === Identify entities with multiple chains of the same size === #
        entity_chains: dict[int, list[int]] = defaultdict(list)
        for chain_i in range(struct.num_chains):
            # NOTE: use chain_idx instead of asym_id for easy indexing
            entity_id = chain_entity_ids[chain_i]
            entity_chains[entity_id].append(chain_i)

        # Exclude entities with only one chain or mismatched sizes
        for entity_id in list(entity_chains.keys()):
            chains = entity_chains[entity_id]
            if len(chains) == 1:
                entity_chains.pop(entity_id)
            ref_num_tokens = struct.chain.num_tokens[chains[0]]
            ref_num_atoms = struct.chain.num_atoms[chains[0]]
            if np.any(struct.chain.num_tokens[chains] != ref_num_tokens) or np.any(
                struct.chain.num_atoms[chains] != ref_num_atoms
            ):
                entity_chains.pop(entity_id)
                continue
        entities_with_symmetry: set[int] = set(entity_chains.keys())

        # === Extract center atom coordinates and resolved masks === #
        # NOTE: apo_mask is synchronized across permuting chains
        align_mask = holo_mask & apo_mask  # [Ntoken, 24]

        # Center atom is CA for protein, C1' for nucleic acid, and centroid for ligand
        token_index = struct.token.token_index  # [Ntoken]
        center_index = struct.token.center_index  # [Ntoken]
        holo_centers = holo_coords[token_index, center_index]  # [Ntoken, 3]
        apo_centers = apo_coords[token_index, center_index]  # [Ntoken, 3]
        center_mask = align_mask[token_index, center_index]  # [Ntoken]

        # === Sample anchor tokens for alignment === #
        # To reduce computation, use a subset of tokens as anchors
        chain_anchor_tokens: list[np.ndarray] = []
        chain_anchor_weights: list[np.ndarray] = []
        entity_anchor_tokens: dict[tuple[int, int, int], np.ndarray] = {}
        for chain_i in range(struct.num_chains):
            entity_id: int = int(chain_entity_ids[chain_i])
            num_tokens: int = int(struct.chain.num_tokens[chain_i])
            num_atoms: int = int(struct.chain.num_atoms[chain_i])
            entity_type = (entity_id, num_tokens, num_atoms)

            st: int = int(struct.chain.token_start[chain_i])
            if entity_type in entity_anchor_tokens:
                # synchronized anchors
                anchor_tokens = entity_anchor_tokens[entity_type]
            else:
                num_anchors: int = min(10, num_tokens)
                stride = max(1, num_tokens // num_anchors)
                anchor_tokens = np.arange(0, num_tokens, stride, dtype=np.int32)
                if entity_type in entities_with_symmetry:
                    # Only store for entities with symmetry
                    entity_anchor_tokens[entity_type] = anchor_tokens
            num_anchors = anchor_tokens.shape[0]
            weights = np.full((num_anchors,), num_tokens / num_anchors, dtype=np.float32)
            chain_anchor_tokens.append(anchor_tokens + st)
            chain_anchor_weights.append(weights)
        del entity_anchor_tokens

        # === Get static variables for alignment === #
        # Following variables are constant during permutation search
        total_anchors = np.concatenate(
            [chain_anchor_tokens[chain_i] for chain_i in range(struct.num_chains)]
        )  # [Nanchor,]
        holo_centers = holo_centers[total_anchors]  # [Nanchor, 3]
        center_mask = center_mask[total_anchors]  # [Nanchor,]
        align_weights: np.ndarray = np.concatenate(chain_anchor_weights, axis=0)
        align_weights[~center_mask] = 0.0  # mask out invalid atoms
        weight_sum = align_weights.sum().clip(min=1)

        # === Collect possible permutations === #
        # Limit the number of permutations to avoid combinatorial explosion
        max_candidates = max_permutations * 10
        original_perm: list[int] = list(range(num_chains))
        permutations: list[list[int]] = [original_perm]

        for entity_id in entities_with_symmetry:
            chains = entity_chains[entity_id]

            if len(chains) > 5:
                # Skip to avoid combinatorial explosion
                # 6!=720
                continue

            group_swaps = list(itertools.permutations(chains))

            if len(group_swaps) <= 1:
                continue

            new_permutations: list[list[int]] = []

            for base_perm in permutations:
                for swap in group_swaps:
                    child_perm = base_perm.copy()
                    for slot_idx, new_chain_idx in zip(chains, swap, strict=True):
                        child_perm[slot_idx] = new_chain_idx
                    new_permutations.append(child_perm)
                    if len(new_permutations) >= max_candidates:
                        break
                if len(new_permutations) >= max_candidates:
                    break

            permutations = new_permutations
            if len(permutations) >= max_candidates:
                break

        if len(permutations) > max_permutations:
            # Randomly sample permutations if too many
            permutations = permutations[1:]  # exclude original permutation
            indices = rng.choice(
                len(permutations), size=max_permutations, replace=False
            ).tolist()
            permutations = [original_perm] + [permutations[i] for i in indices]

        # === Find best permutation with minimum weighted RMSD === #
        best_permutation: list[int] = list(range(num_chains))
        min_weighted_mse: float = float("inf")
        for perm in permutations:
            # Get permuted anchor indices
            permuted_anchors = np.concatenate(
                [chain_anchor_tokens[perm[chain_i]] for chain_i in range(num_chains)]
            )  # [Nanchor,]

            # Get apo coordinates with the current permutation
            permuted_apo_centers = apo_centers[permuted_anchors]  # [Nanchor, 3]

            # Align permuted apo to holo
            permuted_apo_centers = weighted_rigid_align(
                permuted_apo_centers, holo_centers, align_weights, center_mask
            )

            # Compute weighted RMSD
            d = np.linalg.norm(permuted_apo_centers - holo_centers, axis=-1)  # [Nanchor,]
            w = align_weights  # already masked

            w_sum = weight_sum
            weighted_mse = np.sum((d**2) * w) / w_sum
            del permuted_apo_centers, d, w

            if weighted_mse < min_weighted_mse:
                min_weighted_mse = weighted_mse
                best_permutation = perm

        # === Apply best permutation to all apo coordinates === #
        chain_apo_coords: list[np.ndarray] = []
        for permuted_chain_i in best_permutation:
            token_st: int = int(struct.chain.token_start[permuted_chain_i])
            token_num: int = int(struct.chain.num_tokens[permuted_chain_i])
            token_end: int = token_st + token_num
            chain_apo_coords.append(apo_coords[token_st:token_end])

        permuted_apo_coords = np.concatenate(chain_apo_coords, axis=0)  # [Ntoken, 24, 3]
        return permuted_apo_coords

    def get_best_residue_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        holo_mask: np.ndarray,
        apo_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the best residue permutation for symmetry correction.
        Use intra-residue structure comparison to find the best permutation.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        holo_coords : np.ndarray
            Holo structure coordinates. (Ntoken, 24, 3)
        apo_coords : np.ndarray
            Apo structure coordinates. (Ntoken, 24, 3)
        holo_mask : np.ndarray
            Holo structure masks. (Ntoken, 24)
        apo_mask : np.ndarray
            Apo structure masks. (Ntoken, 24)

        Returns
        -------
        permuted_apo_coords : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        """
        for i in range(struct.num_tokens):
            num_atoms = struct.token.num_atoms[i]
            if num_atoms == 1:
                # Skip PTM or unknown residues here.
                continue

            restype = int(struct.token.res_type[i])
            res_name = C.residue.residue_index_to_name[restype]
            perms = get_ambiguous_atoms_in_residue(res_name)
            if len(perms) <= 1:
                # No ambiguous atoms, skip
                continue

            res_holo_coords = holo_coords[i, :num_atoms, :]  # [num_res_atoms, 3]
            res_apo_coords = apo_coords[i, :num_atoms, :]  # [num_res_atoms, 3]
            res_holo_mask = holo_mask[i, :num_atoms]  # [num_res_atoms,]
            res_apo_mask = apo_mask[i, :num_atoms]  # [num_res_atoms,]

            # Find the best permutation
            best_perm = list(range(num_atoms))
            min_rmsd = float("inf")

            for perm in perms:
                permuted_apo_coords = res_apo_coords[perm, :]
                permuted_apo_mask = res_apo_mask[perm]
                align_mask = res_holo_mask & permuted_apo_mask
                if np.sum(align_mask) < 4:
                    continue  # Not enough resolved atoms to align
                rmsd = compute_rmsd(
                    permuted_apo_coords, res_holo_coords, align_mask, align=True
                )
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    best_perm = perm

            # Apply best permutation
            apo_coords[i, :num_atoms, :] = res_apo_coords[best_perm, :]
            apo_mask[i, :num_atoms] = res_apo_mask[best_perm]

        return apo_coords, apo_mask

    def get_best_mol_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        holo_mask: np.ndarray,
        apo_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the best molecule permutation for symmetry correction.
        Use intra-molecule structure comparison to find the best permutation.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        holo_coords : np.ndarray
            Holo structure coordinates. (Ntoken, 24, 3)
        apo_coords : np.ndarray
            Apo structure coordinates. (Ntoken, 24, 3)
        holo_mask : np.ndarray
            Holo structure masks. (Ntoken, 24)
        apo_mask : np.ndarray
            Apo structure masks. (Ntoken, 24)

        Returns
        -------
        permuted_apo_coords : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        permuted_apo_mask : np.ndarray
            Permuted apo structure masks of shape [Ntoken, 24].
        """
        residue_mol: OrderedDict[ResUID, tuple[str, list[str]]] = OrderedDict()
        mol_token_indices: dict[ResUID, tuple[int, int]] = {}

        for res_i in range(struct.num_residues):
            # NOTE: res_i is global residue index (0-based), not per-chain index (1-based)
            if struct.residue.is_standard[res_i]:
                # Skip standard amino acids and nucleotides
                continue

            # Get CCD ID
            ccd_id = str(struct.residue.name[res_i])
            if ccd_id not in self.ccd_symmetry_dict:
                # No symmetry information for this molecule
                continue

            # Get unique residue id (chain id, residue index)
            asym_id = int(struct.residue.asym_id[res_i])
            res_idx = int(struct.residue.residue_index[res_i])
            res_uid = (asym_id, res_idx)

            # Check input valid
            num_atoms = int(struct.residue.num_atoms[res_i])
            num_tokens = int(struct.residue.num_tokens[res_i])
            assert num_atoms == num_tokens, (
                f"Molecule token and atom number mismatch, {num_atoms} != {num_tokens}"
            )

            # Get atom names
            token_st = int(struct.residue.token_start[res_i])
            token_end = token_st + num_tokens
            mol_atom_names: list[str] = [
                C.atom.decode_atom_name(struct.atom.ref_atom_name_chars[i, 0].tolist())
                for i in range(token_st, token_end)
            ]
            residue_mol[res_uid] = (ccd_id, mol_atom_names)
            mol_token_indices[res_uid] = (token_st, token_end)

        # Get the best permutation for each molecule
        for mol_uid in residue_mol.keys():
            ccd_id, mol_atom_names = residue_mol[mol_uid]

            token_st, token_end = mol_token_indices[mol_uid]
            num_atoms = token_end - token_st
            if num_atoms < 4:
                # Not enough atoms to align
                continue

            permutations = get_molecule_symmetries(
                ccd_id, mol_atom_names, self.ccd_symmetry_dict
            )

            # NOTE: for molecule, there is only one atom per token (always index=0)
            # i.e., only the first atom is valid among 24 atom.
            mol_holo_coords = holo_coords[token_st:token_end, 0, :]  # [num_mol_atoms, 3]
            mol_apo_coords = apo_coords[token_st:token_end, 0, :]  # [num_mol_atoms, 3]
            mol_holo_mask = holo_mask[token_st:token_end, 0]  # [num_mol_atoms,]
            mol_apo_mask = apo_mask[token_st:token_end, 0]  # [num_mol_atoms,]

            # Find the best permutation
            best_perm = list(range(num_atoms))
            min_rmsd = float("inf")

            for perm in permutations:
                permuted_coords = mol_apo_coords[perm, :]
                permuted_mask = mol_apo_mask[perm]
                align_mask = mol_holo_mask & permuted_mask
                if np.sum(align_mask) < 4:
                    continue  # Not enough resolved atoms to align

                rmsd = compute_rmsd(
                    permuted_coords, mol_holo_coords, align_mask, align=True
                )
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    best_perm = perm

            # Apply best permutation
            apo_coords[token_st:token_end, 0, :] = mol_apo_coords[best_perm, :]
            apo_mask[token_st:token_end, 0] = mol_apo_mask[best_perm]
        return apo_coords, apo_mask
