import itertools
import os
import pickle
import warnings
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
        use_entity_random_translation: bool = False,
        entity_random_translation_sigma: float = 0.0,
        prob_perturbation: float = 0.5,
        prob_replace_to_holo: float = 0.0,
        mask_nucleic_acids: bool = False,
        ccd_symmetry_dict: dict | None = None,
        seed: int | None = 42,
        metric_lmdb_path: Path | str | None = None,
        metric_comp: dict | None = None,
        random_walk: dict | None = None,
        langevin: dict | None = None,
        rmsd_threshold: float = 15.0,
        log_stats: bool = False,
        log_stats_interval: int = 1000,
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
        use_entity_random_translation : bool, optional
            Whether to apply an entity-level random translation before caching
            synchronized apo structures. A single translation vector is sampled per
            entity_key=(entity_id,num_tokens,num_atoms) and applied to apo_mask=True
            atoms only.
        entity_random_translation_sigma : float, optional
            Standard deviation (Angstrom) for the entity-level random translation
            sampled from N(0, sigma^2) independently per axis. If sigma <= 0, this
            is treated as disabled.
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
            - total_time (maximum total simulation time; sampled uniformly each call)
            - total_time_min (optional minimum; default: 0.0)
            - num_steps
            - metric_calculation_period
            - metric (optional, default: "lagrangian")
            - with_christoffel_term (optional, default: False)
            - use_precomputed_metric (optional, default: True)
            - kabsch_aligned_traj (optional, default: True)
        langevin : dict | None, optional
            Configuration for Langevin-dynamics-based perturbation.
            Used as a fallback for proteins when RieProDy perturbation is unavailable,
            and as the default perturbation for nucleic acids when perturbation is
            enabled.
            Expected keys (all optional):
            - num_steps (int, default: 64)
            - dt (float, default: 0.25)
            - res_r (float, default: 4.0)
            - ent_r (float, default: 10.0)
            - sphere_r (float, default: 10.0)
            - bond_coef (float, default: 2.0)
        rmsd_threshold : float, optional
            RMSD threshold in Angstroms. If perturbation causes RMSD > threshold,
            original coordinates are used instead. Default: 15.0
        log_stats : bool, optional
            Whether to log perturbation statistics periodically. Default: False
        log_stats_interval : int, optional
            Log statistics every N chain perturbations. Default: 1000
        """
        self.use_perturbation: bool = use_perturbation
        self.use_random_rotation: bool = use_random_rotation
        self.use_symmetry_correction: bool = use_symmetry_correction
        self.use_entity_random_translation: bool = use_entity_random_translation
        self.entity_random_translation_sigma: float = float(
            entity_random_translation_sigma
        )
        self.prob_perturbation: float = prob_perturbation
        self.prob_replace_to_holo: float = prob_replace_to_holo
        self.mask_nucleic_acids: bool = mask_nucleic_acids
        self.seed: int | None = seed
        self.rmsd_threshold: float = rmsd_threshold
        self.log_stats: bool = log_stats
        self.log_stats_interval: int = log_stats_interval

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
        self.langevin: dict | None = langevin

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

            from omegaconf import OmegaConf

            # OmegaConf DictConfig -> plain dict (wrapped by SimpleConfig below)
            metric_comp_dict = (
                OmegaConf.to_container(metric_comp, resolve=True)
                if not isinstance(metric_comp, dict)
                else dict(metric_comp)
            )
            random_walk_dict = (
                OmegaConf.to_container(random_walk, resolve=True)
                if not isinstance(random_walk, dict)
                else dict(random_walk)
            )
            assert isinstance(metric_comp_dict, dict)
            assert isinstance(random_walk_dict, dict)

            # RieProDy's __init__ accesses this path (we don't call preprocess()).
            metric_comp_dict.setdefault(
                "apo_internal_coord_metric_information_path", str(PathLib("."))
            )

            config_dict = {
                "metric_comp": metric_comp_dict,
                "random_walk": random_walk_dict,
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

        # Debug / diagnostics (prints only on exceptions)
        self._debug_apo_perturbation: bool = (
            os.environ.get("KFOLD_APO_PERTURB_DEBUG", "0") == "1"
        )
        try:
            self._debug_apo_perturbation_max_errors: int = int(
                os.environ.get("KFOLD_APO_PERTURB_DEBUG_MAX", "3")
            )
        except ValueError:
            self._debug_apo_perturbation_max_errors = 3
        self._debug_apo_perturbation_error_count: int = 0

        self._stats_total_perturbations: int = 0
        self._stats_rmsd_filtered: int = 0
        self._stats_shape_mismatch: int = 0
        self._stats_success: int = 0
        self._stats_langevin_used: int = 0

        if self.entity_random_translation_sigma < 0.0:
            raise ValueError(
                "entity_random_translation_sigma must be >= 0.0 "
                f"(got {self.entity_random_translation_sigma})."
            )

    @staticmethod
    def _apply_masked_translation(
        coords: np.ndarray, mask: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """Apply a 3D translation to masked atoms only (mask=False atoms unchanged)."""
        if coords.size == 0:
            return coords
        if translation is None or translation.shape != (3,):
            return coords
        if not np.any(mask):
            return coords
        if np.allclose(translation, 0.0):
            return coords

        out = coords.astype(np.float32, copy=True)
        mask_bool = mask.astype(bool, copy=False)
        out[mask_bool] += translation.astype(np.float32, copy=False)
        return out

    def _get_or_sample_entity_translation(
        self,
        entity_key: tuple[int, int, int],
        rng: np.random.Generator,
        cache: dict[tuple[int, int, int], np.ndarray],
    ) -> np.ndarray:
        """Return a cached entity translation or sample a new one."""
        if (
            not self.use_entity_random_translation
        ) or self.entity_random_translation_sigma <= 0.0:
            return np.zeros((3,), dtype=np.float32)
        if entity_key in cache:
            return cache[entity_key]

        sigma = float(self.entity_random_translation_sigma)
        t = rng.normal(loc=0.0, scale=sigma, size=(3,)).astype(np.float32)
        cache[entity_key] = t
        return t

    def _sample_random_walk_total_time(self, rng: np.random.Generator) -> float:
        """Sample random-walk total_time for Riemannian Brownian motion.

        Interpretation:
        - `random_walk.total_time` is treated as the maximum value (max_time).
        - Optionally, `random_walk.total_time_min` can be provided as the minimum value.
        - If max_time == min_time, the value is treated as fixed.
        - Otherwise, we sample uniformly from [min_time, max_time).
        """
        if self.random_walk is None:
            return 0.1

        max_time = float(self.random_walk.get("total_time", 0.1))
        min_time = float(self.random_walk.get("total_time_min", 0.0))

        if max_time < min_time:
            warnings.warn(
                (
                    f"random_walk.total_time ({max_time}) < "
                    f"random_walk.total_time_min ({min_time}). Swapping the bounds."
                ),
                UserWarning,
            )
            min_time, max_time = max_time, min_time

        if max_time == min_time:
            return max_time

        return float(rng.uniform(min_time, max_time))

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

    def get_perturbation_stats(self) -> dict[str, float]:
        """Get perturbation statistics as a dictionary with counts and percentages."""
        total = self._stats_total_perturbations
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "rmsd_filtered": 0,
                "shape_mismatch": 0,
                "success_pct": 0.0,
                "rmsd_filtered_pct": 0.0,
                "shape_mismatch_pct": 0.0,
            }
        return {
            "total": total,
            "success": self._stats_success,
            "rmsd_filtered": self._stats_rmsd_filtered,
            "shape_mismatch": self._stats_shape_mismatch,
            "success_pct": 100.0 * self._stats_success / total,
            "rmsd_filtered_pct": 100.0 * self._stats_rmsd_filtered / total,
            "shape_mismatch_pct": 100.0 * self._stats_shape_mismatch / total,
        }

    def reset_perturbation_stats(self) -> None:
        """Reset all perturbation statistics counters."""
        self._stats_total_perturbations = 0
        self._stats_rmsd_filtered = 0
        self._stats_shape_mismatch = 0
        self._stats_success = 0

    def _log_perturbation_stats(self) -> None:
        stats = self.get_perturbation_stats()
        print(
            f"[ApoPerturbation] n={stats['total']}: "
            f"success={stats['success_pct']:.1f}%, "
            f"rmsd_filtered={stats['rmsd_filtered_pct']:.1f}%, "
            f"shape_mismatch={stats['shape_mismatch_pct']:.1f}%"
        )

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
            warnings.warn(
                f"Failed to load metric data for {record_id}-{entity_index}: {e}",
                UserWarning,
            )
            return None
    def __call__(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        return self.run(struct, rng)

    def run(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
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
        rng = rng or np.random.default_rng()

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
        entity_translations: dict[tuple[int, int, int], np.ndarray] = {}

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
                if (
                    chain_type in (C.ChainType.DNA, C.ChainType.RNA)
                    and self.use_perturbation
                ):
                    # For nucleic acids, we define apo_mask from holo-resolved atoms so
                    # the model can use the generated apo prior (LD starts from random).
                    chain_apo_mask = struct.atom.resolved_mask[st:end].astype(bool)
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

            # Entity-level random translation (applied before caching).
            t = self._get_or_sample_entity_translation(
                entity_key, rng, entity_translations
            )
            chain_apo_coords = self._apply_masked_translation(
                chain_apo_coords, chain_apo_mask, t
            )

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

        # Create a Data-like object compatible with RieProDy.
        # RieProDy expects `data["receptor"].to(device)` to exist.
        class _RieProDyReceptorData:
            def to(self, device: torch.device | str):
                for k, v in self.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        setattr(self, k, v.to(device))
                return self

        receptor_data = _RieProDyReceptorData()

        # Set sequence from metric_data (pre-computed)
        receptor_data.sequence = metric_data["sequence"]
        num_res = len(receptor_data.sequence)

        # Use metric data fields (already in correct format from LMDB)
        # We need consider_side_chain early to interpret q-dim.
        consider_side_chain = True
        if self.random_walk is not None:
            consider_side_chain = bool(
                self.random_walk.get("consider_side_chain_in_metric", True)
            )

        receptor_data.internal_coords_mask_R8_to_q = to_torch(
            metric_data["internal_coords_mask_R8_to_q"], dtype=torch.bool
        )

        # Match RDBDock: unpack packed metric_inv_cholesky via RieProDy utility.
        # RieProDy expects `metric_inv_cholesky` shaped like [R, 8, R, 8].
        metric_inv_cholesky = to_torch(metric_data["metric_inv_cholesky"]).to(
            torch.float32
        )
        if metric_inv_cholesky.ndim in (1, 2):
            metric_inv_cholesky = ProteinPerturbationModule.unpack_metric_inv_cholesky(
                metric_inv_cholesky,
                num_res,
                side_chain=consider_side_chain,
            )

        receptor_data.metric_inv_cholesky = metric_inv_cholesky
        receptor_data.mp_nerf_image_R = to_torch(metric_data["mp_nerf_image_R"])
        receptor_data.initial_q_R8 = to_torch(metric_data["initial_q_R8"])
        receptor_data.mp_nerf_non_empty_atom_mask_R14 = to_torch(
            metric_data["mp_nerf_non_empty_atom_mask_R14"], dtype=torch.bool
        )
        receptor_data.rot_atom_mask = to_torch(
            metric_data["rot_atom_mask"], dtype=torch.bool
        )

        # Additional required fields from metric_data
        # cloud_mask / reverse_cloud_mask
        if "cloud_mask" in metric_data and metric_data["cloud_mask"] is not None:
            cloud_mask = to_torch(metric_data["cloud_mask"], dtype=torch.bool)
            if cloud_mask.ndim == 1 and cloud_mask.numel() == num_res * 14:
                cloud_mask = cloud_mask.view(num_res, 14)
            receptor_data.cloud_mask = cloud_mask

            if (
                "reverse_cloud_mask" in metric_data
                and metric_data["reverse_cloud_mask"] is not None
            ):
                receptor_data.reverse_cloud_mask = to_torch(
                    metric_data["reverse_cloud_mask"], dtype=torch.long
                )
            else:
                counts = cloud_mask.to(torch.long).sum(dim=-1)
                receptor_data.reverse_cloud_mask = torch.repeat_interleave(
                    torch.arange(len(counts), device=counts.device), counts
                )

            # R3 masks are required when consider_side_chain_in_metric=False
            if not consider_side_chain:
                if (
                    "cloud_mask_R3" in metric_data
                    and metric_data["cloud_mask_R3"] is not None
                ):
                    cloud_mask_r3 = to_torch(
                        metric_data["cloud_mask_R3"], dtype=torch.bool
                    )
                    if cloud_mask_r3.ndim == 1 and cloud_mask_r3.numel() == num_res * 3:
                        cloud_mask_r3 = cloud_mask_r3.view(num_res, 3)
                else:
                    cloud_mask_r3 = cloud_mask[:, :3].clone()
                receptor_data.cloud_mask_R3 = cloud_mask_r3

                if (
                    "reverse_cloud_mask_R3" in metric_data
                    and metric_data["reverse_cloud_mask_R3"] is not None
                ):
                    receptor_data.reverse_cloud_mask_R3 = to_torch(
                        metric_data["reverse_cloud_mask_R3"], dtype=torch.long
                    )
                else:
                    counts3 = cloud_mask_r3.to(torch.long).sum(dim=-1)
                    receptor_data.reverse_cloud_mask_R3 = torch.repeat_interleave(
                        torch.arange(len(counts3), device=counts3.device), counts3
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

        # Nucleic acids: always apply Langevin perturbation (when enabled).
        if chain_type in (C.ChainType.DNA, C.ChainType.RNA):
            if struct is None or chain_i is None:
                warnings.warn(
                    "struct and chain_i are required for Langevin perturbation. "
                    "Skipping perturbation.",
                    UserWarning,
                )
                return apo_coords
            record_id = struct.metadata.id if struct.metadata else None
            entity_id = int(struct.chain.entity_id[chain_i]) if struct is not None else -1
            return self.langevin_dynamics_perturbation(
                # NOTE: For nucleic acids, Langevin dynamics starts from random
                # atomic positions (Gaussian init) and does NOT use input apo_coords.
                apo_coords=None,
                mask=mask,
                rng=rng,
                struct=struct,
                chain_i=chain_i,
                record_id=record_id or "<unknown>",
                entity_id=entity_id,
            )

        # For other chain types, optionally skip perturbation by probability.
        # (Protein policy below still respects prob_perturbation.)
        if rng.random() > self.prob_perturbation:
            return apo_coords

        # Only apply perturbation to proteins (ligands/others: no perturbation).
        if chain_type != C.ChainType.PROTEIN:
            return apo_coords

        if struct is None or chain_i is None:
            warnings.warn(
                "struct and chain_i are required for apo perturbation. "
                "Skipping perturbation.",
                UserWarning,
            )
            return apo_coords

        # Get record ID and entity for LMDB lookup
        record_id = struct.metadata.id if struct.metadata else None
        entity_id = int(struct.chain.entity_id[chain_i])

        # Try metric-based RieProDy perturbation only when we can look up LMDB.
        metric_data = None
        if self.metric_lmdb_path is not None and record_id is not None:
            metric_data = self._load_metric_from_lmdb(record_id, entity_id)

        if metric_data is not None:
            return self.rieprody_perturbation(
                apo_coords=apo_coords,
                mask=mask,
                rng=rng,
                struct=struct,
                chain_i=chain_i,
                metric_data=metric_data,
                record_id=record_id,
                entity_id=entity_id,
            )

        # Fallback: Langevin dynamics
        return self.langevin_dynamics_perturbation(
            apo_coords=apo_coords,
            mask=mask,
            rng=rng,
            struct=struct,
            chain_i=chain_i,
            record_id=record_id or "<unknown>",
            entity_id=entity_id,
        )

    def rieprody_perturbation(
        self,
        apo_coords: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        struct: TokenizedStructure,
        chain_i: int,
        metric_data: dict,
        record_id: str | None = None,
        entity_id: int | None = None,
    ) -> np.ndarray:
        """Metric-based perturbation using pre-computed LMDB metric + RieProDy RBM."""
        self._stats_total_perturbations += 1

        if (
            self.log_stats
            and self.log_stats_interval > 0
            and self._stats_total_perturbations % self.log_stats_interval == 0
        ):
            self._log_perturbation_stats()

        if self._rieprody_module is None:
            warnings.warn(
                "RieProDy module is not initialized. Skipping perturbation.",
                UserWarning,
            )
            return apo_coords

        try:
            rieprody_data: dict | None = None

            # Convert data to RieProDy format
            rieprody_data = self._convert_to_rieprody_data(
                struct, chain_i, metric_data, apo_coords
            )
            self._rieprody_module.read_computed_data(rieprody_data)

            # NOTE: perturbation via RieProDy's Riemannian Brownian Motion.
            params = {
                "total_time": self._sample_random_walk_total_time(rng),
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
            )  # [num_steps+1, R, 14, 3]
            if torch.isnan(perturbed_coords).any() or torch.isinf(perturbed_coords).any():
                print(
                    "[DEBUG] NaN/Inf detected in RieProDy output!"
                    f"(Record: {record_id}, Entity: {entity_id})"
                )
                return apo_coords  # Fallback to original coords
            # --------------------------

            if perturbed_coords.ndim != 4:
                warnings.warn(
                    f"Unexpected perturbed_coords shape: {perturbed_coords.shape}. "
                    "Using original coordinates.",
                    UserWarning,
                )
                return apo_coords

            if not torch.isfinite(perturbed_coords).all():
                warnings.warn(
                    "perturbed_coords contains NaNs or Infs. Using original coordinates.",
                    UserWarning,
                )
                return apo_coords

            perturbed_r14 = perturbed_coords[-1].detach().cpu().numpy()  # [R, 14, 3]
            num_tokens = int(apo_coords.shape[0])
            num_residues = int(perturbed_r14.shape[0])

            token_st: int = int(struct.chain.token_start[chain_i])
            token_end: int = token_st + num_tokens

            cloud_mask = metric_data.get("cloud_mask", None)
            if cloud_mask is None:
                warnings.warn(
                    "metric_data is missing 'cloud_mask'. "
                    "Skipping RieProDy->kfold atom-order conversion.",
                    UserWarning,
                )
                return apo_coords
            cloud_mask = np.asarray(cloud_mask).astype(bool)  # [R, 14]
            if cloud_mask.ndim == 1 and cloud_mask.size == num_residues * 14:
                cloud_mask = cloud_mask.reshape(num_residues, 14)

            first_residue_idx = int(struct.token.residue_index[token_st])  # 1-based
            residue_to_token_map: dict[int, int] = {}
            for token_local_idx, token_idx in enumerate(range(token_st, token_end)):
                residue_idx = int(struct.token.residue_index[token_idx])
                local_residue_idx = residue_idx - first_residue_idx
                if 0 <= local_residue_idx < num_residues:
                    residue_to_token_map[local_residue_idx] = token_local_idx

            output_coords = apo_coords.copy()
            overwritten_mask = np.zeros(mask.shape, dtype=bool)  # [Ntoken, 24]

            for local_res_idx, token_local_idx in residue_to_token_map.items():
                token_idx = token_st + token_local_idx
                res_type = int(struct.token.res_type[token_idx])
                res_name = C.residue.residue_id_to_name[res_type]

                k_atoms = C.atom.residue_atoms.get(res_name, None)
                if k_atoms is None:
                    continue
                na = len(k_atoms)
                if na == 0 or na > 24:
                    continue

                if local_res_idx >= cloud_mask.shape[0]:
                    continue

                cm = cloud_mask[local_res_idx]
                if cm.shape[0] != 14:
                    coords_rie_valid = perturbed_r14[local_res_idx, :na, :]
                else:
                    coords_rie_valid = perturbed_r14[local_res_idx, cm, :]

                num_valid_atoms = coords_rie_valid.shape[0]
                if num_valid_atoms < na:
                    continue

                perm_rie_to_k = C.atom.RIEPRODY_TO_KFOLD_ATOM_ORDER.get(
                    res_name, tuple(range(na))
                )
                if len(perm_rie_to_k) != na:
                    perm_rie_to_k = tuple(range(na))

                max_perm_idx = max(perm_rie_to_k) if perm_rie_to_k else 0
                if max_perm_idx >= num_valid_atoms:
                    continue

                coords_k = coords_rie_valid[np.asarray(perm_rie_to_k, dtype=np.int32)]
                output_coords[token_local_idx, :na, :] = coords_k.astype(np.float32)
                overwritten_mask[token_local_idx, :na] = True

            align_mask = overwritten_mask & mask

            if not np.isfinite(output_coords).all():
                warnings.warn(
                    "output_coords contains NaNs or Infs before alignment. "
                    "Using original coordinates.",
                    UserWarning,
                )
                return apo_coords

            if int(np.sum(align_mask)) >= 4:
                flat = output_coords.reshape(num_tokens * 24, 3)
                target = apo_coords.reshape(num_tokens * 24, 3)
                flat_mask = align_mask.reshape(num_tokens * 24).astype(np.float32)
                aligned = rigid_align(flat, target, flat_mask)
                output_coords = aligned.reshape(num_tokens, 24, 3).astype(np.float32)

                rmsd = compute_rmsd(
                    output_coords.reshape(num_tokens * 24, 3),
                    apo_coords.reshape(num_tokens * 24, 3),
                    flat_mask,
                    align=True,
                )
                if rmsd > self.rmsd_threshold:
                    self._stats_rmsd_filtered += 1
                    return apo_coords

            self._stats_success += 1
            return output_coords
        except Exception as e:
            self._debug_apo_perturbation_error_count += 1
            should_dump = (
                self._debug_apo_perturbation
                and self._debug_apo_perturbation_error_count
                <= self._debug_apo_perturbation_max_errors
            )

            def _shape_dtype(x) -> str:
                try:
                    if isinstance(x, torch.Tensor):
                        return f"torch[{tuple(x.shape)}]/{x.dtype}"
                    if isinstance(x, np.ndarray):
                        return f"np[{x.shape}]/{x.dtype}"
                    if isinstance(x, (list, tuple)):
                        return f"{type(x).__name__}[len={len(x)}]"
                    return f"{type(x).__name__}"
                except Exception:
                    return f"{type(x).__name__}"

            extra = ""
            if should_dump:
                import traceback

                key_candidates = [
                    "sequence",
                    "metric_inv_cholesky",
                    "internal_coords_mask_R8_to_q",
                    "cloud_mask",
                    "cloud_mask_R3",
                    "reverse_cloud_mask",
                    "reverse_cloud_mask_R3",
                    "point_ref_mask",
                    "mp_nerf_non_empty_atom_mask_R14",
                    "mp_nerf_image_R",
                    "initial_q_R8",
                    "rot_atom_mask",
                ]
                metric_shapes = {
                    k: _shape_dtype(metric_data.get(k, None)) for k in key_candidates
                }
                receptor_shapes = None
                try:
                    if rieprody_data is not None and "receptor" in rieprody_data:
                        receptor = rieprody_data["receptor"]
                        receptor_shapes = {
                            k: _shape_dtype(v)
                            for k, v in getattr(receptor, "__dict__", {}).items()
                        }
                except Exception:
                    receptor_shapes = {"<error>": "failed to introspect receptor_data"}

                extra = (
                    "\n--- ApoPerturbation debug dump (on exception) ---\n"
                    f"record_id={record_id} entity_id={entity_id} chain_i={chain_i}\n"
                    f"apo_coords={_shape_dtype(apo_coords)} mask={_shape_dtype(mask)}\n"
                    f"metric_keys={sorted(list(metric_data.keys()))[:50]}\n"
                    f"metric_shapes={metric_shapes}\n"
                    f"receptor_shapes={receptor_shapes}\n"
                    f"traceback:\n{traceback.format_exc()}\n"
                    "--- end debug dump ---\n"
                )

            warnings.warn(
                f"Error during metric-based perturbation: {e}. "
                "Using original coordinates.",
                UserWarning,
            )
            if extra:
                warnings.warn(extra, UserWarning)
            return apo_coords

    def langevin_dynamics_perturbation(
        self,
        apo_coords: np.ndarray | None,
        mask: np.ndarray,
        rng: np.random.Generator,
        struct: TokenizedStructure,
        chain_i: int,
        record_id: str,
        entity_id: int,
    ) -> np.ndarray:
        """Fallback perturbation when LMDB metric data is missing."""
        self._stats_total_perturbations += 1
        self._stats_langevin_used += 1

        # === Hyperparameters (Algorithm S3 defaults) ===
        cfg = self.langevin or {}
        num_steps = int(cfg.get("num_steps", 64))
        dt = float(cfg.get("dt", 0.25))
        res_r = float(cfg.get("res_r", 4.0))
        ent_r = float(cfg.get("ent_r", 10.0))
        sphere_r = float(cfg.get("sphere_r", 10.0))
        bond_coef = float(cfg.get("bond_coef", 2.0))

        if num_steps <= 0 or dt <= 0.0:
            return apo_coords

        # === Chain slice ===
        token_st: int = int(struct.chain.token_start[chain_i])
        num_tokens: int = int(struct.chain.num_tokens[chain_i])
        token_end: int = token_st + num_tokens

        # Sanity: mask is a per-chain slice ([L,24]) and (optional) apo_coords is [L,24,3]
        if mask.shape != (num_tokens, 24):
            self._stats_shape_mismatch += 1
            if apo_coords is None:
                return np.zeros((num_tokens, 24, 3), dtype=np.float32)
            return apo_coords
        if apo_coords is not None and apo_coords.shape[:2] != (num_tokens, 24):
            self._stats_shape_mismatch += 1
            return apo_coords

        chain_type = C.ChainType(struct.chain.chain_type[chain_i])
        update_mask = mask.astype(bool, copy=False)

        # === Initialize X0 ===
        if apo_coords is None:
            x = np.zeros((num_tokens, 24, 3), dtype=np.float32)
        else:
            x = apo_coords.astype(np.float32, copy=True)
        if chain_type in (C.ChainType.DNA, C.ChainType.RNA):
            # Start from random atomic positions for nucleic acids.
            # Scale by sphere_r so global compactness term has the right magnitude.
            x = rng.normal(loc=0.0, scale=sphere_r, size=x.shape).astype(np.float32)
            x[~update_mask] = 0.0

        # === Build group indices for S_residue and S_entity (asym_id) ===
        # Residue grouping uses residue_index within this chain slice.
        residue_index = struct.token.residue_index[token_st:token_end].astype(np.int32)
        # Map (residue_index -> list of token indices)
        res_to_tokens: dict[int, list[int]] = defaultdict(list)
        for i, ridx in enumerate(residue_index.tolist()):
            res_to_tokens[int(ridx)].append(i)

        # Entity grouping is per asym_id (physical chain instance). Within a single chain
        # slice, this is just a single group (the chain mean).

        # === Build bond adjacency for S_bond ===
        # Flattened atom index: flat = token_local * 24 + atom_in_token (0..23)
        n_flat = num_tokens * 24
        neighbors: list[list[int]] = [[] for _ in range(n_flat)]

        try:
            bond_token = struct.bond.token_index.astype(np.int32, copy=False)  # [Nbond,2]
            bond_atom = struct.bond.atom_index.astype(np.int32, copy=False)  # [Nbond,2]
            # Filter bonds where both endpoints are within this chain slice.
            in_slice = (bond_token[:, 0] >= token_st) & (bond_token[:, 0] < token_end)
            in_slice &= (bond_token[:, 1] >= token_st) & (bond_token[:, 1] < token_end)
            idxs = np.where(in_slice)[0]
            for b in idxs.tolist():
                t1 = int(bond_token[b, 0] - token_st)
                t2 = int(bond_token[b, 1] - token_st)
                a1 = int(bond_atom[b, 0])
                a2 = int(bond_atom[b, 1])
                if not (0 <= a1 < 24 and 0 <= a2 < 24):
                    continue
                i1 = t1 * 24 + a1
                i2 = t2 * 24 + a2
                neighbors[i1].append(i2)
                neighbors[i2].append(i1)
        except Exception:
            # If bonds are missing/unavailable, just skip bond term.
            pass

        # === Helper: masked mean over a group of tokens ===
        def _mean_coords_for_tokens(token_ids: list[int]) -> np.ndarray:
            # Returns array shaped [len(token_ids), 24, 3] filled with the group mean
            # broadcasted to each token, or zeros if no valid atoms.
            # We compute mean over all atoms in the group that are in update_mask.
            group_mask = update_mask[token_ids]  # [G,24]
            if not np.any(group_mask):
                return np.zeros((len(token_ids), 24, 3), dtype=np.float32)
            coords = x[token_ids]  # [G,24,3]
            w = group_mask.astype(np.float32)[..., None]  # [G,24,1]
            denom = float(w.sum())
            if denom <= 0.0:
                return np.zeros((len(token_ids), 24, 3), dtype=np.float32)
            mean = (coords * w).sum(axis=(0, 1), keepdims=False) / denom  # [3]
            out = np.broadcast_to(mean.reshape(1, 1, 3), (len(token_ids), 24, 3)).copy()
            return out.astype(np.float32, copy=False)

        # === Langevin dynamics (Algorithm S3) ===
        res_r2 = res_r * res_r
        ent_r2 = ent_r * ent_r
        sphere_r2 = sphere_r * sphere_r
        noise_scale = float(2.0 * np.sqrt(dt))

        for _ in range(num_steps):
            # d_entity: chain mean
            if np.any(update_mask):
                w_all = update_mask.astype(np.float32)[..., None]
                denom_all = float(w_all.sum())
                if denom_all > 0.0:
                    mean_chain = (x * w_all).sum(axis=(0, 1)) / denom_all  # [3]
                else:
                    mean_chain = np.zeros((3,), dtype=np.float32)
            else:
                mean_chain = np.zeros((3,), dtype=np.float32)

            d_ent = mean_chain.reshape(1, 1, 3) - x  # [L,24,3]

            # d_residue: residue mean
            d_res = np.zeros_like(x, dtype=np.float32)
            for token_ids in res_to_tokens.values():
                mean_broadcast = _mean_coords_for_tokens(token_ids)  # [G,24,3]
                d_res[token_ids] = mean_broadcast - x[token_ids]

            # d_bond: neighbor mean (per atom)
            d_bond = np.zeros_like(x, dtype=np.float32)
            flat_x = x.reshape(n_flat, 3)
            flat_mask = update_mask.reshape(n_flat)
            for i in range(n_flat):
                if not flat_mask[i]:
                    continue
                nb = neighbors[i]
                if not nb:
                    continue
                # Mean of neighbor coords (masked neighbors only)
                nb_idx = [j for j in nb if flat_mask[j]]
                if not nb_idx:
                    continue
                mean_nb = flat_x[nb_idx].mean(axis=0)
                d_bond.reshape(n_flat, 3)[i] = mean_nb - flat_x[i]

            # drift
            drift = bond_coef * d_bond + d_ent / ent_r2 + d_res / res_r2 - x / sphere_r2

            eps = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
            x = x + dt * drift + noise_scale * eps
            x[~update_mask] = 0.0

        # Centering (masked mean to origin)
        if np.any(update_mask):
            w = update_mask.astype(np.float32)[..., None]
            denom = float(w.sum())
            if denom > 0.0:
                center = (x * w).sum(axis=(0, 1)) / denom
                x = x - center.reshape(1, 1, 3)
                x[~update_mask] = 0.0

        if not np.isfinite(x).all():
            # Safety fallback
            if apo_coords is None:
                return np.zeros((num_tokens, 24, 3), dtype=np.float32)
            return apo_coords

        self._stats_success += 1
        return x.astype(np.float32, copy=False)

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
            coords, mask, augmentation=self.use_random_rotation, rng=rng
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
            res_name = C.residue.residue_id_to_name[restype]
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
