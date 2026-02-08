import dataclasses
import itertools
import logging
from collections import defaultdict
from functools import lru_cache

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.utils.simulation.langevin_dynamics import (
    LangevinDynamicsConfig,
    LangevinDynamicsSimulator,
)
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import compute_rmsd

from .apo_initialization import get_ambiguous_atoms_in_residue, get_molecule_symmetries


def sample_uniform_sphere_surface(
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a point uniformly from the surface of a sphere."""
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r_xy = np.sqrt(max(0.0, 1.0 - z * z))
    point = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z], dtype=np.float32)
    return point * np.float32(radius)


@dataclasses.dataclass(kw_only=True)
class PriorSamplerConfig:
    """Configuration for ComplexPriorSampler.

    Attributes
    ----------
    num_samples : int
        Number of prior coordinates to sample.
    use_random_augmentation : bool
        Whether to apply random rotation/translation augmentation
        to apo structures.
    use_chain_com_sampling : bool
        If set, place each chain's center of mass on a sphere surface with
        this radius (uniformly sampled).
    use_ot_permutation : bool
        Whether to apply optimal transport-based permutation
    translation_scale : float
        Scale of random translation augmentation (in Angstrom).
    """

    num_samples: int = 1
    use_random_augmentation: bool = True
    use_chain_com_sampling: bool = False
    use_ot_permutation: bool = False
    translation_scale: float = 1.0  # Angstrom
    # Langevin dynamics parameters for relaxing missing atoms
    relaxation: LangevinDynamicsConfig = dataclasses.field(
        default_factory=lambda: LangevinDynamicsConfig(
            num_steps=200,
            res_r=4.0,
            bond_r=2.0,
        )
    )


class PriorSampler:
    """Class to populate and augment apo structures."""

    def __init__(self, config: PriorSamplerConfig, ccd: CCD) -> None:
        self.config: PriorSamplerConfig = config
        self.ccd: CCD = ccd

        self.num_samples: int = config.num_samples
        self.use_ot_permutation: bool = config.use_ot_permutation
        self.use_chain_com_sampling: bool = config.use_chain_com_sampling
        self.translation_scale: float = config.translation_scale

        # Langevin dynamics simulator for relaxing missing atoms
        self.langevin_simulator = LangevinDynamicsSimulator(config.relaxation)

        # Logger
        self.logger = logging.getLogger("PriorSampler")

    def __call__(self, struct: RefStructure, rng: np.random.Generator) -> np.ndarray:
        """Sample prior coordinates for the given structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        prior_coords : np.ndarray
            Sampled prior coordinates of shape [num_priors, N_atoms, 3].
        """
        return self.sample_prior_coordinates(struct, rng)

    def sample_prior_coordinates(
        self,
        struct: RefStructure,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior coordinates (xT) for the given structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        prior_coords : np.ndarray
            Sampled prior coordinates of shape [num_priors, N_atoms, 3].
        """
        if self.num_samples <= 0:
            return np.empty((0, struct.num_atoms, 3), dtype=np.float32)

        # === 1. Get chain apo coordinates or sample from prior === #
        chain_coords_list: list[np.ndarray] = []
        for i in range(struct.num_chains):
            chain = struct.chains[i]
            chain_coords = chain.atom.apo_coords
            chain_coords_list.append(chain_coords)

        # === 2. Random augmentation === #
        prior_coords_list: list[np.ndarray] = [
            self.sample_prior(chain_coords_list, struct, rng)
            for _ in range(self.num_samples)
        ]
        return np.stack(prior_coords_list, axis=0)

    def sample_prior(
        self,
        chain_apo_list: list[np.ndarray],
        struct: RefStructure,
        rng: np.random.Generator,
    ) -> np.ndarray:
        prior_coords_list: list[np.ndarray] = []
        for chain_i, chain in enumerate(struct.chains):
            chain_coords = chain_apo_list[chain_i]

            # Fill missing atoms and relax
            is_missing: np.ndarray = ~np.isfinite(chain_coords).all(axis=-1)
            if is_missing.any():
                # Insert gaussian noise for missing atoms
                chain_coords = self.fill_missing_atoms(chain_coords, is_missing, rng)
                # Relax using Langevin dynamics
                chain_coords = self.langevin_relaxation(
                    chain_coords, is_missing, chain, rng
                )

            # Apply random rotation/translation augmentation
            augmented_coords = chain_coords
            if self.config.use_random_augmentation:
                augmented_coords = self.apply_random_augmentation(chain_coords, rng)
            prior_coords_list.append(augmented_coords)

        if self.use_ot_permutation:
            # Optimal transport permutation
            prior_coords_list = self.match_optimal_transport_permutation(
                prior_coords_list, struct, rng
            )

        # Combine chains into complex coordinates
        prior_coords = np.concatenate(prior_coords_list, axis=0)
        return prior_coords

    def fill_missing_atoms(
        self,
        prior_coords: np.ndarray,
        is_missing_atom: np.ndarray,
        rng: np.random.Generator,
        scale: float = 16.0,
    ) -> np.ndarray:
        """Fill missing atoms with random noise.

        Parameters
        ----------
        prior_coords : np.ndarray
            Prior coordinates of shape [N_atoms, 3].
        is_missing_atom : np.ndarray
            Boolean mask indicating missing atoms of shape [N_atoms].
        rng : np.random.Generator
            Random number generator for stochastic operations.
        inplace : bool
            Whether to modify prior_coords in-place.
        """
        random_noise = rng.standard_normal(size=prior_coords.shape, dtype=np.float32)
        random_noise *= scale
        return np.where(is_missing_atom[:, None], random_noise, prior_coords)

    def langevin_relaxation(
        self,
        prior_coords: np.ndarray,
        is_missing_atom: np.ndarray,
        ref_chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Relax unresolved atoms using Langevin dynamics."""
        num_residues = ref_chain.num_residues
        residue_index = np.repeat(
            np.arange(num_residues, dtype=np.int32),
            ref_chain.residue.num_atoms.astype(np.int32),
        )
        # Run a few steps of Langevin dynamics to relax the filled coordinates
        prior_coords = self.langevin_simulator(
            prior_coords,
            residue_index,
            rng=rng,
            is_constraint=~is_missing_atom,
        )
        return prior_coords

    def apply_random_augmentation(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Augment coordinates with random rotation/translation.

        Parameters
        ----------
        coords : np.ndarray
            Structure coordinates of shape [Natom, 3].
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        augmented_coords : np.ndarray
            Augmented structure coordinates of shape [Natom, 3].
        """
        assert coords.ndim == 2, "Apo coordinates must be of shape [Natom, 3]."
        # Flatten
        mask: np.ndarray = np.isfinite(coords).all(axis=-1)
        if not mask.any():
            return coords

        # Apply random augmentation
        if self.use_chain_com_sampling:
            augmented_coords = center_random_augmentation(
                coords, mask, augmentation=True, s_trans=0.0, rng=rng
            )
            current_com = augmented_coords[mask].mean(axis=0)
            target_com = sample_uniform_sphere_surface(self.translation_scale, rng)
            shift = target_com - current_com
            augmented_coords += shift[None, :]
        else:
            augmented_coords = center_random_augmentation(
                coords, mask, augmentation=True, s_trans=self.translation_scale, rng=rng
            )

        augmented_coords[~mask] = np.nan
        return augmented_coords

    def match_optimal_transport_permutation(
        self,
        prior_coords_list: list[np.ndarray],
        struct: RefStructure,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Align prior to label to minimize transport cost (RMSD).

        This method solves the discrete optimal transport problem over the
        structure's symmetry group (chain permutations and residue flips).
        It ensures that the flow matching target (label) is aligned to the
        source (apo) with the minimal displacement, constructing the
        optimal straight-line trajectory.

        Parameters
        ----------
        prior_chain_coords_list : list[np.ndarray]
            List of apo chain coordinates. Each element has shape [L, Natom, 3].
        struct : RefStructure
            Reference structure. Apo coords will be permuted in-place.
        rng : np.random.Generator
            Random number generator for stochastic sampling.
        """
        # First, chain permutation
        # RNG state is used for sampling permutations when too many exist
        try:
            prior_coords_list = self.find_best_chain_permutation(
                prior_coords_list, struct, max_permutations=100, rng=rng
            )
        except Exception as e:
            self.logger.error(f"Failed to find best chain permutation: {e}.")

        # Second, residue-level permutation (e.g., flipping)
        try:
            for c_i, chain in enumerate(struct.chains):
                chain_coords = prior_coords_list[c_i]
                self.find_best_residue_permutation(chain_coords, chain)
        except Exception as e:
            self.logger.error(f"Failed to find best residue permutation: {e}.")

        return prior_coords_list

    def find_best_chain_permutation(
        self,
        prior_chain_coords: list[np.ndarray],
        struct: RefStructure,
        max_permutations: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Find the best chain permutation for symmetry correction."""
        # === 0. Validate prior coordinates === #
        assert len(prior_chain_coords) == struct.num_chains, (
            "Number of prior chains must match number of structure chains."
        )
        assert all(
            prior_chain_coords[i].shape[0] == struct.chains[i].num_atoms
            for i in range(struct.num_chains)
        ), "Prior chain coordinates shape must match structure chain apo coords shape."
        assert all(
            np.isfinite(prior_chain_coords[i]).all() for i in range(struct.num_chains)
        ), "All prior chain coordinates must be finite."

        # === 1. Prepare coordinates === #
        entity_ids: list[int] = sorted(set(chain.entity_id for chain in struct.chains))
        entity_ctypes: dict[int, C.ChainType] = {}
        entity_sizes: dict[int, int] = {}
        entity_prior_dict: dict[int, list[np.ndarray]] = defaultdict(list)
        for c_i, chain in enumerate(struct.chains):
            entity_ctypes[chain.entity_id] = chain.ctype
            entity_sizes[chain.entity_id] = chain.num_residues
            entity_prior_dict[chain.entity_id].append(prior_chain_coords[c_i])

        # Sort entities by size (largest first)
        entity_ids.sort(key=lambda x: entity_sizes[x], reverse=True)

        # Remove entities with all-missing apo coordinates
        for eid in list(entity_ids):
            prior_coords_list = entity_prior_dict[eid]
            if any(np.isnan(coords).all() for coords in prior_coords_list):
                entity_ids.remove(eid)

        # Remove entities with covalent ligands
        for chain in struct.chains:
            if chain.is_covalent_ligand:
                if chain.entity_id in entity_ids:
                    entity_ids.remove(chain.entity_id)

        if len(entity_ids) == 0:
            # No entities to permute
            return prior_chain_coords

        if all(len(entity_prior_dict[eid]) == 1 for eid in entity_ids):
            # Only one chain per entity, no permutation needed
            return prior_chain_coords

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
                idx = np.arange(0, chain.num_atoms)
                num_anchors = 2

            if len(idx) == 0:
                # Fallback to atoms (all atoms are resolved)
                idx = np.arange(0, chain.num_atoms)
                num_anchors = 2

            # Take evenly spaced anchors up to num_anchors
            stride = max(1, len(idx) // num_anchors)
            idx = idx[::stride]
            entity_anchors[eid] = idx
            entity_anchor_coords[eid] = [coords[idx] for coords in entity_prior_dict[eid]]

        # === 3. Generate candidate permutations === #
        entity_to_perms = {}
        for eid in entity_ids:
            num_chains = len(entity_anchor_coords[eid])
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
        if not label_mask.any():
            # No resolved anchor atoms in label structure
            return prior_chain_coords

        entity_order = [chain.entity_id for chain in perm_chains]

        # === 5. Evaluate permutations === #
        best_perm = None
        best_rmsd = float("inf")
        prior_centers = np.empty_like(label_centers)
        for perm in final_permutations:
            # Use a simpler way to track which index to take for each entity
            st = 0
            _perm = {eid: list(perm[eid]) for eid in entity_ids}
            for eid in entity_order:
                swap_idx = _perm[eid].pop(0)
                chain_coords = entity_anchor_coords[eid][swap_idx]
                prior_centers[st : st + len(chain_coords)] = chain_coords
                st += len(chain_coords)
            # Eigenvalue-based rmsd computation to avoid memory leakage
            rmsd = compute_rmsd(
                prior_centers, label_centers, label_mask, align=True, no_svd=True
            )
            if rmsd < best_rmsd:
                best_perm, best_rmsd = perm, rmsd

        # === 6. Apply permutations === #
        if best_perm is None:
            return prior_chain_coords

        # 1. Group original coords by entity to handle them structurally
        grouped_coords = defaultdict(list)
        for i, chain in enumerate(struct.chains):
            grouped_coords[chain.entity_id].append(prior_chain_coords[i])

        # 2. Reorder within groups based on best_perm
        permuted_groups = {}
        for eid, coords_list in grouped_coords.items():
            if eid in best_perm:
                # best_perm[eid] is tuple of indices like (1, 0, 2)
                indices = best_perm[eid]
                permuted_groups[eid] = [coords_list[i] for i in indices]
            else:
                permuted_groups[eid] = coords_list

        # 3. Flatten back to original structure order
        new_prior_chain_coords = []
        group_iterators = {eid: iter(lst) for eid, lst in permuted_groups.items()}
        for chain in struct.chains:
            # Pop the next correctly-ordered chain for this entity type
            new_prior_chain_coords.append(next(group_iterators[chain.entity_id]))

        return new_prior_chain_coords

    def find_best_residue_permutation(
        self,
        prior_coords: np.ndarray,
        ref_chain: Chain,
    ) -> None:
        """Find the best residue permutation for symmetry correction.
        Use intra-residue structure comparison to find the best permutation.

        Parameters
        ----------
        prior_coords : np.ndarray
            Prior coordinates of shape [N_atoms, 3].
        ref_chain : Chain
            Reference chain containing residue and atom information.
        """

        @lru_cache
        def get_ref_comp(res_name: str) -> Component:
            assert res_name in self.ccd, f"Residue name {res_name} not found in CCD."
            return self.ccd[res_name]

        if ref_chain.is_ion:
            # Skip ions (single atom)
            return

        ccd_sequence: list[str] = ref_chain.get_ccd_sequence()
        all_atom_names: list[str] = ref_chain.atom.name.tolist()

        for res_i in range(ref_chain.num_residues):
            res_name: str = ccd_sequence[res_i]
            atom_st: int = ref_chain.residue.atom_starts[res_i]
            atom_num: int = ref_chain.residue.num_atoms[res_i]
            atom_end: int = atom_st + atom_num

            if ref_chain.residue.is_standard[res_i]:
                # Get ambiguous atom permutations for this standard residue
                assert ref_chain.is_polymer, (
                    "Only polymer chains are supported for standard residues."
                )
                perms = get_ambiguous_atoms_in_residue(res_name, extended=False)
            elif res_name in self.ccd:
                ref_comp: Component = get_ref_comp(res_name)
                atom_names: list[str] = all_atom_names[atom_st:atom_end]
                perms = get_molecule_symmetries(ref_comp, atom_names)
            else:
                perms = None

            if perms is None or len(perms) <= 1:
                # No ambiguous atoms for this residue
                continue

            # Find the best permutation
            res_prior: np.ndarray = prior_coords[atom_st:atom_end]
            res_label: np.ndarray = ref_chain.atom.coords[atom_st:atom_end]
            res_mask: np.ndarray = np.isfinite(res_label).all(-1)

            best_perm = None
            min_rmsd = float("inf")
            for perm in perms[:10]:
                permuted_prior = res_prior[perm, :]
                rmsd = compute_rmsd(
                    permuted_prior, res_label, res_mask, align=True, no_svd=True
                )
                if rmsd < min_rmsd:
                    min_rmsd, best_perm = rmsd, perm

            if best_perm is not None:
                # Apply best permutation
                res_prior[:, :] = res_prior[best_perm, :]
            else:
                pass

        return
