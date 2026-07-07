import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import Self

import numpy as np

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.utils.simulation.bioprior import BioPriorConfig, BioPriorPerturbation
from kfold.data.utils.simulation.langevin_dynamics import LangevinDynamicsSimulator
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import compute_rmsd
from kfold.utils.misc import spawn_rng


def get_mask(coords: np.ndarray) -> np.ndarray:
    """Get mask for valid coordinates: [*, 3] -> [*]."""
    return np.isfinite(coords).all(axis=-1)


@dataclasses.dataclass(kw_only=True)
class PriorSamplerConfig:
    """Configuration for ComplexPriorSampler.

    Attributes
    ----------
    chain_translation_scale : float
        Scale of random translation augmentation for each chain (in Angstrom).
    use_ot_permutation : bool
        Whether to apply optimal transport-based permutation
    ligand_augmentation_scale : float
        Scale of random noise augmentation for ligand coordinates (in Angstrom).
    prob_perturbation : float
        Probability of applying BioPrior perturbation to protein priors.
    use_holo_if_apo_unavailable : bool
        Whether to seed missing polymer priors from holo coordinates before
        Langevin relaxation and optional BioPrior perturbation.
    bioprior : BioPriorConfig
        BioPrior perturbation configuration for protein priors.
    """

    chain_translation_scale: float = 24.0  # Angstrom
    use_ot_permutation: bool = False
    ligand_augmentation_scale: float = 0.3  # Angstrom
    prob_perturbation: float = 0.0
    use_holo_if_apo_unavailable: bool = True
    bioprior: BioPriorConfig = dataclasses.field(default_factory=BioPriorConfig)
    train: bool = False

    @classmethod
    def inference_mode(cls) -> Self:
        """Get a PriorSampler config configured for inference."""
        return cls(
            chain_translation_scale=24.0,
            use_ot_permutation=False,
            ligand_augmentation_scale=0.3,
            prob_perturbation=0.0,
            use_holo_if_apo_unavailable=True,
            train=False,
        )


class PriorSampler:
    """Class to map, populate, and augment prior structures."""

    def __init__(self, config: PriorSamplerConfig, ccd: CCD) -> None:
        self.config: PriorSamplerConfig = config
        self.logger = logging.getLogger("PriorSampler")

        self.ccd: CCD = ccd

        self.chain_translation_scale: float = config.chain_translation_scale

        self.use_ot_permutation: bool = config.use_ot_permutation

        # Ligand augmentation scale
        self.train: bool = config.train
        self.ligand_augmentation_scale: float = config.ligand_augmentation_scale
        self.use_holo_if_apo_unavailable: bool = config.use_holo_if_apo_unavailable
        self.prob_perturbation: float = config.prob_perturbation
        if not 0.0 <= self.prob_perturbation <= 1.0:
            raise ValueError(
                f"prob_perturbation must be in [0, 1], got {self.prob_perturbation}."
            )
        self.bioprior = BioPriorPerturbation(config.bioprior)

        # Langevin dynamics simulator for relaxing missing atoms
        self.langevin_simulator = LangevinDynamicsSimulator.default()

        if self.use_ot_permutation and not self.train:
            raise ValueError(
                "Optimal transport permutation should only be used during training."
            )

    @classmethod
    def inference_mode(cls, ccd: CCD) -> Self:
        """Get a PriorSampler instance configured for inference."""
        return cls(PriorSamplerConfig.inference_mode(), ccd)

    def sample(
        self,
        struct: RefStructure,
        prior_coords_dict: dict[int, np.ndarray] | None,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample prior coordinates (xT) for the given structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure containing apo coordinates.
        prior_coords_dict : dict[int, np.ndarray] | None
            Optional dictionary mapping polymer asym_id to stacked residue-order
            coordinates of shape [Nsample, L, 37/29, 3].
        num_samples : int
            Number of prior samples to generate.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        prior_coords : np.ndarray
            Sampled prior coordinates of shape [num_priors, N_atoms, 3].
        """
        # Use a separate RNG for sampling to avoid affecting global state
        rng = spawn_rng(rng)

        if num_samples <= 0:
            return np.empty((0, struct.num_atoms, 3), dtype=np.float32)

        prior_uids = self.get_chain_prior_uids(struct)
        skip_ot_permutation = len(set(prior_uids)) < len(prior_uids)
        prior_coords_list: list[np.ndarray] = []
        for _ in range(num_samples):
            # Prepare source prior coordinates for each chain.
            chain_coords_list = self.prepare_chain_prior_coords(
                struct, prior_coords_dict, rng
            )

            # Apply random augmentation to each prior rigid group.
            _chain_coords_list = self.apply_group_random_augmentation(
                chain_coords_list, prior_uids, rng
            )

            if self.use_ot_permutation and not skip_ot_permutation:
                # Optimal transport permutation during training.
                _chain_coords_list = self.match_optimal_transport_permutation(
                    _chain_coords_list, struct, rng
                )

            # Combine chains into complex coordinates
            prior_coords = np.concatenate(_chain_coords_list, axis=0)
            prior_coords_list.append(prior_coords)

        return np.stack(prior_coords_list, axis=0)

    # === Helper methods for preparing prior coordinates and sampling priors === #
    def get_chain_prior_uids(self, struct: RefStructure) -> list[int]:
        """Get prior rigid-group IDs in the same order as `struct.chains`."""
        metadata_by_asym_id = {c.asym_id: c for c in struct.metadata.chains}
        prior_uids = []
        for chain in struct.chains:
            chain_info = metadata_by_asym_id.get(chain.asym_id)
            prior_uid = chain.asym_id if chain_info is None else chain_info.prior_uid
            prior_uids.append(int(prior_uid))
        return prior_uids

    def prepare_chain_prior_coords(
        self,
        struct: RefStructure,
        prior_coords_dict: dict[int, np.ndarray] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Prepare one prior coordinate array per physical chain."""
        if prior_coords_dict is None:
            prior_coords_dict = {}

        chain_prior_coords: list[np.ndarray] = []
        polymer_asym_ids = {c.asym_id for c in struct.chains if c.is_polymer}
        unknown_keys = set(prior_coords_dict) - polymer_asym_ids
        if unknown_keys:
            raise KeyError(
                "Prior coordinates must be keyed by polymer asym_id. "
                f"Unknown keys for structure {struct.id}: {sorted(unknown_keys)}"
            )

        for c in struct.chains:
            chain_key = f"{struct.id}_{c.asym_id}"
            if c.is_polymer:
                if c.asym_id in prior_coords_dict:
                    coords = self.sample_polymer_prior_coords(
                        c, prior_coords_dict[c.asym_id], rng
                    )
                elif self.use_holo_if_apo_unavailable:
                    self.logger.warning(
                        f"{str(c.ctype)} chain {chain_key} has no prior "
                        "coordinates. Seeding from holo coordinates."
                    )
                    coords = c.atom.coords.copy()
                else:
                    raise ValueError(
                        f"{str(c.ctype)} chain {chain_key} has no prior coordinates."
                    )
            else:
                # For ligands, use ETKDG conformer.
                if c.smiles is not None:
                    assert c.num_residues == 1, (
                        "Multiple residues with SMILES not supported."
                    )
                    ref_comp = Component.from_smiles("LIG", c.smiles)
                    coords = ref_comp.get_ref_conformer(rng, self.train)
                else:
                    coords = np.full_like(c.atom.coords, np.nan)
                    ccd_sequence = c.get_ccd_sequence()
                    for res_i in range(c.num_residues):
                        res_idx = res_i + 1  # 1-based
                        code = ccd_sequence[res_i]
                        if code not in self.ccd:
                            self.logger.warning(
                                f"CCD code {code} not found for ligand chain "
                                f"{chain_key}. Filling with NaN coordinates."
                            )
                            continue
                        ref_comp = self.ccd[code]
                        ref_pos = ref_comp.get_ref_conformer(rng, self.train)
                        ref_atom_order = ref_comp.get_atom_index_map()
                        # Map reference conformer to chain's atom order
                        src_atom_indices: list[int] = []
                        dst_atom_indices: list[int] = []
                        for atom_i in c.residue.iter_residue_atoms(res_idx):
                            an = c.atom.name[atom_i]
                            if an in ref_atom_order:
                                src_atom_indices.append(ref_atom_order[an])
                                dst_atom_indices.append(atom_i)
                        coords[dst_atom_indices] = ref_pos[src_atom_indices]

            # Relax with Langevin dynamics
            coords = self.langevin_relaxation(coords, c, rng)

            if c.is_protein:
                # Apply apo perturbation with probability prob_perturbation
                if rng.random() < self.prob_perturbation:
                    coords = self.apo_perturbation(coords, c, rng)
            elif c.is_ligand:
                # Apply random noise augmentation to ligand coordinates
                scale = self.ligand_augmentation_scale
                if scale > 0.0:
                    noise = rng.normal(scale=scale, size=coords.shape)
                    coords += noise

            if coords.shape != (c.num_atoms, 3):
                if c.is_polymer:
                    raise ValueError(
                        f"Prior coordinates for chain {chain_key} have shape "
                        f"{coords.shape}, expected {(c.num_atoms, 3)}."
                    )
                # NOTE: For non-polymer chains (e.g. ligands), the number of
                # atoms can be mismatched due to different bonding.
                coords = np.zeros((c.num_atoms, 3), dtype=np.float32)

            chain_prior_coords.append(coords)

        return chain_prior_coords

    def sample_polymer_prior_coords(
        self,
        chain: Chain,
        prior_coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one stacked residue-order prior and map it to chain atom order."""
        expected_width = chain.get_polymer_residue_coord_width()
        expected_shape = (chain.num_residues, expected_width, 3)
        if prior_coords.ndim != 4 or prior_coords.shape[1:] != expected_shape:
            raise ValueError(
                f"Prior coordinates for chain {chain.asym_id} have shape "
                f"{prior_coords.shape}; expected (Nsample, {chain.num_residues}, "
                f"{expected_width}, 3)."
            )
        if prior_coords.shape[0] <= 0:
            raise ValueError(
                f"Prior coordinate stack for chain {chain.asym_id} is empty."
            )

        sample_i = int(rng.integers(0, prior_coords.shape[0]))
        residue_coords = prior_coords[sample_i]
        return chain.map_polymer_residue_coords_to_atom_coords(
            residue_coords, context="Prior coordinates"
        )

    def langevin_sampling(
        self,
        chain: Chain,
        rng: np.random.Generator,
        scale: float = 16.0,
    ) -> np.ndarray:
        """Sample coordinates for a chain using Langevin dynamics"""
        num_residues = chain.num_residues
        residue_index = np.repeat(np.arange(num_residues), chain.residue.num_atoms)
        # Start from random noise
        noise = rng.standard_normal(size=chain.atom.coords.shape, dtype=np.float32)
        noise *= scale
        # Run Langevin dynamics to sample coordinates
        return self.langevin_simulator(noise, residue_index, rng=rng)

    def langevin_relaxation(
        self,
        coords: np.ndarray,
        chain: Chain,
        rng: np.random.Generator,
        scale: float = 16.0,
    ) -> np.ndarray:
        """Relax unresolved atoms using Langevin dynamics."""
        is_resolved: np.ndarray = np.isfinite(coords).all(axis=-1)
        if is_resolved.all():
            return coords

        # Fill missing coordinates with random noise before relaxation
        random_noise = rng.standard_normal(size=coords.shape, dtype=np.float32)
        random_noise *= scale
        coords = np.where(is_resolved[:, None], coords, random_noise)

        # Run a few steps of Langevin dynamics to relax the filled coordinates
        num_residues = chain.num_residues
        residue_index = np.repeat(np.arange(num_residues), chain.residue.num_atoms)
        return self.langevin_simulator(
            coords, residue_index, rng=rng, is_constraint=is_resolved
        )

    def apo_perturbation(
        self,
        coords: np.ndarray,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply BioPrior perturbation to a protein prior in chain atom order."""
        atom37_coords = self._chain_coords_to_atom37(coords, chain)
        sequence = chain.get_sequence(map_to_standard=True)
        perturbed_atom37 = self.bioprior.run(sequence, atom37_coords, rng=rng)
        if perturbed_atom37 is None:
            return coords
        return self._atom37_to_chain_coords(perturbed_atom37, coords, chain)

    def _chain_coords_to_atom37(self, coords: np.ndarray, chain: Chain) -> np.ndarray:
        """Convert full chain atom coordinates to protein atom37 order."""
        atom_order = C.atom.protein_atom37_order
        atom37_coords = np.full((chain.num_residues, 37, 3), np.nan, dtype=np.float32)
        atom_names = chain.atom.name.tolist()
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1
            for atom_i in chain.residue.iter_residue_atoms(residue_index):
                atom37_i = atom_order.get(atom_names[atom_i])
                if atom37_i is not None:
                    atom37_coords[res_i, atom37_i] = coords[atom_i]
        return atom37_coords

    def _atom37_to_chain_coords(
        self,
        atom37_coords: np.ndarray,
        original_coords: np.ndarray,
        chain: Chain,
    ) -> np.ndarray:
        """Map perturbed protein atom37 coordinates back to chain atom order."""
        atom_order = C.atom.protein_atom37_order
        coords = original_coords.copy()
        atom_names = chain.atom.name.tolist()
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1
            for atom_i in chain.residue.iter_residue_atoms(residue_index):
                atom37_i = atom_order.get(atom_names[atom_i])
                if atom37_i is None:
                    continue
                coord = atom37_coords[res_i, atom37_i]
                if np.isfinite(coord).all():
                    coords[atom_i] = coord
        return coords

    # === Helper methods for augmentation and optimal transport permutation === #
    def apply_random_augmentation(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Augment coordinates with centering & random rotation.

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
        mask = get_mask(coords)
        if not mask.any():
            return coords
        s_trans = self.chain_translation_scale
        augmented_coords = center_random_augmentation(
            coords, mask, s_trans=s_trans, rng=rng, mask_to_zero=False
        )
        return augmented_coords

    def apply_group_random_augmentation(
        self,
        chain_coords_list: list[np.ndarray],
        prior_uids: list[int],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Apply one random augmentation per prior_uid rigid group."""
        assert len(chain_coords_list) == len(prior_uids), (
            "Number of chain coordinate arrays must match number of prior_uids."
        )

        group_to_indices: dict[int, list[int]] = defaultdict(list)
        for c_i, prior_uid in enumerate(prior_uids):
            group_to_indices[prior_uid].append(c_i)

        augmented_coords_list: list[np.ndarray] = [
            np.empty_like(coords) for coords in chain_coords_list
        ]
        for indices in group_to_indices.values():
            group_coords = [chain_coords_list[i] for i in indices]
            group_sizes = [coords.shape[0] for coords in group_coords]
            coords = np.concatenate(group_coords, axis=0)
            coords = self.apply_random_augmentation(coords, rng)
            split_coords = np.split(coords, np.cumsum(group_sizes)[:-1])
            for i, chain_coords in zip(indices, split_coords, strict=True):
                augmented_coords_list[i] = chain_coords

        return augmented_coords_list

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
        try:
            prior_coords_list = self.find_best_chain_permutation(
                prior_coords_list, struct, max_permutations=100, rng=rng
            )
        except Exception as e:
            self.logger.error(f"Failed to find best chain permutation: {e}.")
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
        label_centers_masked = label_centers[label_mask]
        label_centers_masked -= label_centers_masked.mean(axis=0)  # Center label anchors

        # === 5. Evaluate permutations === #
        entity_order = [chain.entity_id for chain in perm_chains]
        prior_centers = np.empty_like(label_centers)
        best_perm = None
        best_rmsd = float("inf")
        for perm in final_permutations:
            # Use a simpler way to track which index to take for each entity
            st = 0
            _perm = {eid: list(perm[eid]) for eid in entity_ids}
            for eid in entity_order:
                swap_idx = _perm[eid].pop(0)
                chain_coords = entity_anchor_coords[eid][swap_idx]
                prior_centers[st : st + len(chain_coords)] = chain_coords
                st += len(chain_coords)

            # if align=True, eigenvalue-based rmsd computation to avoid memory leakage
            # else, compute RMSD directly on unaligned coordinates
            rmsd = compute_rmsd(
                prior_centers[label_mask],
                label_centers_masked,
                mask=None,
                align=True,
                no_svd=True,
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
