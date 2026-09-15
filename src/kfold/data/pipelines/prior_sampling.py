import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import Self

import numpy as np

import kfold.constants as C
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
    ligand_gaussian_scale : float
        Standard deviation of centered IID Gaussian ligand coordinates (in Angstrom).
    bioprior : BioPriorConfig
        BioPrior perturbation configuration for protein prior sources.
    """

    chain_translation_scale: float = 24.0  # Angstrom
    ligand_gaussian_scale: float = 8.0  # Angstrom
    bioprior: BioPriorConfig = dataclasses.field(
        default_factory=lambda: BioPriorConfig(noise_scale=0.3, max_steps=10)
    )
    train: bool = False

    @classmethod
    def inference_mode(cls) -> Self:
        """Get a PriorSampler config configured for inference."""
        return cls(
            chain_translation_scale=24.0,
            ligand_gaussian_scale=8.0,
            bioprior=BioPriorConfig(max_steps=0),
            train=False,
        )


class PriorSampler:
    """Class to map, populate, and augment prior structures."""

    def __init__(self, config: PriorSamplerConfig) -> None:
        self.config: PriorSamplerConfig = config
        self.logger = logging.getLogger("PriorSampler")

        self.chain_translation_scale: float = config.chain_translation_scale
        self.ligand_gaussian_scale: float = config.ligand_gaussian_scale
        if config.bioprior.max_steps > 0:
            self.bioprior: BioPriorPerturbation = BioPriorPerturbation(config.bioprior)
        else:
            self.bioprior: BioPriorPerturbation = None  # type: ignore

        # Langevin dynamics simulator for relaxing missing atoms
        self.langevin_simulator = LangevinDynamicsSimulator.default()

        self.train: bool = config.train

    @classmethod
    def inference_mode(cls) -> Self:
        """Get a PriorSampler instance configured for inference."""
        return cls(PriorSamplerConfig.inference_mode())

    def __call__(
        self,
        struct: RefStructure,
        prior_candidates: dict[int, np.ndarray],
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample prior coordinates (xT) for the given structure.

        Parameters
        ----------
        struct : RefStructure
            Reference structure to sample priors for.
        prior_candidates : dict[int, np.ndarray]
            Dictionary mapping protein asym_id to candidate prior coordinates
            with shape [N, L, 37, 3]. Chains in the same prior rigid group
            must use candidate stacks with the same length and ordering.
            Nucleic-acid and ligand coordinates are sampled internally.
        num_samples : int
            Number of prior samples to generate.
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        prior_coords : np.ndarray
            Sampled prior coordinates of shape [num_priors, Natom, 3].
        """
        return self.sample(struct, prior_candidates, num_samples, rng)

    def sample(
        self,
        struct: RefStructure,
        prior_candidates: dict[int, np.ndarray],
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        # Use a separate RNG for sampling to avoid affecting global state
        rng = spawn_rng(rng)

        if num_samples <= 0:
            return np.empty((0, struct.num_atoms, 3), dtype=np.float32)

        prior_uids = self.get_chain_prior_uids(struct)
        skip_ot_permutation = len(set(prior_uids)) < len(prior_uids)

        prior_coords_list: list[np.ndarray] = []
        for _ in range(num_samples):
            selected_coords = self._select_prior_coords(
                struct, prior_candidates, prior_uids, rng
            )
            chain_coords_list = self.prepare_chain_coords(struct, selected_coords, rng)

            # Apply one random augmentation per prior rigid group.
            _chain_coords_list = self.apply_group_random_augmentation(
                chain_coords_list, prior_uids, rng
            )

            if self.train and not skip_ot_permutation:
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

    def _select_prior_coords(
        self,
        struct: RefStructure,
        prior_candidates: dict[int, np.ndarray],
        prior_uids: list[int],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Select one prior candidate for each prior rigid group."""
        protein_chains = [chain for chain in struct.chains if chain.is_protein]
        protein_asym_ids = {chain.asym_id for chain in protein_chains}
        unknown_keys = set(prior_candidates) - protein_asym_ids
        missing_keys = protein_asym_ids - set(prior_candidates)
        if unknown_keys or missing_keys:
            raise KeyError(
                f"Protein prior candidates for {struct.id} must match its protein "
                f"chains; missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}."
            )

        uid_by_asym_id = {
            chain.asym_id: prior_uids[i] for i, chain in enumerate(struct.chains)
        }
        chains_by_uid: dict[int, list[Chain]] = defaultdict(list)
        for chain in protein_chains:
            chains_by_uid[uid_by_asym_id[chain.asym_id]].append(chain)

        selected: dict[int, np.ndarray] = {}
        for chains in chains_by_uid.values():
            first = chains[0]
            first_candidates = prior_candidates[first.asym_id]
            expected_shape = (first.num_residues, 37, 3)
            if (
                first_candidates.ndim != 4
                or first_candidates.shape[0] == 0
                or first_candidates.shape[1:] != expected_shape
            ):
                raise ValueError(
                    f"Prior candidates for {struct.id}:{first.asym_id} have shape "
                    f"{first_candidates.shape}; expected (N, {expected_shape[0]}, 37, 3) "
                    "with N > 0."
                )

            candidate_i = int(rng.integers(first_candidates.shape[0]))
            for chain in chains:
                candidates = prior_candidates[chain.asym_id]
                expected_shape = (chain.num_residues, 37, 3)
                if (
                    candidates.ndim != 4
                    or candidates.shape[0] != first_candidates.shape[0]
                    or candidates.shape[1:] != expected_shape
                ):
                    raise ValueError(
                        f"Prior candidates for {struct.id}:{chain.asym_id} have shape "
                        f"{candidates.shape}; expected ({first_candidates.shape[0]}, "
                        f"{expected_shape[0]}, 37, 3)."
                    )
                selected[chain.asym_id] = candidates[candidate_i]

        return selected

    def prepare_chain_coords(
        self,
        struct: RefStructure,
        selected_coords: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Prepare one atom-order prior source for each chain."""
        source_chain_ids = {chain.asym_id for chain in struct.chains if chain.is_protein}
        unknown_keys = set(selected_coords) - source_chain_ids
        if unknown_keys:
            raise KeyError(
                "Protein prior coordinates must be keyed by asym_id. "
                f"Unknown keys for structure {struct.id}: {sorted(unknown_keys)}"
            )

        chain_coords: list[np.ndarray] = []
        for c in struct.chains:
            chain_key = f"{struct.id}:{c.asym_id}"

            if c.is_protein:
                if c.asym_id not in selected_coords:
                    raise KeyError(
                        f"Missing protein prior source for chain {c.asym_id} "
                        f"in {struct.id}"
                    )
                coords = self.perturb_protein_apo_coords(
                    c, selected_coords[c.asym_id], rng
                )
                coords = c.map_residue_coords_to_atom_coords(coords)
                coords = self.langevin_relaxation(coords, c, rng)
            elif c.is_nucleic_acid:
                coords = self._sample_nucleic_acid_coords(c, rng)
            elif c.is_ligand:
                coords = self._sample_ligand_gaussian_coords(c, rng)
            else:
                raise ValueError(f"Unsupported chain type for {chain_key}.")

            if coords.shape != (c.num_atoms, 3):
                raise ValueError(
                    f"Prior coordinates for chain {chain_key} have shape "
                    f"{coords.shape}, expected {(c.num_atoms, 3)}."
                )

            chain_coords.append(coords)

        return chain_coords

    def _sample_nucleic_acid_coords(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a 16 A Gaussian initialization followed by polymer Langevin."""
        assert chain.is_nucleic_acid
        coords = rng.standard_normal(
            size=(chain.num_atoms, 3),
            dtype=np.float32,
        )
        coords *= np.float32(16.0)
        residue_index = np.repeat(
            np.arange(chain.num_residues),
            chain.residue.num_atoms,
        )
        return self.langevin_simulator(coords, residue_index, rng=rng)

    def _sample_ligand_gaussian_coords(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample centered IID Gaussian ligand coordinates."""
        assert chain.is_ligand
        coords = rng.standard_normal(
            size=(chain.num_atoms, 3),
            dtype=np.float32,
        )
        coords *= np.float32(self.ligand_gaussian_scale)
        coords -= coords.mean(axis=0, keepdims=True)
        return coords

    def perturb_protein_apo_coords(
        self,
        chain: Chain,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply BioPrior perturbation to one protein prior source."""
        assert chain.is_protein
        mask = np.isfinite(coords).all(axis=-1)
        if not mask.any():
            return coords
        if self.bioprior is None:
            return coords
        sequence = chain.get_sequence(map_to_standard=True)
        perturbed = self.bioprior.run(sequence, coords, rng=rng)
        if perturbed is None:
            perturbed = coords.copy()  # Fallback to original coordinates
        perturbed[~mask] = np.nan
        return perturbed

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
