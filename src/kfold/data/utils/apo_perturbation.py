import itertools
from collections import defaultdict

import numpy as np

import kfold.constants as C
from kfold.data.structure import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import weighted_rigid_align


class ApoPerturbation:
    """Class to handle apo structure perturbation."""

    def __init__(
        self,
        use_perturbation: bool = False,
        use_random_rotation: bool = False,
        use_symmetry_correction: bool = False,
        mask_nucleic_acid: bool = False,
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
            NOTE: This is used during training only (both training/validation).
        mask_nucleic_acid : bool, optional
            Whether to mask nucleic acid chains during perturbation.
            TODO: (SeonghwanSeo) Remove this argument after DNA/RNA apo coordinates
            is prepared.
        """

        self.use_perturbation: bool = use_perturbation
        self.use_random_rotation: bool = use_random_rotation
        self.use_symmetry_correction: bool = use_symmetry_correction
        self.mask_nucleic_acid: bool = mask_nucleic_acid

    def augment_structure(
        self, struct: TokenizedStructure, rng: np.random.Generator
    ) -> TokenizedStructure:
        """Augment apo structure coordinates.

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
        # Get coordinates and masks for all apo structures
        all_apo_coords = struct.atom.apo_coords  # [Ntoken, 24, Napo, 3]
        all_apo_mask = struct.atom.apo_mask  # [Ntoken, 24, Napo]

        num_apos = all_apo_coords.shape[2]
        assert num_apos >= 1, "Apo structure must have at least one apo conformation."

        # Synchronized apo structures for entities
        entity_apo_coords: dict[int, np.ndarray] = {}
        entity_apo_masks: dict[int, np.ndarray] = {}

        # Process each chain
        chain_apo_coords_list: list[np.ndarray] = []
        chain_apo_mask_list: list[np.ndarray] = []
        for chain_i in range(struct.num_chains):
            entity_id = struct.chain.entity_id[chain_i]

            # synchronized apo structures for the same entity
            if entity_id in entity_apo_coords:
                chain_apo_coords_list.append(entity_apo_coords[entity_id])
                chain_apo_mask_list.append(entity_apo_masks[entity_id])
                continue

            # Slice indices for the current chain
            token_st: int = int(struct.chain.token_starts[chain_i])
            token_num: int = int(struct.chain.num_tokens[chain_i])
            token_end: int = token_st + token_num
            all_chain_apo_coords = all_apo_coords[token_st:token_end]
            all_chain_apo_mask = all_apo_mask[token_st:token_end]

            # TODO: (SeonghwanSeo) Remove this after NA apo prediction is prepared.
            if self.mask_nucleic_acid:
                # Mask out nucleic acid apo structures
                chain_type = C.ChainType(struct.chain.chain_type[chain_i])
                if chain_type in (C.ChainType.DNA, C.ChainType.RNA):
                    all_chain_apo_mask = np.zeros_like(all_chain_apo_mask)

            # Sample one apo structure
            chain_apo_coords, chain_apo_mask = self.sample_apo_structure(
                all_chain_apo_coords, all_chain_apo_mask, rng
            )  # [Ntoken', 24, 3], [Ntoken', 24], bool

            # Apply apo perturbation
            if self.use_perturbation:
                chain_type = C.ChainType(struct.chain.chain_type[chain_i])
                chain_apo_coords = self.apply_perturbation(
                    chain_apo_coords, chain_apo_mask, chain_type, rng
                )

            # Store for synchronized apo structures
            entity_apo_coords[entity_id] = chain_apo_coords
            entity_apo_masks[entity_id] = chain_apo_mask

            chain_apo_coords_list.append(chain_apo_coords)
            chain_apo_mask_list.append(chain_apo_mask)

        # Apply random rotation augmentation
        if self.use_random_rotation:
            chain_apo_coords_list = [
                self.apply_random_rotation(
                    chain_apo_coords_list[chain_i], chain_apo_mask_list[chain_i], rng
                )
                for chain_i in range(struct.num_chains)
            ]

        apo_coords: np.ndarray = np.concatenate(chain_apo_coords_list, axis=0)
        apo_mask: np.ndarray = np.concatenate(chain_apo_mask_list, axis=0)

        # If symmetry correction is enabled, align apo to holo
        if self.use_symmetry_correction:
            apo_coords = self.align_apo_to_holo(struct, apo_coords, apo_mask, rng)

        # Create new atom structure
        num_tokens = struct.num_tokens
        atom_struct = struct.atom.copy_with(
            apo_coords=apo_coords.reshape(num_tokens, 24, 1, 3),
            apo_mask=apo_mask.reshape(num_tokens, 24, 1),
        )
        return struct.copy_with(atom=atom_struct)

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

    # === Apo perturbation / augmentation === #

    def apply_perturbation(
        self,
        apo_coords: np.ndarray,
        mask: np.ndarray,
        chain_type: C.ChainType,
        rng: np.random.Generator,
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

        Returns
        -------
        perturbed_apo_coords : np.ndarray
            Perturbed apo structure coordinates of shape [Ntoken, 24, 3].
        """
        # TODO: Implement specific perturbation logic here.
        raise NotImplementedError("Apo perturbation logic is not implemented yet.")

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
        augmented_apo_coords : np.ndarray
            Augmented apo structure coordinates of shape [Ntoken, 24, 3].
        """
        if not np.any(mask):
            # If there are no valid atoms, skip augmentation.
            return apo_coords
        return center_random_augmentation(apo_coords, mask, rng=rng)

    # === Symmetry correction === #

    def align_apo_to_holo(
        self,
        struct: TokenizedStructure,
        apo_coords: np.ndarray,
        apo_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
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
        pmeruted_apo_mask : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        """
        return self.get_best_chain_permutation(struct, apo_coords, apo_mask, 100, rng=rng)

    def get_best_chain_permutation(
        self,
        struct: TokenizedStructure,
        apo_coords: np.ndarray,
        apo_mask: np.ndarray,
        max_permutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Find the best chain permutation for symmetry correction.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        apo_coords : np.ndarray
            Apo structure coordinates. (Ntoke', 24, 3)
        apo_mask : np.ndarray
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
        if not np.any(apo_mask):
            # If there are no valid apo atoms, skip permutation.
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

        # === Prepare holo center coordinates and masks === #
        # Use the first bioassembly for symmetry correction
        holo_coords: np.ndarray = struct.atom.coords  # [Ntoken, 24, Nholo, 3]
        holo_mask: np.ndarray = struct.atom.resolved_mask  # [Ntoken, 24]
        if holo_coords.shape[-2] != 1:
            raise NotImplementedError(
                "Symmetry correction for multiple bioassemblies is not implemented."
            )
        target_coords: np.ndarray = holo_coords[..., 0, :]  # [Ntoken, 24, 3]
        target_mask: np.ndarray = holo_mask  # [Ntoken, 24]
        del holo_coords, holo_mask

        # === Extract center atom coordinates and resolved masks === #
        # Center atom is CA for protein, C1' for nucleic acid, and centroid for ligand
        token_index = struct.token.token_index  # [Ntoken]
        center_index = struct.token.center_index  # [Ntoken]
        target_coords = target_coords[token_index, center_index]  # [Ntoken, 3]
        target_mask = target_mask[token_index, center_index]  # [Ntoken]
        apo_coords = apo_coords[token_index, center_index]  # [Ntoken, 3]
        apo_mask = apo_mask[token_index, center_index]  # [Ntoken]

        # NOTE: apo mask is not changed during permutation since they are synchronized
        align_mask = target_mask & apo_mask

        # === Sample anchor tokens for alignment === #
        # To reduce computation, use a subset of tokens as anchors
        chain_anchor_tokens: list[np.ndarray] = []
        chain_anchor_weights: list[np.ndarray] = []
        entity_anchor_tokens: dict[int, np.ndarray] = {}
        for chain_i in range(struct.num_chains):
            entity_id = chain_entity_ids[chain_i]
            token_st: int = int(struct.chain.token_starts[chain_i])
            token_num: int = int(struct.chain.num_tokens[chain_i])
            num_anchors = min(10, token_num)
            if entity_id in entity_anchor_tokens:
                # synchronized anchors
                anchor_tokens = entity_anchor_tokens[entity_id]
            else:
                stride = max(1, token_num // num_anchors)
                anchor_tokens = np.arange(0, token_num, stride, dtype=np.int32)
                if entity_id in entities_with_symmetry:
                    # Only store for entities with symmetry
                    entity_anchor_tokens[entity_id] = anchor_tokens
            weights = np.full((num_anchors,), token_num / num_anchors, dtype=np.float32)
            chain_anchor_tokens.append(anchor_tokens + token_st)
            chain_anchor_weights.append(weights)
        del entity_anchor_tokens

        # === Get static variables for alignment === #
        total_anchors = np.concatenate(
            [chain_anchor_tokens[chain_i] for chain_i in range(struct.num_chains)]
        )  # [Nanchor,]
        target_coords = target_coords[total_anchors]  # [Nanchor, 3]
        align_mask = align_mask[total_anchors]  # [Nanchor,]
        align_weights = np.concatenate(chain_anchor_weights, axis=0)
        align_weights[~align_mask] = 0.0

        # === Collect possible permutations === #
        original_perm: list[int] = list(range(num_chains))
        permutations: list[list[int]] = [original_perm]

        # Limit the number of permutations to avoid combinatorial explosion
        max_candidates = max_permutations * 10
        for entity_id, chains in entity_chains.items():  # noqa: B007
            chain_permutations = list(itertools.permutations(chains, r=len(chains)))
            new_permutations: list[list[int]] = []
            for base_perm in permutations:
                for chain_perm in chain_permutations:
                    new_perm = base_perm.copy()
                    for idx, chain_i in enumerate(chains):
                        new_perm[chain_i] = chain_perm[idx]
                    new_permutations.append(new_perm)
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
            indices = rng.choice(len(permutations), size=max_permutations).tolist()
            permutations = [original_perm] + [permutations[i] for i in indices]

        # Include the original permutation at the beginning
        original_permutation = list()
        if original_permutation in permutations:
            permutations.remove(original_permutation)
        permutations = [original_permutation] + permutations

        # === Find best permutation with minimum weighted RMSD === #
        best_permutation: list[int] = list(range(num_chains))
        min_weighted_mse: float = float("inf")
        for perm in permutations:
            # Get permutaed anchor indices
            permuted_anchors = np.concatenate(
                [chain_anchor_tokens[perm[chain_i]] for chain_i in range(num_chains)]
            )  # [Nanchor,]

            # Get apo coordinates with the current permutation
            permuted_apo_coords = apo_coords[permuted_anchors]  # [Nanchor, 3]

            # Align permuted apo to holo
            permuted_apo_coords = weighted_rigid_align(
                permuted_apo_coords, target_coords, align_mask, align_weights
            )

            # Compute weighted RMSD
            d = np.linalg.norm(permuted_apo_coords - target_coords, axis=-1)  # [Ntoken]
            w = align_weights  # already masked
            weighted_mse = np.sum((d**2) * w) / (w.sum() + 1e-8)
            del permuted_apo_coords, d, w

            if weighted_mse < min_weighted_mse:
                min_weighted_mse = weighted_mse
                best_permutation = perm

        # === Apply best permutation to all apo coordinates === #
        chain_apo_coords: list[np.ndarray] = []
        for permuted_chain_i in best_permutation:
            token_st: int = int(struct.chain.token_starts[permuted_chain_i])
            token_num: int = int(struct.chain.num_tokens[permuted_chain_i])
            token_end: int = token_st + token_num
            chain_apo_coords.append(apo_coords[token_st:token_end])

        permuted_apo_coords = np.concatenate(chain_apo_coords, axis=0)  # [Ntoken, 3]
        return permuted_apo_coords
