import itertools
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np

import kfold.constants as C
from kfold.data.structure import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.geometry.rigid_align import compute_rmsd, weighted_rigid_align


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
            apo_coords = self.align_apo_to_holo(struct, apo_coords, apo_mask, rng)

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
        # HACK: Sine some chains with the same entity id have different number of
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
                    chain_apo_coords, chain_apo_mask, chain_type, rng
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

        # Apply random rotation augmentation
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
        if not self.use_perturbation:
            return apo_coords
        if rng.random() > self.prob_perturbation:
            return apo_coords
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

        # NOTE: apo mask is not changed during permutation since they are synchronized
        align_mask = apo_mask & holo_mask  # [Ntoken, 24]

        if not np.any(align_mask):
            # If there are no valid apo/holo atoms to align, skip permutation.
            return apo_coords

        # Safe in-place modification
        apo_coords = apo_coords.copy()

        # First, correct chain-level symmetry
        apo_coords = self.get_best_chain_permutation(
            struct, holo_coords, apo_coords, align_mask, max_permutations=100, rng=rng
        )
        # Second, correct residue-level symmetry
        apo_coords = self.get_best_residue_permutation(
            struct, holo_coords, apo_coords, align_mask, rng=rng
        )
        # Third, correct molecule-level symmetry
        apo_coords = self.get_best_mol_permutation(
            struct, holo_coords, apo_coords, align_mask, rng=rng
        )
        return apo_coords

    def get_best_chain_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        align_mask: np.ndarray,
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
        align_mask : np.ndarray
            Mask for alignment. (Ntoken, 24)
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
            # Get permutaed anchor indices
            permuted_anchors = np.concatenate(
                [chain_anchor_tokens[perm[chain_i]] for chain_i in range(num_chains)]
            )  # [Nanchor,]

            # Get apo coordinates with the current permutation
            permuted_apo_centers = apo_centers[permuted_anchors]  # [Nanchor, 3]

            # Align permuted apo to holo
            permuted_apo_centers = weighted_rigid_align(
                permuted_apo_centers, holo_centers, center_mask, align_weights
            )

            # Compute weighted RMSD
            d = np.linalg.norm(permuted_apo_centers - holo_centers, axis=-1)  # [Ntoken,]
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

        permuted_apo_coords = np.concatenate(chain_apo_coords, axis=0)  # [Ntoken, 3]
        return permuted_apo_coords

    def get_best_residue_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        align_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
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
        align_mask : np.ndarray
            Mask for alignment. (Ntoken, 24)
        rng : np.random.Generator
            Random number generator for stochastic operations.

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
            res_mask = align_mask[i, :num_atoms]  # [num_res_atoms,]
            if res_mask.sum() < 4:
                # Not enough resolved atoms to align
                continue

            # Find the best permutation
            best_perm = list(range(num_atoms))
            min_rmsd = float("inf")

            for perm in perms:
                permuted_apo_coords = res_apo_coords[perm, :]
                rmsd = compute_rmsd(
                    permuted_apo_coords, res_holo_coords, res_mask, align=True
                )
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    best_perm = perm

            # Apply best permutation
            apo_coords[i, :num_atoms, :] = res_apo_coords[best_perm, :]

        return apo_coords

    def get_best_mol_permutation(
        self,
        struct: TokenizedStructure,
        holo_coords: np.ndarray,
        apo_coords: np.ndarray,
        align_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Find the best residue permutation for symmetry correction.
        Use intra-molecule structure comparison to find the best permutation.

        Parameters
        ----------
        struct : TokenizedStructure
            Tokenized structure containing holo coordinates.
        holo_coords : np.ndarray
            Holo structure coordinates. (Ntoken, 24, 3)
        apo_coords : np.ndarray
            Apo structure coordinates. (Ntoke', 24, 3)
        align_mask : np.ndarray
            Mask for alignment. (Ntoken, 24)
        rng : np.random.Generator
            Random number generator for stochastic operations.

        Returns
        -------
        permuted_apo_coords : np.ndarray
            Permuted apo structure coordinates of shape [Ntoken, 24, 3].
        """
        # type alias
        ResUID = tuple[int, int]  # (asym_id, res_idx)

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
            mol_mask = align_mask[token_st:token_end, 0]  # [num_mol_atoms,]
            if mol_mask.sum() < 4:
                # Not enough resolved atoms to align
                continue

            # Find the best permutation
            best_perm = list(range(num_atoms))
            min_rmsd = float("inf")

            for perm in permutations:
                permuted_apo_coords = mol_apo_coords[perm, :]
                rmsd = compute_rmsd(
                    permuted_apo_coords, mol_holo_coords, mol_mask, align=True
                )
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    best_perm = perm

            # Apply best permutation
            apo_coords[token_st:token_end, 0, :] = mol_apo_coords[best_perm, :]
        return apo_coords
