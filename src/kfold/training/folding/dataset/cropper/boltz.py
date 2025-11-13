import numpy as np
from scipy.spatial.distance import cdist

import kfold.constants as C
from kfold.data import tokenized
from kfold.utils.registry import DATA_CROPPER

from .base import BaseCropper


def pick_chain_token(
    structure: tokenized.TokenizedStructure,
    asym_id: int,
) -> int:
    """Pick a random token from a chain.

    Parameters
    ----------
    structure : TokenizedStructure
        The tokenized data.
    chain_id : int
        The chain ID.

    Returns
    -------
    token_index : int
        The selected token index.
    """
    # Filter to chain
    tokens = structure.token
    chain_mask = tokens.asym_id == asym_id

    # Pick from chain, fallback to all tokens
    token_indices = tokens.token_index
    chain_tokens = token_indices[chain_mask & tokens.resolved_mask]
    if chain_tokens.size:
        return np.random.choice(chain_tokens)
    else:
        valid_tokens = token_indices[tokens.resolved_mask]
        return np.random.choice(valid_tokens)


def pick_interface_token(
    structure: tokenized.TokenizedStructure,
    asym_ids: tuple[int, ...],
    center_coords: np.ndarray,
) -> int:
    """Pick a random token from an interface.

    Parameters
    ----------
    structure : TokenizedStructure
        The tokenized data.
    asym_ids : tuple[int, ...]
        The chain IDs defining the interface.
    center_coords : np.ndarray
        The center coordinates of all tokens.

    Returns
    -------
    token_index : int
        The selected token index.
    """

    # Sample random interface
    if len(asym_ids) != 2:
        raise ValueError("asym_ids must have length 2 for interface picking")

    chain_1, chain_2 = asym_ids

    token_indices = structure.token.token_index

    tokens_1 = token_indices[
        (structure.token.asym_id == chain_1) & structure.token.resolved_mask
    ]

    tokens_2 = token_indices[
        (structure.token.asym_id == chain_2) & structure.token.resolved_mask
    ]

    # If no interface, pick from the chains
    if tokens_1.size and (not tokens_2.size):
        return np.random.choice(tokens_1)
    elif tokens_2.size and (not tokens_1.size):
        return np.random.choice(tokens_2)
    elif (not tokens_1.size) and (not tokens_2.size):
        # Fallback to all tokens
        valid_tokens = token_indices[structure.token.resolved_mask]
        return np.random.choice(valid_tokens)
    else:
        # If we have tokens, compute distances to find interface tokens
        tokens_1_coords = center_coords[tokens_1]  # (num_tokens_1, 3)
        tokens_2_coords = center_coords[tokens_2]  # (num_tokens_2, 3)

        dists = cdist(tokens_1_coords, tokens_2_coords)
        cuttoff = dists < C.INTERFACE_CUTOFF

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (C.INTERFACE_CUTOFF + 5.0)

        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Select random token
        candidates = np.concatenate([tokens_1, tokens_2])
        return np.random.choice(candidates)


@DATA_CROPPER.register()
class BoltzCropper(BaseCropper):
    """Interpolate between contiguous and spatial crops."""

    class Config(BaseCropper.Config):
        """Configuration for the BoltzCropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.
        """

        min_neighborhood: int = 0
        max_neighborhood: int = 40

    def __init__(self, config: Config):
        sizes = list(range(config.min_neighborhood, config.max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def get_token_indices(  # noqa: PLR0915
        self,
        structure: tokenized.TokenizedStructure,
        max_tokens: int,
        asym_ids: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        structure : TokenizedStructure
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None
            The chain IDs to center the crop on. If None, a random chain

        Returns
        -------
        token_indices: np.ndarray
            The selected token indices.
        """

        token_data = structure.token  # features: [L, ...]
        atom_data = structure.atom  # features: [L, 24, ...]

        # Check inputs
        resolved_mask = token_data.resolved_mask
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # Frequently used variables
        all_tokens = token_data.token_index
        valid_tokens = all_tokens[resolved_mask]
        all_asym_ids = token_data.asym_id
        all_residue_indices = token_data.residue_index
        # NOTE: (seonghwanseo) Here we use the first holo coordinates.
        all_token_centers = atom_data.label_coords[
            token_data.token_index, token_data.center_index, 0, :
        ]  # (num_tokens, 3)

        # Randomly select a neighborhood size
        neighborhood_size = np.random.choice(self.neighborhood_sizes)

        # Pick a random token, chain, or interface
        if asym_ids is None:
            valid_chain_asym_ids = np.unique(all_asym_ids[valid_tokens])
            asym_id = np.random.choice(valid_chain_asym_ids)
            query = pick_chain_token(structure, asym_id)
        elif len(asym_ids) == 1:
            query = pick_chain_token(structure, asym_ids[0])
        elif len(asym_ids) == 2:
            query = pick_interface_token(structure, asym_ids, all_token_centers)
        else:
            raise ValueError("asym_ids must be None, length 1, or length 2")

        query_coords = all_token_centers[query]  # [3,]
        valid_coords = all_token_centers[valid_tokens]  # [num_valid_tokens, 3]

        # Sort all tokens by distance to query_coords
        dists = valid_coords - query_coords  # [num_valid_tokens, 3]
        indices = np.argsort(np.linalg.norm(dists, axis=1))
        neighbor_indices = valid_tokens[indices]

        # Select cropped indices
        cropped: set[int] = set()
        for token_idx in neighbor_indices:
            # Get the token
            asym_id = all_asym_ids[token_idx]
            residue_idx = all_residue_indices[token_idx]

            # Get all tokens from this chain
            chain_mask = all_asym_ids == asym_id
            chain_tokens = all_tokens[chain_mask & resolved_mask]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= neighborhood_size:
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighborhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                min_idx = residue_idx - neighborhood_size
                max_idx = residue_idx + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[
                    (all_residue_indices[max_token_set] >= min_idx)
                ]
                max_token_set = max_token_set[
                    (all_residue_indices[max_token_set] <= max_idx)
                ]

                # Start by adding just the query token
                new_tokens = max_token_set[
                    all_residue_indices[max_token_set] == residue_idx
                ]

                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                min_idx = max_idx = residue_idx
                old_size = -1
                while new_tokens.size < neighborhood_size and (
                    new_tokens.size != old_size
                ):
                    old_size = new_tokens.size
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    new_tokens = max_token_set
                    new_tokens = new_tokens[all_residue_indices[new_tokens] >= min_idx]
                    new_tokens = new_tokens[all_residue_indices[new_tokens] <= max_idx]

            # Compute new tokens and new atoms
            new_tokens = set(new_tokens) - cropped

            # Stop if we exceed the max number of tokens or atoms
            if len(new_tokens) > (max_tokens - len(cropped)):
                break

            # Add new indices
            cropped.update(new_tokens)

        return np.array(sorted(cropped))
