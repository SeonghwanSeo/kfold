# started from code from https://github.com/jwohlwend/boltz, MIT License
import numpy as np

from kfold.data.structure import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from . import utils
from .base import BaseCropper


@DATA_CROPPER.register()
class BoltzCropper(BaseCropper):
    """Unified cropper used in Boltz1"""

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
        self.neighborhood_sizes: list[int] = sizes

    def get_token_indices(  # noqa: PLR0915
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None, optional
            The chain ID(s) to center the crop on. If None, a random chain
            or interface will be selected.
        rng : np.random.Generator
            The random number generator

        Returns
        -------
        token_indices: np.ndarray
            The selected token indices.
        """
        token_data = struct.token  # features: [L, ...]
        atom_data = struct.atom  # features: [L, 24, ...]

        # Get resolved mask with valid center atoms
        token_center_mask = atom_data.resolved_mask[
            token_data.token_index, token_data.center_index
        ]  # (num_tokens,)
        resolved_mask = token_data.resolved_mask & token_center_mask

        if not resolved_mask.any():
            raise ValueError("No valid tokens in struct")

        # Frequently used variables
        all_tokens = token_data.token_index
        valid_tokens = all_tokens[resolved_mask]
        all_asym_ids = token_data.asym_id
        all_residue_indices = token_data.residue_index
        # NOTE: (seonghwanseo) Here we use the first holo coordinates.
        all_token_centers = atom_data.coords[
            token_data.token_index, token_data.center_index, 0, :
        ]  # (num_tokens, 3)

        # Randomly select a neighborhood size
        neighborhood_size = utils.random_choice(self.neighborhood_sizes, rng=rng)

        # Pick a random token, chain, or interface
        if bias_asym_id is None:
            valid_chain_asym_ids = np.unique(all_asym_ids[valid_tokens])
            chain_id = rng.choice(valid_chain_asym_ids)
            query = utils.pick_token(struct, chain_id, resolved_mask, rng)
        elif isinstance(bias_asym_id, int):
            chain_id = bias_asym_id
            query = utils.pick_token(struct, chain_id, resolved_mask, rng)
        else:  # tuple[int, int]
            interface_id = bias_asym_id
            query = utils.pick_token(struct, interface_id, resolved_mask, rng)

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
            chain_tokens = all_tokens[chain_mask]

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
