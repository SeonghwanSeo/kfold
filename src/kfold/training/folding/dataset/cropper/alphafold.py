"""AlphaFold-Multimer/AlphaFold-3 style cropping.

Implements three kinds of cropping strategies:
- Contiguous: crops a contiguous regions of tokens from a single chain.
- Spatial: crops a region of tokens centered on a random token.
- Spatial-interface: crops a region of tokens centered on a random interface between
    two chains.

NOTE: (pminha01) I understood the cropping logic as follows:
Input: a TokenizedStructure object, max_tokens, asym_ids
    (None for random, length 1 for single chain, length 2 for interface)
Output: a numpy array of token indices to include in the crop.
 1. Randomly select a cropping strategy (based on the weights created in the Config)
 2. If the strategy is contiguous, then crop a contiguous region of tokens from a single
    chain. Repeat for all chains until max_tokens is reached.
 3. If the strategy is spatial, then crop a region of tokens centered on a token.
    If asym_ids is provided, then the crop is centered on the specified chain. Else, a
    random token is selected from a random chain.
 4. If the strategy is spatial-interface, then crop a region of tokens centered on
    an interface-representing token between two chains. If asym_ids is provided,
    then the crop is centered on the specified interface or specified chain. Else, a
    random interface is selected from a random interface between two chains.

NOTE: (pminha01) This relies heavily on the fact that the TokenizedStructure object
contains a valid metadata object. If such an object does not exist, then any crop
involving a random interface (spatial-interface cropping) will default to a random center.
"""

import numpy as np

from kfold.data.structure import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from .base import BaseCropper
from .boltz import pick_chain_token, pick_interface_token


def get_random_interface(
    structure: TokenizedStructure, chain_id: int | None
) -> tuple[int, int] | None:
    """Get a random interface from the metadata.

    Parameters
    ----------
    structure: TokenizedStructure
        The tokenized structure.
    chain_id: int | None
        The chain ID to center the crop on. If None, a random interface is selected.

    Returns
    ----------
    interface_asym_ids: tuple[int, int] | None
        The asymmetric unit IDs of the interface or None if no valid interface is found.
    """
    if structure.metadata is None:
        return None
    if structure.metadata.interfaces is None or len(structure.metadata.interfaces) == 0:
        return None

    # any random interface
    if chain_id is None:
        interface = np.random.choice(structure.metadata.interfaces)
        return interface.asym_ids

    # random interface that includes chain_id
    valid_interfaces = [
        i for i in structure.metadata.interfaces if chain_id in i.asym_ids
    ]
    if valid_interfaces:
        interface = np.random.choice(valid_interfaces)
        return interface.asym_ids

    # should never reach here
    return None


@DATA_CROPPER.register()
class AlphaFold3Cropper(BaseCropper):
    """Perform AlphaFold-style cropping by selecting between
    contiguous, spatial, and spatial-interface cropping strategies.
    """

    class Config(BaseCropper.Config):
        """Configuration for AlphaFoldCropper.

        Attributes
        ----------
        contiguous: float
            The weight for contiguous cropping.
        spatial: float
            The weight for spatial cropping.
        spatial_interface: float
            The weight for spatial-interface cropping.
        max_chains: int
            The maximum number of chains to consider during contiguous cropping.
        """

        contiguous: float = 0.2
        spatial: float = 0.4
        spatial_interface: float = 0.4
        max_chains: int = 20

    def __init__(self, config: Config):
        self.contiguous: float = config.contiguous
        self.spatial: float = config.spatial
        self.spatial_interface: float = config.spatial_interface
        self.max_chains: int = config.max_chains
        if self.contiguous + self.spatial + self.spatial_interface != 1.0:
            raise ValueError("Cropping strategy weights must sum to 1.0")

    def get_token_indices(  # noqa: PLR0915
        self,
        structure: TokenizedStructure,
        max_tokens: int,
        asym_ids: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Get the token indices to include in the crop.

        Parameters
        ----------
        structure: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.
        asym_ids: tuple[int, ...] | None
            The chain IDs to center the crop on.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """
        # random choice on cropping strategy
        strategy = np.random.choice(
            ["contiguous", "spatial", "spatial_interface"],
            p=[self.contiguous, self.spatial, self.spatial_interface],
        )
        if strategy == "contiguous":
            return self.get_contiguous_token_indices(structure, max_tokens)
        if strategy == "spatial":
            query = self.pick_spatial_crop_query(structure, asym_ids)
        else:
            query = self.pick_spatial_interface_crop_query(structure, asym_ids)
        return self.get_closest_tokens(structure, max_tokens, query)

    def get_contiguous_token_indices(
        self,
        structure: TokenizedStructure,
        max_tokens: int,
    ) -> np.ndarray:
        """Get the token indices to include in the contiguous crop.

        Follows AlphaFold-Multimer SI Algorithm 1 with modification
        in initializing n_remaining. Modifications are in line with
        OpenFold3's implementation.

        Parameters
        ----------
        structure: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """

        all_tokens = structure.token.token_index  # =np.arange(num_tokens)

        # randomly shuffle chains
        chain_ids = np.random.permutation(structure.chain.asym_id)

        # Initialize counters; n_remaining is the sum of all tokens
        n_added = 0
        n_remaining = np.sum(np.isin(structure.token.asym_id, chain_ids))
        cropped: set[int] = set()

        # iterate over chains
        for chain_id in chain_ids:
            # get chain length as number of tokens
            chain_mask = structure.token.asym_id == chain_id
            chain_tokens = all_tokens[chain_mask]
            chain_length = len(chain_tokens)
            n_remaining -= chain_length

            # sample crop length and start
            crop_size_max = min(max_tokens - n_added, chain_length)
            crop_size_min = min(chain_length, max(0, max_tokens - n_added - n_remaining))
            crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
            crop_start = np.random.randint(0, chain_length - crop_size + 1)
            n_added += crop_size

            # get token indices in crop
            crop_tokens = chain_tokens[crop_start : crop_start + crop_size]

            # slice using sampled crop start and length for this chain
            cropped.update(crop_tokens.tolist())

        return np.array(sorted(cropped))

    def pick_spatial_crop_query(
        self,
        struct: TokenizedStructure,
        asym_ids: tuple[int, ...] | None,
    ) -> int:
        """Pick a random token, chain, or interface to crop around.
        If no preferred chain or interface is provided, then a random
        chain is selected.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        asym_ids: tuple[int, ...] | None
            The chain IDs to center the crop on.

        Returns
        ----------
        query: int
            The selected token index to crop around.
        """

        # check inputs
        resolved_mask = struct.token.resolved_mask
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # frequently used variables
        token_data = struct.token  # features: [L, ...]
        atom_data = struct.atom  # features: [L, 24, ...]

        all_tokens = token_data.token_index
        valid_tokens = all_tokens[resolved_mask]
        all_asym_ids = token_data.asym_id

        # use the first bioassembly
        holo_coords = atom_data.coords[..., 0, :]  # (num_tokens, 24, 3)
        # coordinates.
        all_token_centers = holo_coords[
            token_data.token_index, token_data.center_index, :
        ]  # (num_tokens, 3)

        # pick a random token, chain, or interface
        if asym_ids is None:
            # pick a random token from a random chain
            valid_chain_asym_ids = np.unique(all_asym_ids[valid_tokens])
            asym_id = np.random.choice(valid_chain_asym_ids)
            return pick_chain_token(struct, asym_id)
        elif len(asym_ids) == 1:
            # pick a random token from the preferred chain
            return pick_chain_token(struct, asym_ids[0])
        elif len(asym_ids) == 2:
            # pick a random token from the preferred interface
            return pick_interface_token(struct, asym_ids, all_token_centers)
        else:
            raise ValueError("asym_ids must be None, length 1, or length 2")

    def pick_spatial_interface_crop_query(
        self,
        structure: TokenizedStructure,
        asym_ids: tuple[int, ...] | None,
    ) -> int:
        """Pick a random token from an interface to crop around.
        If no bias interface is provided, then a random interface is selected.
        If a bais interface is provided, then select a random token from that interface.
        If a bais chain is provided, then select a random interface involving that chain.

        Parameters
        ----------
        structure: TokenizedStructure
            The tokenized structure.
        asym_ids: tuple[int, ...] | None
            The chain IDs to center the crop on.

        Returns
        ----------
        query: int
            The selected token index to crop around.
        """

        # check inputs
        resolved_mask = structure.token.resolved_mask
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # frequently used variables
        token_data = structure.token  # features: [L, ...]
        atom_data = structure.atom  # features: [L, 24, ...]

        # get the first bioassembly
        holo_coords = atom_data.coords[..., 0, :]  # (num_tokens, 24, 3)
        all_token_centers = holo_coords[
            token_data.token_index, token_data.center_index, :
        ]  # (num_tokens, 3)

        # pick a random token from an interface
        if asym_ids is None:
            # pick a random token from any random interface
            interface_asym_ids = get_random_interface(structure, None)
            if interface_asym_ids is None:
                # no valid interface found; default to a random center
                return self.pick_spatial_crop_query(structure, None)
            else:
                return pick_interface_token(
                    structure, interface_asym_ids, all_token_centers
                )
        elif len(asym_ids) == 1:
            # pick a random token from a random interface in the preferred chain
            interface_asym_ids = get_random_interface(structure, asym_ids[0])
            if interface_asym_ids is None:
                # no valid interface found; default to a random center
                return self.pick_spatial_crop_query(structure, asym_ids)
            else:
                return pick_interface_token(
                    structure, interface_asym_ids, all_token_centers
                )
        elif len(asym_ids) == 2:
            # pick a random token from the preferred interface
            return pick_interface_token(structure, asym_ids, all_token_centers)
        else:
            raise ValueError("asym_ids must be None, length 1, or length 2")

    def get_closest_tokens(
        self,
        structure: TokenizedStructure,
        max_tokens: int,
        query: int,
    ) -> np.ndarray:
        """Get the closest tokens to the query token."""
        # check inputs
        resolved_mask = structure.token.resolved_mask
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # frequently used variables
        token_data = structure.token  # features: [L, ...]
        atom_data = structure.atom  # features: [L, 24, ...]
        all_tokens = token_data.token_index
        valid_tokens = all_tokens[resolved_mask]

        # get the first bioassembly
        holo_coords = atom_data.coords[..., 0, :]  # (num_tokens, 24, 3)
        all_token_centers = holo_coords[
            token_data.token_index, token_data.center_index, :
        ]  # (num_tokens, 3)

        query_coords = all_token_centers[query]  # [3,]
        valid_coords = all_token_centers[valid_tokens]  # [num_valid_tokens, 3]

        # sort all tokens by distance to query_coords
        dists = valid_coords - query_coords  # [num_valid_tokens, 3]
        indices = np.argsort(np.linalg.norm(dists, axis=1))
        neighbor_indices = valid_tokens[indices]

        # select cropped indices
        cropped: set[int] = set(neighbor_indices[:max_tokens])
        return np.array(sorted(cropped))
