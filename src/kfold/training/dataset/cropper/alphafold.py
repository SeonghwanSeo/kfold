"""AlphaFold-Multimer/AlphaFold-3 style cropping.

Implements three kinds of cropping strategies:
- Contiguous: crops a contiguous regions of tokens from a single chain.
- Spatial: crops a region of tokens centered on a random token.
- Spatial-interface: crops a region of tokens centered on a random interface.
"""

import numpy as np

from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER

from . import utils
from .base import BaseCropper


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

        w_contiguous: float = 0.2
        w_spatial: float = 0.4
        w_spatial_interface: float = 0.4

    def __init__(self, config: Config):
        self.w_contiguous: float = config.w_contiguous
        self.w_spatial: float = config.w_spatial
        self.w_spatial_interface: float = config.w_spatial_interface
        if self.w_contiguous + self.w_spatial + self.w_spatial_interface != 1.0:
            raise ValueError("Cropping strategy weights must sum to 1.0")

    def get_token_indices(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Get the token indices to include in the crop.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.
        bias_asym_id: int | tuple[int, ...] | None
            The chain ID(s) to center the crop on. If None, a random chain or interface
            will be selected.
        rng: np.random.Generator
            The random number generator.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """
        rng = rng or np.random.default_rng()

        # random choice on cropping strategy
        strategy = utils.random_choice(
            ["contiguous", "spatial", "spatial_interface"],
            p=[self.w_contiguous, self.w_spatial, self.w_spatial_interface],
            rng=rng,
        )
        match strategy:
            case "contiguous":
                return self.crop_contiguous(struct, max_tokens, rng)
            case "spatial":
                return self.crop_spatial(struct, max_tokens, bias_asym_id, rng)
            case "spatial_interface":
                return self.crop_spatial_interface(struct, max_tokens, bias_asym_id, rng)
            case _:
                raise ValueError(f"Unknown cropping strategy: {strategy}")

    def crop_contiguous(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Get the token indices to include in the contiguous crop.

        Follows AlphaFold-Multimer SI Algorithm 1 with modification
        in initializing n_remaining. Modifications are in line with
        OpenFold3's implementation.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.
        rng: np.random.Generator
            The random number generator.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """

        all_tokens = struct.token.token_index  # =np.arange(num_tokens)

        # randomly shuffle chains
        chain_ids = rng.permutation(struct.chain.asym_id)

        # Initialize counters; n_remaining is the sum of all tokens
        n_added = 0
        n_remaining = np.sum(np.isin(struct.token.asym_id, chain_ids))
        cropped: set[int] = set()

        # iterate over chains
        for chain_id in chain_ids:
            # get chain length as number of tokens
            chain_mask = struct.token.asym_id == chain_id
            chain_tokens = all_tokens[chain_mask]
            chain_length = len(chain_tokens)
            n_remaining -= chain_length

            # sample crop length and start
            crop_size_max = min(max_tokens - n_added, chain_length)
            crop_size_min = min(chain_length, max(0, max_tokens - n_added - n_remaining))
            crop_size = rng.integers(crop_size_min, crop_size_max + 1, dtype=int)
            crop_start = rng.integers(0, chain_length - crop_size + 1, dtype=int)
            n_added += crop_size

            # get token indices in crop
            crop_tokens = chain_tokens[crop_start : crop_start + crop_size]

            # slice using sampled crop start and length for this chain
            cropped.update(crop_tokens.tolist())

        return np.array(sorted(cropped))

    def crop_spatial(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a spatial region around a random token.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.
        bias_asym_id: int | tuple[int, ...] | None
            The chain ID(s) to center the crop on. If None, a random chain or interface
            will be selected.
        rng: np.random.Generator
            The random number generator.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """
        # get the tokens with valid center atom
        resolved_mask = struct.atom.resolved_mask[
            struct.token.token_index, struct.token.center_index
        ]
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # pick a random token from a chain or interface if specified
        anchor = utils.pick_token(struct, bias_asym_id, mask=resolved_mask, rng=rng)
        return self.get_closest_tokens(struct, max_tokens, anchor)

    def crop_spatial_interface(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a spatial region around a random interface tokens

        If no bias interface is provided, then a random interface is selected.
        If a bias interface is provided, then select a random token from that interface.
        If a bias chain is provided, then select a random interface involving that chain.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        max_tokens: int
            The maximum number of tokens to crop.
        bias_asym_id: int | tuple[int, ...] | None
            The chain ID(s) to center the crop on. If None, a random chain or interface
            will be selected.
        rng: np.random.Generator
            The random number generator.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """
        # get valid interfaces
        all_interfaces: list[tuple[int, int]] = self.get_valid_interfaces(struct)
        if len(all_interfaces) == 0:
            # no valid interfaces found; default to a random center
            return self.crop_spatial(struct, max_tokens, bias_asym_id, rng)

        # pick a random token from an interface
        if bias_asym_id is None:
            # pick a random token from any random interface
            interface_id = utils.random_choice(all_interfaces, rng=rng)
        elif isinstance(bias_asym_id, int):
            # pick a random token from a random interface in the preferred chain
            candidate_interfaces = [v for v in all_interfaces if bias_asym_id in v]
            if len(candidate_interfaces) == 0:
                # no valid interfaces found; default to a random center
                candidate_interfaces = all_interfaces
            interface_id = utils.random_choice(candidate_interfaces, rng=rng)
        else:
            # pick a random token from the preferred interface
            if bias_asym_id in all_interfaces:
                interface_id = bias_asym_id
            else:
                # no valid interfaces found; default to a random interface
                interface_id = utils.random_choice(all_interfaces, rng=rng)
        anchor = utils.pick_interface_token(struct, interface_id, rng=rng)
        return self.get_closest_tokens(struct, max_tokens, anchor)

    def get_closest_tokens(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        query: int,
    ) -> np.ndarray:
        """Get the closest tokens to the query token."""
        # check inputs
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        if resolved_mask.sum() <= max_tokens:
            # all valid tokens fit in the crop
            return struct.token.token_index[resolved_mask]

        # frequently used variables
        token_data = struct.token  # [n_tokens, ...]
        atom_data = struct.atom  # [n_tokens, 24, ...]
        all_tokens = token_data.token_index
        valid_tokens = all_tokens[resolved_mask]

        # get the first bioassembly
        holo_coords = atom_data.label_coords  # [n_tokens, 24, 3]
        all_token_centers = holo_coords[
            token_data.token_index, token_data.center_index, :
        ]  # (num_tokens, 3)

        query_coords = all_token_centers[query]  # [3,]
        valid_coords = all_token_centers[valid_tokens]  # [n_val_tokens, 3]

        # sort all tokens by distance to query_coords
        dists = np.linalg.norm(valid_coords - query_coords, axis=1)  # [n_val_tokens]
        indices = np.argpartition(dists, max_tokens - 1)[:max_tokens]
        neighbor_indices = valid_tokens[indices]
        neighbor_indices.sort()
        return neighbor_indices

    @staticmethod
    def get_valid_interfaces(struct: TokenizedStructure) -> list[tuple[int, int]]:
        """Get all valid interfaces in the structure.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.

        Returns
        -------
        interface_ids : list[tuple[int, int]]
            The valid interfaces in the structure.
        """
        metadata = struct.metadata
        assert metadata is not None, "Structure metadata is required"
        all_chains: set[int] = set(struct.chain.asym_id.tolist())
        all_interfaces: list[tuple[int, int]] = [
            interface.asym_ids for interface in metadata.interfaces if interface.is_valid
        ]
        all_interfaces = [v for v in all_interfaces if set(v).issubset(all_chains)]
        return sorted(set(all_interfaces))
