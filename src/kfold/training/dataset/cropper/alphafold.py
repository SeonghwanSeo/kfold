"""AlphaFold-Multimer/AlphaFold-3 style cropping.

Implements three kinds of cropping strategies:
- Contiguous: crops a contiguous regions of tokens from a single chain.
- Spatial: crops a region of tokens centered on a random token.
- Spatial-interface: crops a region of tokens centered on a random interface.
"""

import numpy as np

from kfold.data.types.metadata import Metadata
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
        metadata: Metadata,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Get the token indices to include in the crop.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        metadata: Metadata
            The structure metadata.
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
        v = rng.random()
        if v < self.w_contiguous:
            # Contiguous cropping
            crop_indices = self.crop_contiguous(struct, metadata, max_tokens, rng=rng)
        elif v < self.w_contiguous + self.w_spatial:
            # Spatial cropping
            crop_indices = self.crop_spatial(
                struct, metadata, max_tokens, bias_asym_id, rng=rng
            )
        else:  # Spatial interface cropping
            crop_indices = self.crop_spatial_interface(
                struct, metadata, max_tokens, bias_asym_id, rng=rng
            )

        # Ensure sorted order and limit to max_tokens
        crop_indices.sort()
        if len(crop_indices) > max_tokens:
            crop_indices = crop_indices[:max_tokens]

        return crop_indices

    def crop_contiguous(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
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
        metadata: Metadata
            The structure metadata.
        max_tokens: int
            The maximum number of tokens to crop.
        rng: np.random.Generator
            The random number generator.

        Returns
        ----------
        token_indices: np.ndarray
            The selected token indices.
        """

        # Compute the number of tokens and start indices per chain
        asym_id_to_chain_idx: dict[int, int] = {
            v: i for i, v in enumerate(struct.chain.asym_id.tolist())
        }
        chain_sizes: dict[int, int] = {
            asym_id: int(struct.chain.num_tokens[chain_idx])
            for asym_id, chain_idx in asym_id_to_chain_idx.items()
        }

        # Randomly permute the chain order
        asym_ids = struct.chain.asym_id
        selected_asym_ids = rng.permutation(asym_ids)

        # Line 1
        n_added: int = 0
        # Line 2
        # NOTE: This differs from the original algorithm which uses max_tokens.
        n_remaining: int = sum(chain_sizes[asym_id] for asym_id in selected_asym_ids)

        is_selected = np.zeros(struct.num_tokens, dtype=bool)

        # Line 3-13
        for asym_id in selected_asym_ids:
            if n_added >= max_tokens:
                break

            n_k = chain_sizes[asym_id]
            # Line 4
            n_remaining -= n_k

            # Sample length of crop for current chain
            # Line 5
            max_crop = min(max_tokens - n_added, n_k)
            # Line 6
            min_crop = min(n_k, max(0, max_tokens - n_added - n_remaining))
            # Line 7
            crop_size = int(rng.integers(min_crop, max_crop + 1))
            # Line 8
            n_added += crop_size

            if crop_size == 0:
                continue

            # Line 9
            crop_start = int(rng.integers(0, n_k - crop_size + 1))

            # Line 11
            chain_idx = asym_id_to_chain_idx[asym_id]
            chain_st = int(struct.chain.token_start[chain_idx])
            crop_start += chain_st
            selected_tokens = np.arange(crop_start, crop_start + crop_size)

            # Line 12
            is_selected[selected_tokens] = True

        return np.where(is_selected)[0]

    def crop_spatial(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a spatial region around a random token.

        Parameters
        ----------
        struct: TokenizedStructure
            The tokenized structure.
        metadata: Metadata
            The structure metadata.
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
        tokens = struct.token.token_index  # =np.arange(n_tokens)
        center_idx = struct.token.center_index  # (n_tokens, 3)
        resolved_mask = struct.atom.resolved_mask[tokens, center_idx]  # (n_tokens,)
        if not resolved_mask.any():
            raise ValueError("No valid tokens in structure")

        # pick a random token from a chain or interface if specified
        anchor = utils.pick_token(struct, bias_asym_id, mask=resolved_mask, rng=rng)
        return self.get_closest_tokens(struct, anchor, max_tokens)

    def crop_spatial_interface(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
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
        metadata: Metadata
            The structure metadata.
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
        all_interfaces: list[tuple[int, int]] = self.get_valid_interfaces(metadata)
        if len(all_interfaces) == 0:
            # no valid interfaces found; default to a random center
            return self.crop_spatial(struct, metadata, max_tokens, bias_asym_id, rng)

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
        return self.get_closest_tokens(struct, anchor, max_tokens)

    def get_closest_tokens(
        self,
        struct: TokenizedStructure,
        anchor_token: int,
        crop_size: int,
        center_coords: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Crop tokens spatially around an anchor token.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        anchor_token : int
            The anchor token index.
        crop_size : int
            The number of tokens to crop.
        center_coords : np.ndarray | None, optional
            Precomputed center coordinates of tokens.
        mask : np.ndarray | None, optional
            Precomputed resolved mask of tokens.

        Returns
        -------
        token_indices : np.ndarray
            The selected token indices.
        """
        tokens = struct.token.token_index  # =np.arange(num_tokens)
        center_idx = struct.token.center_index  # (num_tokens, 3)
        if center_coords is None:
            center_coords = struct.atom.label_coords[
                tokens, center_idx
            ]  # (num_tokens, 3)
        if mask is None:
            mask = struct.atom.resolved_mask[tokens, center_idx]  # (num_tokens,)

        if mask.sum() <= crop_size:
            # If all resolved tokens fit in the budget, return all
            return np.where(mask)[0]

        # Compute distances to all tokens
        anchor_coord = center_coords[anchor_token]  # (3,)
        assert np.isfinite(anchor_coord).all(), "Anchor token has non-finite coordinates."
        dists = np.linalg.norm(center_coords - anchor_coord, axis=1)  # (num_tokens,)
        dists[~mask] = np.inf
        # Get tokens within budget (this includes the anchor token itself)
        neighbor_indices = np.argpartition(dists, crop_size - 1)[:crop_size]
        neighbor_indices.sort()
        return neighbor_indices

    @staticmethod
    def get_valid_interfaces(metadata: Metadata) -> list[tuple[int, int]]:
        """Get all valid interfaces in the structure.

        Parameters
        ----------
        metadata : Metadata
            The structure metadata.

        Returns
        -------
        interface_ids : list[tuple[int, int]]
            The valid interfaces in the structure.
        """
        all_interfaces: list[tuple[int, int]] = [
            interface.asym_ids for interface in metadata.interfaces
        ]
        return sorted(set(all_interfaces))
