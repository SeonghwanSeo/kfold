from abc import ABC, abstractmethod

import numpy as np

from kfold.data.structure import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER, BaseConfig


@DATA_CROPPER.register()
class BaseCropper(ABC):
    """Interpolate between contiguous and spatial crops."""

    class Config(BaseConfig):
        """Configuration for BaseCropper."""

    def crop(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        asym_ids: tuple[int, ...] | None,
    ) -> TokenizedStructure:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : Tokenized
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None, optional
            The chain IDs to center the crop on. If None, a random chain

        Returns
        -------
        TokenizedStructure
            The cropped data.
        """

        # Check if structure have any valid tokens
        if struct.num_tokens == 0:
            raise ValueError("No valid tokens in struct")

        if struct.num_tokens <= max_tokens:
            # No cropping needed
            return struct

        # Get the token indices to include in the crop
        selected_token_indices = self.get_token_indices(struct, max_tokens, asym_ids)
        return struct.crop(selected_token_indices)

    @abstractmethod
    def get_token_indices(
        self,
        struct: TokenizedStructure,
        max_tokens: int,
        asym_ids: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Get the indices of the tokens to include in the crop.

        Parameters
        ----------
        struct : Tokenized
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        asym_ids : tuple[int, ...] | None, optional
            The chain IDs to center the crop on. If None, a random chain
        """
        raise NotImplementedError
