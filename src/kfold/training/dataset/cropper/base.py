from abc import ABC, abstractmethod

import numpy as np

from kfold.data.types.metadata import Metadata
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.registry import DATA_CROPPER, BaseConfig


@DATA_CROPPER.register()
class BaseCropper(ABC):
    """Interpolate between contiguous and spatial crops."""

    class Config(BaseConfig):
        """Configuration for BaseCropper."""

    def crop(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_id : int | tuple[int, int] | None, optional
            The chain IDs to center the crop on. If None, a random chain or interface
            will be selected.

        Returns
        -------
        cropped_struct: TokenizedStructure
            The cropped data.
        """
        rng = rng or np.random.default_rng()

        # Check if structure have any valid tokens
        if struct.num_tokens == 0:
            raise ValueError("No valid tokens in struct")

        if struct.num_tokens <= max_tokens:
            # No cropping needed
            return struct

        # Get the token indices to include in the crop
        selected_token_indices = self.get_token_indices(
            struct,
            metadata,
            max_tokens,
            bias_asym_id,
            rng=rng,
        )
        return struct.crop(selected_token_indices)

    @abstractmethod
    def get_token_indices(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        max_tokens: int,
        bias_asym_id: int | tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Get the indices of the tokens to include in the crop.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        metadata : Metadata
            The metadata for the structure.
        max_tokens : int
            The maximum number of tokens to crop.
        bias_asym_id : int | tuple[int, int] | None
            The chain ID(s) to center the crop on. If None, a random chain or interface
            will be selected.
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        selected_token_indices : np.ndarray
            The indices of the tokens to include in the crop.
        """
        raise NotImplementedError
