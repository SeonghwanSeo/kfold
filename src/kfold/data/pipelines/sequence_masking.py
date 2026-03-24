"""Masking pipeline for evolutionary feature augmentation.

While the regular MLM approach masks tokens with a three distinct strategies
(mask token, random token, or unchanged), this implementation simplifies the
process by only masking tokens with a mask token. This approach is effective
for augmenting evolutionary features without additional bias from random token
replacement.
"""

from typing import Self

import numpy as np

from kfold.constants.sequence import (
    BOS_TOKEN_INDEX,
    EOS_TOKEN_INDEX,
    MASK_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
)
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.misc import spawn_rng


class SequenceMasking:
    def __init__(self, mask_prob: float = 0.9, mask_ratio: float = 0.15) -> None:
        """

        Parameters
        ----------
        mask_prob : float
            The masking probability for applying the masking to the input sequence.
        mask_ratio : float
            The ratio of tokens to mask in the input sequence. Default is 0.15.
        """
        self.prob: float = mask_prob
        self.mask_ratio: float = mask_ratio
        self.special_token_indices: np.ndarray = np.array(
            [BOS_TOKEN_INDEX, EOS_TOKEN_INDEX, PAD_TOKEN_INDEX]
        )
        self.mask_token_index: int = MASK_TOKEN_INDEX

    @classmethod
    def inference_mode(cls) -> Self:
        """Create a SequenceMasking instance for inference mode with no masking."""
        return cls(mask_prob=0.0)

    def __call__(
        self,
        input: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> None:
        """

        Parameters
        ----------
        input : TokenizedStructure
            The input tokenized structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        """
        self.mask(input, rng)

    def mask(
        self,
        input: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> None:
        """

        Parameters
        ----------
        input : TokenizedStructure
            The input tokenized structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        """
        # Create new rng for this sampling to avoid affecting global state
        rng = spawn_rng(rng)

        if self.prob <= 0.0:
            return  # No masking needed

        if rng.random() >= self.prob:
            return  # Skip masking based on probability

        # Determine the masking ratio
        mask_ratio = rng.uniform(0.0, self.mask_ratio)

        sequence_input = input.sequence.seq_token_id

        # Determine which positions to mask
        mask_positions = rng.random(size=sequence_input.shape) < mask_ratio
        # Ensure special tokens are not masked
        mask_positions &= ~np.isin(sequence_input, self.special_token_indices)

        # Set masked positions to the mask token index
        input.sequence.mlm_mask[mask_positions] = True
