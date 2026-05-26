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
        max_sequence_tokens: int,
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
        max_sequence_tokens : int
            The maximum sequence length for the model. This is used to ensure that the
            cropped structure does not exceed the model's input size.
        bias_asym_id : int | tuple[int, int] | None, optional
            The chain IDs to center the crop on. If None, a random chain or interface
            will be selected.

        Returns
        -------
        cropped_struct: TokenizedStructure
            The cropped data.
        """
        rng = rng or np.random.default_rng()

        # +2 for CLS and SEP tokens
        assert max_sequence_tokens >= max_tokens + 2 * len(
            np.unique(struct.chain.entity_id)
        )

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
        )[:max_tokens]

        # Get the sequence token indices to include in the crop, ensuring that
        # all sequence tokens corresponding to the selected tokens are included.
        selected_seq_token_indices = self.get_sequence_token_indices(
            struct, selected_token_indices, max_sequence_tokens
        )[:max_sequence_tokens]

        return struct.crop(selected_token_indices, selected_seq_token_indices)

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

    def get_sequence_token_indices(
        self,
        struct: TokenizedStructure,
        token_indices: np.ndarray,
        max_sequence_tokens: int,
    ) -> np.ndarray:
        """Get the indices of the sequence tokens to include in the crop.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        token_indices : np.ndarray
            The indices of the tokens to include in the crop.
        max_sequence_tokens : int
            The maximum number of sequence tokens to keep.
        """
        # Get the asym IDs in the selected tokens
        asym_ids_in_crop = np.unique(struct.token.asym_id[token_indices])

        # If all seq-tokens belonging to these chains fit in memory
        all_seq_indices = np.where(np.isin(struct.sequence.asym_id, asym_ids_in_crop))[0]
        if len(all_seq_indices) <= max_sequence_tokens:
            return all_seq_indices

        # Otherwise, we need to expand from the selected seq-tokens.
        required_seq_tokens = np.sort(
            np.unique(struct.token.seq_token_index[token_indices])
        )
        # Filter out potential padded values
        required_seq_tokens = required_seq_tokens[required_seq_tokens >= 0]

        seqlen = len(struct.sequence)
        asym_ids = struct.sequence.asym_id

        seq_tok_indices = set(required_seq_tokens)

        # Initialize cursors for BFS expansion
        left_cursors = {
            i
            for i in required_seq_tokens
            if i > 0 and asym_ids[i - 1] == asym_ids[i] and i - 1 not in seq_tok_indices
        }
        right_cursors = {
            i
            for i in required_seq_tokens
            if i < seqlen - 1
            and asym_ids[i + 1] == asym_ids[i]
            and i + 1 not in seq_tok_indices
        }

        # Expand left and right until reach max_sequence_tokens
        # or run out of valid neighbors
        while len(seq_tok_indices) < max_sequence_tokens:
            # Expand left
            new_left_cursors = set()
            for i in sorted(left_cursors):
                left_i = i - 1
                if left_i >= 0 and left_i not in seq_tok_indices:
                    if asym_ids[left_i] == asym_ids[i]:
                        seq_tok_indices.add(left_i)
                        if left_i > 0:
                            new_left_cursors.add(left_i)
                        if len(seq_tok_indices) >= max_sequence_tokens:
                            break

            if len(seq_tok_indices) >= max_sequence_tokens:
                break

            # Expand right
            new_right_cursors = set()
            for i in sorted(right_cursors):
                right_i = i + 1
                if right_i < seqlen and right_i not in seq_tok_indices:
                    if asym_ids[right_i] == asym_ids[i]:
                        seq_tok_indices.add(right_i)
                        if right_i < seqlen - 1:
                            new_right_cursors.add(right_i)
                        if len(seq_tok_indices) >= max_sequence_tokens:
                            break

            if len(seq_tok_indices) >= max_sequence_tokens:
                break

            left_cursors = new_left_cursors
            right_cursors = new_right_cursors
            if not left_cursors and not right_cursors:
                # No more neighbors to expand
                break

        return np.array(sorted(seq_tok_indices))
