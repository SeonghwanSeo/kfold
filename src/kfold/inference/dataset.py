"""Inference dataset for structure prediction."""

from typing import NamedTuple

import torch

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure

from .data_pipeline import InputDataPipeline
from .query import Query


class InferenceInput(NamedTuple):
    """An input for inference (single query)."""

    query: Query
    ref_struct: RefStructure
    f_input: FoldingInput
    struct_tok_input: dict


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for structure prediction inference."""

    def __init__(
        self,
        queries: list[Query],
        ccd: CCD,
        num_samples: int = 5,
        use_sequence_masking: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        queries : list[Query]
            List of queries.
        ccd : CCD
            Component for handling common chemical components.
        num_samples : int
            Number of diffusion samples to generate for each query.
        use_sequence_masking : bool
            Whether to use sequence masking for sampling diversity
        """
        self.queries: list[Query] = queries
        self.data_pipeline = InputDataPipeline(ccd, num_samples, use_sequence_masking)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> InferenceInput:
        """Get the folding input for the given input."""
        query: Query = self.queries[index]

        # Prepare input data
        ref_struct, _, f_input, struct_tok_input = self.data_pipeline.run(query)

        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        return InferenceInput(query, ref_struct, f_input, struct_tok_input)

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 64"""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 32)
        # Pad max_sequence length for CUDA efficiency.
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 64)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 64)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence_tokens=num_sequence_tokens,
        )
