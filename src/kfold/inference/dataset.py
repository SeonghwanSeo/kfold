"""Define training modules for k-fold"""

from typing import Any

import torch

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure

from .data_pipeline import InputDataPipeline
from .query import Query


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


def collate_fn_single(batch: list[Any]) -> Any:
    """Collate function that returns the first element of the batch."""
    return batch[0]


def prepare_inference_dataloader(
    queries: list[Query],
    ccd: CCD,
    num_samples: int = 5,
    use_sequence_masking: bool = False,
    seed: int = 1,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = InferenceDataset(
        queries=queries,
        ccd=ccd,
        num_samples=num_samples,
        use_sequence_masking=use_sequence_masking,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_single,
    )


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for structure prediction inference."""

    def __init__(
        self,
        queries: list[Query],
        ccd: CCD,
        num_samples: int = 5,
        use_sequence_masking: bool = False,
        seed: int = 1,
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
        seed : int | None
            Random seed for reproducibility.
        """
        self.queries: list[Query] = queries
        self.seed: int = seed

        # Data pipeline components
        self.data_pipeline = InputDataPipeline(
            ccd=ccd,
            num_samples=num_samples,
            use_sequence_masking=use_sequence_masking,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> tuple[Query, RefStructure, FoldingInput] | None:
        """Get the folding input for the given input."""
        query: Query = self.queries[index]

        # Prepare input data
        ref_struct, tok_struct, f_input = self.data_pipeline.process_query(query)

        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        # Add batch dimension
        f_input = FoldingInput.from_list([f_input])

        return query, ref_struct, f_input

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 32 for LocalAtomAttention."""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 16)
        # Pad max_sequence length for CUDA efficiency.
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 128)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 32)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence=num_sequence_tokens,
        )
