from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from typing_extensions import override

from kfold.data import featurize, metadata, model_input, tokenized
from kfold.utils.boltz.process import tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.registry import Registry

from .cropper import BaseCropper
from .sampler import BaseSampler, Sample

"""
This dataset implementation includes a safe loading mechanism that retries
"""


class SafeLoadingDataset(torch.utils.data.Dataset, ABC):
    """A dataset that safely retries loading data on failure."""

    def __init__(self, records: list[metadata.Metadata]) -> None:
        self.records: list[metadata.Metadata] = records

    def __len__(self) -> int:
        return len(self.records)

    # === TO-DO Implement in subclasses === #
    @abstractmethod
    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Get the tokenized structure for the given index."""

    # === Optional to-override in subclasses === #
    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        return f_input.pad_to_multiple_of(64)

    def __getitem__(self, index: int) -> model_input.FoldingInput:
        """Get the folding input for the given index, with retry on failure."""
        return self.get_item_safe(index, num_trials=10)

    def get_item_safe(self, index: int, num_trials: int = 10) -> model_input.FoldingInput:
        trials = []
        for i in range(num_trials):
            sample = self.records[index]
            try:
                return self.get_item(sample)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                print(f"Error loading index {index}: {e}. Retrying...")
                if i < num_trials - 1:
                    index = np.random.randint(0, len(self))
                    trials.append(sample)
        else:
            raise RuntimeError(
                f"Failed to load data after {num_trials} attempts. Tried: {trials}"
            )

    def get_item(self, record: metadata.Metadata, **kwargs) -> model_input.FoldingInput:
        """Get the folding input for the given sample."""
        # Tokenization
        tokenized_structure = self.load_tokenized_structure(record)
        # Featurization
        f_input = featurize.featurize_structure(
            structure=tokenized_structure,
            metadata={"metadata": record},
        )
        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)
        return f_input


class TrainingDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
    ) -> None:
        """
        Parameters
        ----------
        records : list[metadata.Metadata]
            List of samples to use in the dataset.
        max_tokens : int
            Maximum number of tokens per sample. Must be a multiple of 64 for
            LocalAtomAttention.
        cropper : BaseCropper
            Cropper to use for cropping samples.
        sampler_config : BaseSampler.Config | None
            Sampler configuration to use for sampling samples.
            If None, uniform sampling is used.

        Notes
        -----
        This dataset implements AF3-style sampling (chain/interface-based).
        1. Samples are generated based on chains/interfaces in the structures.
        2. During data loading, samples are cropped to fit within `max_tokens`
           using the provided `cropper`.
        """
        super().__init__(records)
        self.max_tokens: int = max_tokens
        self.cropper: BaseCropper = cropper
        assert self.max_tokens % 64 == 0, f"max_tokens must be a multiple of {64}."

        # AF3-style sampling (chain/interface-based)
        if sampler_config is None:
            sampler_config = BaseSampler.Config()  # uniform sampler
        sampler = Registry.instantiate(sampler_config)
        samples, weights = sampler.get_samples(records)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray | None = weights

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def get_item_safe(self, index: int, num_trials: int = 10) -> model_input.FoldingInput:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.records`.
        """
        trials = []
        for _ in range(num_trials):
            sample = self.samples[index]
            try:
                return self.get_item(sample.metadata, asym_ids=sample.asym_ids)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                raise e
                print(f"Error loading index {index}: {e}. Retrying...")
                index = np.random.randint(0, len(self))
                trials.append(sample)
        else:
            raise RuntimeError(
                f"Failed to load data after {num_trials} attempts. Tried: {trials}"
            )

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        return f_input.pad_to_max_token(max_tokens=self.max_tokens)

    def get_item(self, record: metadata.Metadata, **kwargs) -> model_input.FoldingInput:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."

        tokenized_structure = self.load_tokenized_structure(record)

        # Cropping
        if self.max_tokens < tokenized_structure.num_tokens:
            # Crop the tokenized structure
            asym_ids: tuple[int, ...] | None = kwargs["asym_ids"]
            tokenized_structure = self.cropper.crop(
                tokenized_structure, self.max_tokens, asym_ids
            )

        # Featurization
        f_input = featurize.featurize_structure(
            structure=tokenized_structure,
            metadata={"metadata": record},
        )
        # Pad the folding input to max_tokens for LocalAtomAttention.
        f_input = self.pad_input(f_input)
        return f_input


class ValidationDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        max_tokens: int | None,
    ) -> None:
        """
        Parameters
        ----------
        records : list[metadata.Metadata]
            List of samples to use in the dataset.
        max_tokens : int | None
            Maximum number of tokens per sample. If None, padding is done to
            the nearest multiple of 64.
        """
        super().__init__(records)
        self.max_tokens: int | None = max_tokens
        if self.max_tokens is not None:
            assert self.max_tokens % 64 == 0, f"max_tokens must be a multiple of {64}."

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        if self.max_tokens is not None:
            return f_input.pad_to_max_token(max_tokens=self.max_tokens)
        else:
            return f_input.pad_to_multiple_of(64)


class BoltzTrainingDataset(TrainingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        structure_dir: Path,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
    ) -> None:
        super().__init__(records, max_tokens, cropper, sampler_config)
        self.structure_dir: Path = structure_dir

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        name = record.id
        path = self.structure_dir / f"{name}.npz"
        boltz_structure = BoltzStructure.load(path)
        tokenized_structure = tokenize_structure(boltz_structure)
        return tokenized_structure


class BoltzValidationDataset(ValidationDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        structure_dir: Path,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(records, max_tokens)
        self.structure_dir: Path = structure_dir

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        name = record.id
        path = self.structure_dir / f"{name}.npz"
        boltz_structure = BoltzStructure.load(path)
        tokenized_structure = tokenize_structure(boltz_structure)
        return tokenized_structure
