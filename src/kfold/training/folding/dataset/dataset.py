import io
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
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

    def __init__(
        self,
        records: list[metadata.Metadata],
        safe_load: bool = True,
    ) -> None:
        self.records: list[metadata.Metadata] = records
        self.safe_load: bool = safe_load

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
        for _ in range(num_trials):
            sample: metadata.Metadata = self.records[index]
            try:
                return self.get_item(sample)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                if not self.safe_load:
                    raise e
                print(f"Error loading index {index}: {e}. Retrying...")
                index = np.random.randint(0, len(self))
                trials.append(sample)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(self, record: metadata.Metadata, **kwargs) -> model_input.FoldingInput:
        """Get the folding input for the given sample."""
        # Tokenization
        tokenized_structure = self.load_tokenized_structure(record)
        # Featurization
        f_input = featurize.featurize_structure(tokenized_structure)
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
        safe_load: bool = True,
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
        super().__init__(records, safe_load=safe_load)
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
                print(f"Error loading index {index}: {e}. Retrying...")
                index = np.random.randint(0, len(self))
                trials.append(sample)
                if not self.safe_load:
                    raise e
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
        f_input = featurize.featurize_structure(tokenized_structure)
        # Pad the folding input to max_tokens for LocalAtomAttention.
        f_input = self.pad_input(f_input)
        return f_input


class ValidationDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        max_tokens: int | None,
        safe_load: bool = True,
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
        super().__init__(records, safe_load=safe_load)
        self.max_tokens: int | None = max_tokens
        if self.max_tokens is not None:
            assert self.max_tokens % 64 == 0, f"max_tokens must be a multiple of {64}."

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        if self.max_tokens is not None:
            return f_input.pad_to_max_token(max_tokens=self.max_tokens)
        else:
            return f_input.pad_to_multiple_of(64)


class BoltzDatabase:
    structure_dir: Path

    def load_from_boltz(self, record: metadata.Metadata) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        name = record.id
        path = self.structure_dir / f"{name}.npz"
        boltz_structure = BoltzStructure.load(path)
        tokenized_structure = tokenize_structure(boltz_structure)
        return tokenized_structure


class BoltzTrainingDataset(TrainingDataset, BoltzDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        structure_dir: Path,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
    ) -> None:
        TrainingDataset.__init__(self, records, max_tokens, cropper, sampler_config)
        self.structure_dir: Path = structure_dir

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        return self.load_from_boltz(record)


class BoltzValidationDataset(ValidationDataset, BoltzDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        structure_dir: Path,
        max_tokens: int | None = None,
    ) -> None:
        ValidationDataset.__init__(self, records, max_tokens)
        self.structure_dir: Path = structure_dir

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        return self.load_from_boltz(record)


class LMDBDatabase:
    lmdb_path: Path

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                map_size=100 * 1024**3,  # 100 GB
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_env

    def load_from_lmdb(self, record: metadata.Metadata) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from BoltzStructure."""
        name = record.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB at {self.lmdb_path}.")

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            tokenized_structure = tokenized.TokenizedStructure.load_npz(byte_stream)
        return tokenized_structure


class LMDBTrainingDataset(TrainingDataset, LMDBDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        lmdb_path: Path,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool = True,
    ) -> None:
        TrainingDataset.__init__(
            self, records, max_tokens, cropper, sampler_config, safe_load
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(record)


class LMDBValidationDataset(ValidationDataset, LMDBDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        lmdb_path: Path,
        max_tokens: int | None = None,
        safe_load: bool = True,
    ) -> None:
        ValidationDataset.__init__(self, records, max_tokens, safe_load)
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> tokenized.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(record)
