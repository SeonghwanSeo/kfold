import io
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
import numpy as np
import torch
from typing_extensions import override

from kfold.data import featurize, metadata, model_input, structure
from kfold.data.utils import symmetry
from kfold.utils.registry import Registry

from .cropper import BaseCropper
from .sampler import BaseSampler, Sample

"""
This dataset implementation includes a safe loading mechanism that retries
"""

# Type alias
SymmetryInfo = dict


class SafeLoadingDataset(torch.utils.data.Dataset, ABC):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        records: list[metadata.Metadata],
        safe_load: bool,
        featurization_args: dict | None,
        pretrained_embedding_paths: dict | None,
        mode: str = "train",
    ) -> None:
        self.records: list[metadata.Metadata] = records
        self.safe_load: bool = safe_load
        self.featurization_args = featurization_args or {}

        if mode == "train":
            return_symmetry = self.featurization_args.get("return_train_symmetry", False)
            return_structure = False
        elif mode == "validation":
            return_symmetry = self.featurization_args.get(
                "return_validation_symmetry", False
            )
            return_structure = True
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure

        # Check validity and assign pretrained embedding paths and dimensions
        pretrained_embedding_paths = pretrained_embedding_paths or {}
        known_keys = {
            "seq_embedding_path",
            "struct_embedding_path",
            "seq_embedding_dim",
            "struct_embedding_dim",
        }
        for key in pretrained_embedding_paths.keys():
            if key not in known_keys:
                raise ValueError(
                    f"Unknown key '{key}' in pretrained_embedding_paths. "
                    f"Known keys are: {known_keys}"
                )
        self.seq_embedding_path: str | None = pretrained_embedding_paths.get(
            "seq_embedding_path"
        )
        self.struct_embedding_path: str | None = pretrained_embedding_paths.get(
            "struct_embedding_path"
        )
        self.seq_embedding_dim: int | None = pretrained_embedding_paths.get(
            "seq_embedding_dim"
        )
        self.struct_embedding_dim: int | None = pretrained_embedding_paths.get(
            "struct_embedding_dim"
        )

        if self.seq_embedding_path is not None:
            assert Path(self.seq_embedding_path).exists(), (
                f"seq_embedding_path '{self.seq_embedding_path}' does not exist."
            )
            assert self.seq_embedding_dim is not None, (
                "seq_embedding_dim must be provided when seq_embedding_path is set."
            )
        if self.struct_embedding_path is not None:
            assert Path(self.struct_embedding_path).exists(), (
                f"struct_embedding_path '{self.struct_embedding_path}' does not exist."
            )
            assert self.struct_embedding_dim is not None, (
                "struct_embedding_dim must be provided when struct_embedding_path is set."
            )

    def __len__(self) -> int:
        return len(self.records)

    # === TO-DO Implement in subclasses === #
    @abstractmethod
    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> structure.TokenizedStructure:
        """Get the tokenized structure for the given index."""

    # === Optional to-override in subclasses === #
    def crop_structure(
        self,
        struct: structure.TokenizedStructure,
        **kwargs,
    ) -> structure.TokenizedStructure:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        return struct

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        return f_input.pad_to_multiple_of(64)

    def __getitem__(self, index: int) -> tuple[model_input.FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index, with retry on failure."""
        return self.get_item_safe(index, num_trials=10)

    def get_item_safe(
        self,
        index: int,
        num_trials: int = 10,
    ) -> tuple[model_input.FoldingInput, SymmetryInfo]:
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
                sample_id = sample.id
                print(f"Error loading index {sample_id}({index}): {e}. Retrying...")
                index = np.random.randint(0, len(self))
                trials.append(sample)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(
        self,
        record: metadata.Metadata,
        **kwargs,
    ) -> tuple[model_input.FoldingInput, SymmetryInfo]:
        """Get the folding input for the given sample."""
        record_id = record.id

        # Tokenization
        struct = self.load_tokenized_structure(record)

        # Cropping
        cropped_struct = self.crop_structure(struct, **kwargs)

        # Featurization
        f_input = self.featurize(cropped_struct, record)

        symmetry_dict = {}
        symmetry_dict["id"] = record_id
        if self.return_structure:
            symmetry_dict["structure"] = struct
        if self.return_symmetry:
            symmetry_dict["symmetry"] = symmetry.get_symmetries(
                f_input, cropped_struct, struct, {}
            )
        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        return f_input, symmetry_dict

    def featurize(
        self, struct: structure.TokenizedStructure, record: metadata.Metadata
    ) -> model_input.FoldingInput:
        """Featurize the given tokenized structure."""
        record_id = record.id

        # Featurization
        f_input = featurize.featurize_structure(struct, **self.featurization_args)

        # Add pretrained embeddings if provided
        if self.seq_embedding_path is not None:
            seq_emb_prefix = (
                f"{self.seq_embedding_path}/{record_id[:2]}/{record_id}/{record_id}_"
            )
        else:
            seq_emb_prefix = None
        if self.struct_embedding_path is not None:
            struct_emb_prefix = (
                f"{self.struct_embedding_path}/{record_id[:2]}/{record_id}/{record_id}_"
            )
        else:
            struct_emb_prefix = None

        f_input = featurize.add_pretrained_embeddings(
            f_input,
            seq_emb_prefix,
            struct_emb_prefix,
            self.seq_embedding_dim,
            self.struct_embedding_dim,
        )
        return f_input


class TrainingDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool,
        featurization_args: dict | None,
        pretrained_embedding_paths: dict | None,
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
        super().__init__(
            records,
            safe_load,
            featurization_args,
            pretrained_embedding_paths,
            mode="train",
        )
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
    def crop_structure(
        self,
        struct: structure.TokenizedStructure,
        **kwargs,
    ) -> structure.TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: list[str] = kwargs["asym_ids"]
        if self.max_tokens < struct.num_tokens:
            # Crop the tokenized structure
            struct = self.cropper.crop(struct, self.max_tokens, asym_ids)
        return struct

    @override
    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        return f_input.pad_to_max_token(max_tokens=self.max_tokens)

    @override
    def get_item_safe(
        self,
        index: int,
        num_trials: int = 10,
    ) -> tuple[model_input.FoldingInput, SymmetryInfo]:
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
                sample_id = sample.metadata.id
                print(f"Error loading index {sample_id}({index}): {e}. Retrying...")
                index = np.random.randint(0, len(self))
                if not self.safe_load:
                    raise e
                trials.append(sample)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )


class ValidationDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        max_tokens: int | None,
        safe_load: bool = True,
        featurization_args: dict | None = None,
        pretrained_embedding_paths: dict | None = None,
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
        super().__init__(
            records,
            safe_load,
            featurization_args,
            pretrained_embedding_paths,
            mode="validation",
        )
        self.max_tokens: int | None = max_tokens
        if self.max_tokens is not None:
            assert self.max_tokens % 64 == 0, f"max_tokens must be a multiple of {64}."

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        if self.max_tokens is not None:
            return f_input.pad_to_max_token(max_tokens=self.max_tokens)
        else:
            return f_input.pad_to_multiple_of(64)


class LMDBDatabase:
    """
    Provides LMDB-backed access to tokenized structures.
    The `lmdb_env` property lazily initializes and caches the LMDB environment
    on first access, ensuring efficient resource usage. The `load_from_lmdb`
    method retrieves a tokenized structure from the LMDB database using a
    record's ID as the key.
    """

    lmdb_path: Path

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                map_size=1024**4,  # 1 TB
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_env

    def load_from_lmdb(self, record: metadata.Metadata) -> structure.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        name = record.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB at {self.lmdb_path}.")

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            tokenized_structure = structure.TokenizedStructure.load_npz(byte_stream)
        return tokenized_structure

    def __del__(self):
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()


class LMDBTrainingDataset(TrainingDataset, LMDBDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        lmdb_path: Path,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool,
        featurization_args: dict | None,
        pretrained_embedding_paths: dict | None,
    ) -> None:
        TrainingDataset.__init__(
            self,
            records,
            max_tokens,
            cropper,
            sampler_config,
            safe_load,
            featurization_args,
            pretrained_embedding_paths,
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> structure.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(record)


class LMDBValidationDataset(ValidationDataset, LMDBDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        lmdb_path: Path,
        max_tokens: int | None,
        safe_load: bool,
        featurization_args: dict | None,
        pretrained_embedding_paths: dict | None,
    ) -> None:
        ValidationDataset.__init__(
            self,
            records,
            max_tokens,
            safe_load,
            featurization_args,
            pretrained_embedding_paths,
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> structure.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(record)
