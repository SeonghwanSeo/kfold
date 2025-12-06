import io
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
import numpy as np
import torch
from typing_extensions import override

from kfold.data import featurize, metadata, model_input, structure
from kfold.utils.registry import Registry

from .cropper import BaseCropper, PreCropper
from .sampler import BaseSampler, Sample
from .utils import symmetry

"""
This dataset implementation includes a safe loading mechanism that retries
"""

# Type alias
SymmetryInfo = dict


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class SafeLoadingDataset(torch.utils.data.Dataset, ABC):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        records: list[metadata.Metadata],
        paths: dict[str, Path | None],
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        return_structure: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        self.records: list[metadata.Metadata] = records
        self.safe_load: bool = safe_load

        # Featurization arguments (copy to avoid mutation)
        featurization_args = featurization_args.copy()
        self.featurization_args: dict = featurization_args

        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure

        if self.return_symmetry:
            assert ccd_symmetry_dict is not None, (
                "ccd_symmetry_dict must be provided when return_symmetry is True."
            )
            self.ccd_symmetry_dict: dict = ccd_symmetry_dict

        # Pretrained embeddings
        self.seq_embedding_path: Path | None = paths["seq_embedding_path"]
        self.seq_embedding_dim: int | None = featurization_args.pop("seq_embedding_dim")
        if self.seq_embedding_path is not None:
            assert self.seq_embedding_path.exists(), (
                f"seq_embedding_path '{self.seq_embedding_path}' does not exist."
            )
            assert self.seq_embedding_dim is not None, (
                "seq_embedding_dim must be provided when seq_embedding_path is provided."
            )
        else:
            assert self.seq_embedding_dim is None, (
                "seq_embedding_dim must be None when seq_embedding_path is not provided."
            )

        self.struct_embedding_path: Path | None = paths["struct_embedding_path"]
        self.struct_embedding_dim: int | None = featurization_args.pop(
            "struct_embedding_dim"
        )
        if self.struct_embedding_path is not None:
            assert self.struct_embedding_path.exists(), (
                f"struct_embedding_path '{self.struct_embedding_path}' does not exist."
            )
            assert self.struct_embedding_dim is not None, (
                "struct_embedding_dim must be provided when struct_embedding_path is"
                " provided."
            )
        else:
            assert self.struct_embedding_dim is None, (
                "struct_embedding_dim must be None when struct_embedding_path is not"
                " provided."
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
    def pre_crop_structure(
        self,
        struct: structure.TokenizedStructure,
        **kwargs,
    ) -> structure.TokenizedStructure:
        """Pre-crop the folding input structure as needed.
        See Section 2.5.4 of AlphaFold3 SI

        In contrast to `crop_structure`, this method is intended for
        sample sub-complexes from the original structure before
        applying the main cropping strategy.
        """
        return struct

    def crop_structure(
        self,
        struct: structure.TokenizedStructure,
        **kwargs,
    ) -> structure.TokenizedStructure:
        """Crop the folding input structure as needed."""
        return struct

    def pad_input(self, f_input: model_input.FoldingInput) -> model_input.FoldingInput:
        """Pad the folding input to multiple of 32 for LocalAtomAttention."""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 16)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 32)
        return f_input.pad(max_tokens=num_tokens, max_atoms=num_atoms)

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

        # Pre-cropping (on-the-fly pipeline of AlphaFold3 SI Section 2.5.4)
        struct = self.pre_crop_structure(struct, **kwargs)

        # Cropping
        cropped_struct = self.crop_structure(struct, **kwargs)

        # Featurization
        f_input = self.featurize(cropped_struct, record)

        symmetry_dict = {}
        symmetry_dict["id"] = record_id
        if self.return_structure:
            symmetry_dict["structure"] = struct
        if self.return_symmetry:
            # WARN: symmetry computation should be done before padding
            symmetry_dict["symmetry"] = symmetry.get_symmetries(
                f_input, cropped_struct, struct, self.ccd_symmetry_dict
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
            seq_emb_prefix = str(
                self.seq_embedding_path / record_id[:2] / record_id / f"{record_id}_"
            )
        else:
            seq_emb_prefix = None
        if self.struct_embedding_path is not None:
            struct_emb_prefix = str(
                self.struct_embedding_path / record_id[:2] / record_id / f"{record_id}_"
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
        paths: dict[str, Path | None],
        featurization_args: dict,
        max_chains: int,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool = True,
        return_symmetry: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        records : list[metadata.Metadata]
            List of samples to use in the dataset.
        paths : dict[str, Path | None]
            Paths for various resources.
        featurization_args : dict
            Arguments for featurization.
        max_chains : int
            Maximum number of chains per sample.
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
            paths,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=False,
            ccd_symmetry_dict=ccd_symmetry_dict,
        )
        self.max_tokens: int = max_tokens
        self.max_chains: int = max_chains

        self.pre_cropper = PreCropper(PreCropper.Config(max_chains=max_chains))
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
    def pre_crop_structure(
        self,
        struct: structure.TokenizedStructure,
        **kwargs,
    ) -> structure.TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: list[str] = kwargs["asym_ids"]
        if self.max_chains < struct.num_chains:
            # Get sub-complex with limited number of chains
            struct = self.pre_cropper.crop(struct, self.max_tokens, asym_ids)
        return struct

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
        max_tokens = self.max_tokens
        max_chains = max_tokens // 4  # min 4 tokens per chain
        max_atoms = max_tokens * 24  # max 24 atoms per token
        max_bonds = max_tokens * 10  # max 10 bonds per token
        return f_input.pad(max_tokens, max_chains, max_atoms, max_bonds)

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
                return self.get_item(sample.metadata, asym_ids=sample.asym_id)
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
        paths: dict[str, Path | None],
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        records : list[metadata.Metadata]
            List of samples to use in the dataset.
        """
        super().__init__(
            records,
            paths,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=True,
            ccd_symmetry_dict=ccd_symmetry_dict,
        )


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
            struct = structure.TokenizedStructure.load_npz(byte_stream)
        struct = struct.copy_with(metadata=record)
        return struct

    def __del__(self):
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()


class LMDBTrainingDataset(TrainingDataset, LMDBDatabase):
    def __init__(
        self,
        records: list[metadata.Metadata],
        lmdb_path: Path,
        paths: dict[str, Path | None],
        featurization_args: dict,
        max_chains: int,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool = True,
        return_symmetry: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        TrainingDataset.__init__(
            self,
            records,
            paths,
            featurization_args,
            max_chains,
            max_tokens,
            cropper,
            sampler_config,
            safe_load,
            return_symmetry,
            ccd_symmetry_dict,
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
        paths: dict[str, Path | None],
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        ValidationDataset.__init__(
            self,
            records,
            paths,
            featurization_args,
            safe_load,
            return_symmetry,
            ccd_symmetry_dict,
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(
        self, record: metadata.Metadata
    ) -> structure.TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(record)
