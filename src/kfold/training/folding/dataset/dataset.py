import io
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
import numpy as np
import torch
from typing_extensions import override

from kfold.data import apo_perturbation, featurize, metadata, model_input, structure
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
        apo_perturbation_args: dict,
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        return_structure: bool = False,
        ccd_symmetry_dict: dict | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        records : list[metadata.Metadata]
            List of samples to use in the dataset.
        paths : dict[str, Path | None]
            Paths for various resources.
        apo_perturbation_args : dict
            Arguments for apo perturbation.
        featurization_args : dict
            Arguments for featurization.
        safe_load : bool
            Whether to enable safe loading with retries on failure.
        return_symmetry : bool
            Whether to return symmetry information.
        return_structure : bool
            Whether to return the original tokenized structure.
        ccd_symmetry_dict : dict | None
            Dictionary mapping CCD IDs to symmetry information.
        seed : int | None
            Random seed for reproducibility.
        """
        self.records: list[metadata.Metadata] = records
        self.safe_load: bool = safe_load
        self.seed: int | None = seed

        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure

        if self.return_symmetry:
            assert ccd_symmetry_dict is not None, (
                "ccd_symmetry_dict must be provided when return_symmetry is True."
            )
            self.ccd_symmetry_dict: dict = ccd_symmetry_dict

        # Initialize featurizer and apo perturbation
        self.apo_perturbation: apo_perturbation.ApoPerturbation = (
            apo_perturbation.ApoPerturbation(
                **apo_perturbation_args,
                ccd_symmetry_dict=ccd_symmetry_dict,
            )
        )
        self.featurizer: featurize.InputFeaturizer = featurize.InputFeaturizer(
            **featurization_args,
            seq_embedding_path=paths["seq_embedding_path"],
            struct_embedding_path=paths["struct_embedding_path"],
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
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> structure.TokenizedStructure:
        """Pre-crop the folding input structure as needed.
        See Section 2.5.4 of AlphaFold3 SI

        In contrast to `crop_structure`, this method is intended for
        sampling sub-complexes from the original structure before applying
        the main cropping strategy.
        """
        return struct

    def crop_structure(
        self,
        struct: structure.TokenizedStructure,
        rng: np.random.Generator | None = None,
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
        return self.get_item_safe(index, num_trials=100)

    def get_item_safe(
        self,
        index: int,
        num_trials: int = 100,
    ) -> tuple[model_input.FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index, with retry on failure."""
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + index % (1 << 15))
        else:
            rng = np.random.default_rng()

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
                index = int(rng.integers(0, len(self)))
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
        record_id: str = record.id

        # Initialize random number generator (create new rng based on record_id)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + hash(record_id) % (1 << 15))
        else:
            rng = np.random.default_rng()

        # Load tokenized structure
        struct = self.load_tokenized_structure(record)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        struct = self.pre_crop_structure(struct, rng=rng, **kwargs)

        # Apo perturbation
        struct = self.augment_apo_structure(struct, rng=rng)

        # Cropping
        cropped_struct = self.crop_structure(struct, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped_struct, record, rng=rng)

        symmetry_dict = {}
        symmetry_dict["id"] = record_id
        if self.return_structure:
            symmetry_dict["structure"] = struct
        if self.return_symmetry:
            # WARN: symmetry computation should be done before padding
            symmetry_dict["symmetry"] = symmetry.get_symmetries(
                f_input, cropped_struct, struct, self.ccd_symmetry_dict, rng=rng
            )

        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        return f_input, symmetry_dict

    def augment_apo_structure(
        self,
        struct: structure.TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> structure.TokenizedStructure:
        """Apply random perturbation/rotation to apo structure"""
        return self.apo_perturbation.run(struct, rng=rng)

    def featurize(
        self,
        struct: structure.TokenizedStructure,
        record: metadata.Metadata,
        rng: np.random.Generator | None = None,
    ) -> model_input.FoldingInput:
        """Featurize the given tokenized structure."""
        # Featurization
        record_id = record.id
        # HACK: This is the rule to save the pre-computed features in the directory.
        # e.g., "{seq_embedding_path}/4l/4l8g/4l8g_*"
        prefix = f"{record_id[:2]}/{record_id}/{record_id}_"
        f_input = self.featurizer.run(struct, prefix, rng=rng)
        return f_input


class TrainingDataset(SafeLoadingDataset):
    def __init__(
        self,
        records: list[metadata.Metadata],
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
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
        apo_perturbation_args : dict
            Arguments for apo perturbation.
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
        safe_load : bool
            Whether to enable safe loading with retries on failure.
        return_symmetry : bool
            Whether to return symmetry information.

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
            apo_perturbation_args,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=False,
            ccd_symmetry_dict=ccd_symmetry_dict,
            seed=None,  # Do not fix seed for training dataset
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
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> structure.TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_chains < struct.num_chains:
            # Get sub-complex with limited number of chains
            struct = self.pre_cropper.crop(
                struct,
                self.max_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return struct

    @override
    def crop_structure(
        self,
        struct: structure.TokenizedStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> structure.TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_tokens < struct.num_tokens:
            # Crop the tokenized structure
            struct = self.cropper.crop(
                struct,
                self.max_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
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
        apo_perturbation_args: dict,
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
        # To validate the folding performance in usage scenario, where holo
        # structures are not available, we disable symmetry correction and
        # holo replacement during apo perturbation.
        apo_perturbation_args = apo_perturbation_args.copy()
        apo_perturbation_args["use_symmetry_correction"] = False
        apo_perturbation_args["prob_replace_to_holo"] = 0.0

        super().__init__(
            records,
            paths,
            apo_perturbation_args,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=True,
            ccd_symmetry_dict=ccd_symmetry_dict,
            seed=42,  # Fix seed for validation dataset
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
        apo_perturbation_args: dict,
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
            apo_perturbation_args,
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
        apo_perturbation_args: dict,
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        ccd_symmetry_dict: dict | None = None,
    ) -> None:
        ValidationDataset.__init__(
            self,
            records,
            paths,
            apo_perturbation_args,
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
