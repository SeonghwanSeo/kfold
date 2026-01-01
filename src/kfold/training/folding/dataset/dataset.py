import io
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
import numpy as np
import torch
from typing_extensions import override

from kfold.data.ccd import CCD
from kfold.data.model_input import FoldingInput
from kfold.data.pipelines import apo_perturbation, featurize, tokenize
from kfold.data.schema import Metadata
from kfold.data.structure import RefStructure
from kfold.data.tokenized import TokenizedStructure
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
        metadatas: list[Metadata],
        ccd: CCD,
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
        return_structure: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        metadatas : list[Metadata]
            List of samples to use in the dataset.
        ccd: CCD
            CCD database
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
        seed : int | None
            Random seed for reproducibility.
        """
        self.metadatas: list[Metadata] = metadatas
        self.safe_load: bool = safe_load
        self.seed: int | None = seed
        self.ccd: CCD = ccd

        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure

        # Initialize featurizer and apo perturbation
        self.apo_perturbation: apo_perturbation.ApoPerturbation = (
            apo_perturbation.ApoPerturbation(
                ccd=ccd,
                **apo_perturbation_args,
            )
        )
        self.featurizer: featurize.InputFeaturizer = featurize.InputFeaturizer(
            **featurization_args,
        )
        # additional paths
        self.paths: dict[str, Path | None] = paths

    def __len__(self) -> int:
        return len(self.metadatas)

    # === TO-DO Implement in subclasses === #
    @abstractmethod
    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""

    def tokenize(
        self,
        struct: RefStructure,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        return tokenize.tokenize_structure(
            struct,
            ccd=self.ccd,
            rng=rng,
            use_only_cached_conformers=True,
        )

    # === Optional to-override in subclasses === #
    def pre_crop_structure(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> TokenizedStructure:
        """Pre-crop the folding input structure as needed.
        See Section 2.5.4 of AlphaFold3 SI

        In contrast to `crop_structure`, this method is intended for
        sampling sub-complexes from the original structure before applying
        the main cropping strategy.
        """
        return struct

    def crop_structure(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> TokenizedStructure:
        """Crop the folding input structure as needed."""
        return struct

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 32 for LocalAtomAttention."""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 16)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 32)
        return f_input.pad(max_tokens=num_tokens, max_atoms=num_atoms)

    def __getitem__(self, index: int) -> tuple[FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index, with retry on failure."""
        return self.get_item_safe(index, num_trials=100)

    def get_item_safe(
        self,
        index: int,
        num_trials: int = 100,
    ) -> tuple[FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index, with retry on failure."""
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + index % (1 << 15))
        else:
            rng = np.random.default_rng()

        trials = []
        for _ in range(num_trials):
            sample: Metadata = self.metadatas[index]
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
        metadata: Metadata,
        **kwargs,
    ) -> tuple[FoldingInput, SymmetryInfo]:
        """Get the folding input for the given sample."""
        metadata_id: str = metadata.id

        # Initialize random number generator (create new rng based on metadata_id)
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + hash(metadata_id) % (1 << 15))
        else:
            rng = np.random.default_rng()

        # Load structure
        ref_struct = self.load_ref_structure(metadata)

        # Tokenization
        struct = self.tokenize(ref_struct, rng=rng)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        struct = self.pre_crop_structure(struct, rng=rng, **kwargs)

        # Apo perturbation
        struct = self.augment_apo_structure(struct, rng=rng)

        # Cropping
        cropped_struct = self.crop_structure(struct, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped_struct, metadata, rng=rng)

        symmetry_dict = {}
        symmetry_dict["id"] = metadata_id
        if self.return_structure:
            symmetry_dict["structure"] = struct
        if self.return_symmetry:
            # WARN: symmetry computation should be done before padding
            symmetry_dict["symmetry"] = symmetry.get_symmetries(
                f_input, cropped_struct, struct, self.ccd, rng=rng
            )

        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        return f_input, symmetry_dict

    def augment_apo_structure(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        """Apply random perturbation/rotation to apo structure"""
        return self.apo_perturbation(struct, rng=rng)

    def featurize(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        """Featurize the given tokenized structure."""
        # Get precomputed embeddings if available
        # sequence embeddings
        seq_emb_root = self.paths.get("seq_embedding_path", None)
        seq_embedding_paths = self.find_precomputed_embeddings(
            struct, metadata.id, seq_emb_root
        )

        # structure embeddings
        struct_emb_root = self.paths.get("struct_embedding_path", None)
        struct_embedding_paths = self.find_precomputed_embeddings(
            struct, metadata.id, struct_emb_root
        )

        # Featurization
        f_input = self.featurizer(
            struct,
            seq_embedding_paths=seq_embedding_paths,
            struct_embedding_paths=struct_embedding_paths,
            rng=rng,
        )
        return f_input

    def find_precomputed_embeddings(
        self,
        struct: TokenizedStructure,
        name: str,
        root_dir: Path | None = None,
    ) -> dict[int, Path] | None:
        """Find precomputed embeddings for the given structure.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        name : str
            The name/ID of the structure.
        root_dir : Path
            The root directory containing precomputed embeddings.

        Returns
        -------
        dict[int, Path] | None
            A dictionary mapping entity IDs to embedding file paths,
            or None if no embeddings are found.
        """
        if root_dir is None:
            return None

        # Check existence
        # if name='6oim', possible paths are:
        # - {root_dir}/6o/6oim/* ...
        # - {root_dir}/oi/6oim/* ...
        subdir = root_dir / name[1:3] / name
        if subdir.exists():
            assert subdir.is_dir()
        else:
            subdir = root_dir / name[0:2] / name
            if not subdir.exists():
                return {}
            assert subdir.is_dir()

        embedding_paths: dict[int, Path] = {}
        for path in subdir.iterdir():
            if not path.is_file():
                continue
            # Expecting filenames like:
            #   {pdb_id}_{entity_id}_{chain_type}.pt
            pdb_id, entity_id, chain_type = path.stem.split("_")
            assert pdb_id == name, f"Unexpected pdb_id {pdb_id} in embedding file {path}."
            entity_id_int = int(entity_id)
            embedding_paths[entity_id_int] = path

        return embedding_paths


class TrainingDataset(SafeLoadingDataset):
    def __init__(
        self,
        metadatas: list[Metadata],
        ccd: CCD,
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
        featurization_args: dict,
        max_chains: int,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool = True,
        return_symmetry: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        metadatas : list[Metadata]
            List of samples to use in the dataset.
        ccd: CCD
            CCD database
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
            metadatas,
            ccd,
            paths,
            apo_perturbation_args,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=False,
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
        samples, weights = sampler.get_samples(metadatas)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray | None = weights

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def pre_crop_structure(
        self,
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> TokenizedStructure:
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
        struct: TokenizedStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> TokenizedStructure:
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
    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
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
    ) -> tuple[FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.metadatas`.
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
        metadatas: list[Metadata],
        ccd: CCD,
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        metadatas : list[Metadata]
            List of samples to use in the dataset.
        """
        # To validate the folding performance in usage scenario, where holo
        # structures are not available, we disable symmetry correction,
        # holo replacement, and perturbation during apo perturbation.
        apo_perturbation_args = apo_perturbation_args.copy()
        apo_perturbation_args["use_symmetry_correction"] = False
        apo_perturbation_args["prob_replace_to_holo"] = 0.0
        apo_perturbation_args["use_perturbation"] = False

        super().__init__(
            metadatas,
            ccd,
            paths,
            apo_perturbation_args,
            featurization_args,
            safe_load,
            return_symmetry,
            return_structure=True,
            seed=42,  # Fix seed for validation dataset
        )


class LMDBDatabase:
    """
    Provides LMDB-backed access to tokenized structures.
    The `lmdb_env` property lazily initializes and caches the LMDB environment
    on first access, ensuring efficient resource usage. The `load_from_lmdb`
    method retrieves a tokenized structure from the LMDB database using a
    metadata's ID as the key.
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

    def load_from_lmdb(self, metadata: Metadata) -> TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        name = metadata.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB at {self.lmdb_path}.")

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            struct = TokenizedStructure.load_npz(byte_stream)
        struct = struct.copy_with(metadata=metadata)
        return struct

    def __del__(self):
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()


class LMDBTrainingDataset(TrainingDataset, LMDBDatabase):
    def __init__(
        self,
        metadatas: list[Metadata],
        lmdb_path: Path,
        ccd: CCD,
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
        featurization_args: dict,
        max_chains: int,
        max_tokens: int,
        cropper: BaseCropper,
        sampler_config: BaseSampler.Config | None,
        safe_load: bool = True,
        return_symmetry: bool = False,
    ) -> None:
        TrainingDataset.__init__(
            self,
            metadatas,
            ccd,
            paths,
            apo_perturbation_args,
            featurization_args,
            max_chains,
            max_tokens,
            cropper,
            sampler_config,
            safe_load,
            return_symmetry,
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(self, metadata: Metadata) -> TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(metadata)


class LMDBValidationDataset(ValidationDataset, LMDBDatabase):
    def __init__(
        self,
        metadatas: list[Metadata],
        lmdb_path: Path,
        ccd: CCD,
        paths: dict[str, Path | None],
        apo_perturbation_args: dict,
        featurization_args: dict,
        safe_load: bool = True,
        return_symmetry: bool = False,
    ) -> None:
        ValidationDataset.__init__(
            self,
            metadatas,
            ccd,
            paths,
            apo_perturbation_args,
            featurization_args,
            safe_load,
            return_symmetry,
        )
        self.lmdb_path: Path = lmdb_path

    def load_tokenized_structure(self, metadata: Metadata) -> TokenizedStructure:
        """Load the tokenized structure from LMDB."""
        return self.load_from_lmdb(metadata)
