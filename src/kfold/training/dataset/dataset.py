"""
Dataset structures:

(Shared across datasets)
ccd-train.pkl

(For each dataset)
rcsb-train/
    manifest.json
    structure.lmdb
    lookup.json  # mapping from each chain to seq-id and apo structure(s).
    seq_embedding/
        esm2/
        esmc/
    struct_embedding/
        saprot/
    apo/
        ESMFold/
            uniq_prot1-esmfold.pdb
            ...
        AFDB/
            F-P01116-F1-model_v6.cif.gz
            ...
afdb-distillation/ ...
rcsb-validation/ ...


* lookup.json format
```json
{
  "6oim": {
    "1": {
      "type": "protein",
      "seq_emb": {
        "path": "uniq_protein_000020.pt",
        "residue_map": "1:250->1:250"
      },
      "struct_emb": {
        "path": "AF-P01116-F1-model_v6.pt",
        "residue_map": "1:235->11:245"
      },
      "apo": [
        {
          "name": "uniq_protein_000020-esmfold",
          "path": "uniq_protein_000020-esmfold.pdb.gz",
          "residue_map": "1:250->1:250",
          "source": "esmfold"
        },
        {
          "name": "AF-P01116-F1-model_v6",
          "path": "AF-P01116-F1-model_v6.cif.gz",
          "residue_map": "1:235->11:245",
          "source": "afdb"
        },
        {
          "name": "51d6-A",
          "path": "51d6-A.pdb.gz",
          "residue_map": "5:250->5:250",
          "source": "pdb"
        }
      ]
    },
    "2": {...}
  },
  "1a2c": {...}
}
```
"""

import dataclasses
import io
import json
from abc import ABC, abstractmethod
from pathlib import Path

import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf
from typing_extensions import override

import kfold.constants as C
from kfold.data.pipelines import apo_initialization, featurization, tokenization
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.registry import Registry

from .cropper import BaseCropper
from .filter import BaseFilter
from .sampler import BaseSampler, Sample
from .utils import pre_crop, symmetry


# === Dataset Classes === #
@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    """Base configuration for dataset.

    Attributes
    ----------
    name : str
        Name of the dataset.
    data_path : str | Path
        Path to the dataset directory.
    seed : int | None
        Random seed for data loading.
    apo_initialize : apo_initialize.ApoInitializerConfig
        Configuration for apo structure initialization.
    """

    name: str
    data_path: str | Path
    seed: int | None = None

    apo_init: apo_initialization.ApoInitializerConfig = dataclasses.field(
        default_factory=apo_initialization.ApoInitializerConfig
    )

    @classmethod
    def from_dict(cls, config) -> "DatasetConfig":
        default_config = OmegaConf.create(cls)
        merged_config = OmegaConf.merge(default_config, OmegaConf.create(config))
        return OmegaConf.to_object(merged_config)


@dataclasses.dataclass(kw_only=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for training dataset.

    Attributes
    ----------
    weight : float
        Weight of the dataset during training.
    filters : list[BaseFilter.Config]
        List of filters to apply to the dataset.
    sampler : BaseSampler.Config | None
        Sampler configuration for generating samples.
    cropper : BaseCropper.Config | None
        Cropper configuration for cropping structures.
    """

    weight: float = 1.0
    filters: list[BaseFilter.Config] = dataclasses.field(default_factory=list)
    sampler: BaseSampler.Config | None
    cropper: BaseCropper.Config | None


@dataclasses.dataclass(kw_only=True)
class ValidationDatasetConfig(DatasetConfig): ...


# Type alias
SymmetryInfo = dict


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class SafeLoadingDataset(torch.utils.data.Dataset, ABC):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        config: DatasetConfig,
        ccd: CCD,
        pretrained_embedding: dict,
        featurization_args: dict,
        return_symmetry: bool = False,
        return_structure: bool = False,
        safe_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        pretrained_embedding : dict
            Pretrained embedding configuration.
        featurization_args : dict
            Additional arguments for featurization.
        return_symmetry : bool
            Whether to return symmetry information.
        return_structure : bool
            Whether to return the original tokenized structure.
        safe_load : bool
            Whether to retry loading on failure.
        """
        # === Initialize parameters === #
        self.config: DatasetConfig = config
        self.name: str = config.name
        self.data_root = Path(config.data_path)
        self.seed: int | None = config.seed
        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure
        self.safe_load: bool = safe_load

        pretrained_embedding: dict = pretrained_embedding.copy()
        featurization_args: dict = featurization_args.copy()

        for k in ["seq", "seq_dim", "struct", "struct_dim", "max_struct_ensembles"]:
            if k not in pretrained_embedding:
                print(
                    f"Warning: Pretrained embedding key '{k}' not found. Setting to None."
                )
                pretrained_embedding[k] = None

        self.seq_embedding: str | None = pretrained_embedding["seq"]
        self.seq_embedding_dim: int | None = pretrained_embedding["seq_dim"]
        self.struct_embedding: str | None = pretrained_embedding["struct"]
        self.struct_embedding_dim: int | None = pretrained_embedding["struct_dim"]
        self.max_struct_ensembles: int = pretrained_embedding["max_struct_ensembles"]

        # === Validate parameters === #
        assert self.data_root.exists(), f"Dataset path {self.data_root} does not exist."
        if self.seq_embedding is not None:
            self.seq_emb_root = self.data_root / "seq_embedding" / self.seq_embedding
            assert self.seq_emb_root.exists(), (
                f"Sequence embedding root {self.seq_emb_root} does not exist."
            )
            assert self.seq_embedding_dim is not None, (
                "seq_embedding_dim must be provided when seq_embedding is set."
            )

        if self.struct_embedding is not None:
            self.struct_emb_root = (
                self.data_root / "struct_embedding" / self.struct_embedding
            )
            assert self.struct_emb_root.exists(), (
                f"Structure embedding root {self.struct_emb_root} does not exist."
            )
            assert self.struct_embedding_dim is not None, (
                "struct_embedding_dim must be provided when struct_embedding is set."
            )

        # Update apo initializer config
        rieprody_lmdb_path = self.data_root / "rieprody_metric.lmdb"
        if rieprody_lmdb_path.exists():
            if config.apo_init.use_perturbation is False:
                # Skip warning if perturbation is disabled
                pass
            elif config.apo_init.apo_perturbation is None:
                print("Warning: RieProDy LMDB path found but apo_perturbation is None.")
            elif config.apo_init.apo_perturbation.rieprody is None:
                print("Warning: RieProDy LMDB path found but rieprody is disabled.")
            else:
                config.apo_init.apo_perturbation.rieprody.metric_lmdb_path = (
                    rieprody_lmdb_path
                )

        # === Load dataset components === #
        # CCD (shared across datasets)
        self.ccd: CCD = ccd

        # Metadata
        self.metadatas: list[Metadata] = self.load_manifest()

        # Lookup table
        self.lookup_table: dict = self.load_lookup_table()

        # === Initialize modules === #
        self.apo_initializer = apo_initialization.ApoInitializer(
            config.apo_init, self.ccd
        )
        self.tokenizer = tokenization.Tokenizer(self.ccd)
        self.featurizer = featurization.InputFeaturizer(
            **featurization_args,
            seq_embedding_dim=self.seq_embedding_dim,
            struct_embedding_dim=self.struct_embedding_dim,
            max_struct_ensembles=self.max_struct_ensembles,
        )

        # Additional setup can be done in subclasses
        self.setup()

    def __len__(self) -> int:
        return len(self.metadatas)

    # === Setup === #
    def load_manifest(self) -> list[Metadata]:
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest_path} not found.")
        with open(manifest_path) as f:
            metadata_dicts: list[dict] = json.load(f)
        metadatas: list[Metadata] = [Metadata.from_dict(d) for d in metadata_dicts]
        del metadata_dicts
        # Ensure all chains and interfaces are valid
        for m in metadatas:
            m.check_all_chains_valid()
            m.check_all_interfaces_valid()
        return metadatas

    def load_lookup_table(self) -> dict:
        lookup_path = self.data_root / "lookup.json"
        with open(lookup_path) as f:
            lookup_table = json.load(f)
        for m in self.metadatas:
            if m.id not in lookup_table:
                raise KeyError(f"Metadata ID {m.id} not found in lookup table.")
        return lookup_table

    def setup(self) -> None:
        """Additional setup for subclasses."""
        pass

    # === Core dataset methods === #
    @abstractmethod
    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""

    def load_apo_structure(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Populate the apo structure for the given reference structure."""
        rng = rng or np.random.default_rng()

        # Fetch apo info from lookup table
        name = ref_struct.metadata.id
        entry_info = self.lookup_table[name]

        apo_dir = self.data_root / "apo"
        apo_lookup_map: dict[int, dict] = {}
        for c in ref_struct.chains:
            entity_id = c.entity_id
            entity_info = entry_info[str(entity_id)]
            if c.ctype.is_protein:
                apo_list: list = entity_info.get("apo", [])

                # Select apo structure (randomly if multiple)
                if len(apo_list) == 0:
                    print(
                        "Warning: No apo structure found for entity "
                        f"{entity_id} in entry {name}."
                    )
                    continue
                elif len(apo_list) == 1:
                    apo_info = apo_list[0]
                else:
                    apo_info = rng.choice(apo_list)

                source = apo_info["source"]
                name = apo_info["name"]
                path = apo_info["path"]
                residue_map = apo_info["residue_map"]

                # Check apo structure file existence
                apo_path = apo_dir / source / path
                if not apo_path.exists():
                    print(
                        "Warning: Apo structure file not found for entity "
                        f"{entity_id} in entry {name}."
                    )
                    continue

                # Get lmdb key
                lmdb_key = f"{source}:{name}"

                # Add to apo lookup map
                apo_lookup_map[entity_id] = {
                    "source": source,
                    "name": name,
                    "path": apo_path,
                    "residue_map": residue_map,
                    "rieprody_key": lmdb_key,
                }

        # Populate apo structure
        self.apo_initializer(ref_struct, apo_lookup_map, rng)

    def tokenize(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator | None = None,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        return self.tokenizer(ref_struct, rng, use_only_cached_conformers=True)

    # === Optional to-override in subclasses === #
    def extract_substructure(
        self,
        struct: RefStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> RefStructure:
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

        # Load structure (NOTE: ref_struct.metadata == metadata)
        ref_struct: RefStructure = self.load_ref_structure(metadata)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        ref_struct = self.extract_substructure(ref_struct, rng=rng, **kwargs)

        # Populate apo structure (in-place)
        self.load_apo_structure(ref_struct, rng=rng)

        # Tokenization
        struct: TokenizedStructure = self.tokenize(ref_struct, rng=rng)

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

    def featurize(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator | None = None,
    ) -> FoldingInput:
        """Featurize the given tokenized structure."""
        if self.seq_embedding is not None:
            # sequence embeddings
            seq_embeddings = self.find_precomputed_embeddings(struct, metadata.id, "seq")
        else:
            seq_embeddings = None

        if self.struct_embedding is not None:
            # structure embeddings
            struct_embeddings = self.find_precomputed_embeddings(
                struct, metadata.id, "struct"
            )
        else:
            struct_embeddings = None

        # Featurization
        f_input = self.featurizer(
            struct,
            seq_embeddings=seq_embeddings,
            struct_embeddings=struct_embeddings,
            rng=rng,
        )
        return f_input

    def find_precomputed_embeddings(
        self, struct: TokenizedStructure, name: str, emb_type: str
    ) -> dict[int, dict]:
        """Find precomputed embeddings for the given structure.

        Parameters
        ----------
        struct : TokenizedStructure
            The tokenized structure.
        name : str
            The name/ID of the structure.
        emb_type : str
            The type of embedding ("seq" or "struct").

        Returns
        -------
        dict[int, dict]
            A dictionary mapping entity IDs to embedding file paths,
            e.g., {entity_id: {"path": Path, "residue_map": str}, ...}
        """
        if emb_type == "seq":
            root_dir = self.seq_emb_root
        elif emb_type == "struct":
            root_dir = self.struct_emb_root

        # Fetch entry info from lookup table
        entry_info = self.lookup_table[name]

        # Get sequence id
        embedding_paths: dict[int, dict] = {}
        for chain_i in range(struct.num_chains):
            entity_id = int(struct.chain.entity_id[chain_i])
            if entity_id in embedding_paths:
                continue  # already found
            ctype = C.ChainType(struct.chain.chain_type[chain_i].item())
            if ctype.is_polymer:
                # Get embedding for polymer chain
                entity_info = entry_info[str(entity_id)]
                emb_id_info = (
                    entity_info.get("seq_emb")
                    if emb_type == "seq"
                    else entity_info.get("struct_emb")
                )
                if emb_id_info is not None:
                    emb_path = root_dir / emb_id_info["path"]
                    residue_map = emb_id_info["residue_map"]
                    embedding_paths[entity_id] = {
                        "path": emb_path,
                        "residue_map": residue_map,
                    }
                else:
                    # FIXME: Temporary warning for missing protein embeddings
                    if ctype.is_protein:
                        print(
                            f"Warning: Missing {emb_type} embedding for entity "
                            f"{entity_id} in entry {name}."
                        )
        return embedding_paths


class LMDBDataset(SafeLoadingDataset):
    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self.lmdb_path = self.data_root / "structure.lmdb"
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                map_size=1024**4,  # 1 TB
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_env

    def __del__(self):
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()

    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""
        name = metadata.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB.")

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            ref_struct = RefStructure.load_npz(byte_stream)

        # NOTE: Validate loaded record matches requested metadata
        # If there is no problem, only the cluster ID should differ.
        ref_metadata = ref_struct.metadata
        assert ref_metadata.id == name, (
            f"Loaded record ID {ref_metadata.id} does not match requested ID {name}."
        )
        assert ref_metadata.num_chains == metadata.num_chains, (
            f"Loaded record num_chains {ref_metadata.num_chains} does not match "
            f"requested num_chains {metadata.num_chains}."
        )
        assert ref_metadata.num_residues == metadata.num_residues, (
            f"Loaded record num_residues {ref_metadata.num_residues} does not match "
            f"requested num_residues {metadata.num_residues}."
        )
        assert ref_metadata.num_interfaces == metadata.num_interfaces, (
            f"Loaded record num_interfaces {ref_metadata.num_interfaces} does not match "
            f"requested num_interfaces {metadata.num_interfaces}."
        )
        return ref_struct


class TrainingDataset(LMDBDataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        ccd: CCD,
        pretrained_embedding: dict,
        featurization_args: dict,
        safe_load: bool = True,
        max_chains: int = 20,
        max_tokens: int = 384,
    ) -> None:
        """
        Parameters
        ----------
        config : TrainingDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        pretrained_embedding : dict
            Pretrained embedding configuration.
        featurization_args : dict
            Additional arguments for featurization.
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample. Must be a multiple of 64 for
            LocalAtomAttention.

        Notes
        -----
        This dataset implements AF3-style sampling (chain/interface-based).
        1. Samples are generated based on chains/interfaces in the structures.
        2. During data loading, samples are cropped to fit within `max_tokens`
           using the provided `cropper`.
        """
        super().__init__(
            config,
            ccd,
            pretrained_embedding,
            featurization_args,
            return_symmetry=False,
            return_structure=False,
            safe_load=safe_load,
        )
        if self.seed is not None:
            # Warn about fixed seed affecting randomness
            print(
                "WARNING: Seed is set for TrainingDataset, which may affect randomness."
            )
        self.max_tokens: int = max_tokens
        self.max_chains: int = max_chains

        # Initialize filters
        self.filters: list[BaseFilter] = [
            Registry.instantiate(config=c) for c in config.filters
        ]

        def do_filter(m: Metadata) -> bool:
            return all(filt(m) for filt in self.filters)

        self.metadatas = [m for m in self.metadatas if do_filter(m)]

        assert config.cropper is not None, "Cropper config must be provided."
        self.cropper: BaseCropper = Registry.instantiate(config.cropper)

        assert self.max_tokens % 64 == 0, f"max_tokens must be a multiple of {64}."

        # AF3-style sampling (chain/interface-based)
        assert config.sampler is not None, "Sampler config must be provided."
        sampler: BaseSampler = Registry.instantiate(config.sampler)
        samples, weights = sampler.get_samples(self.metadatas)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray = weights

        self.setup()

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def extract_substructure(
        self,
        struct: RefStructure,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> RefStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_chains < struct.num_chains:
            # Get sub-complex with limited number of chains
            struct = pre_crop.extract_substructure(
                struct,
                max_chains=self.max_chains,
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


class MultiTrainingDataset(torch.utils.data.Dataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        configs: list[TrainingDatasetConfig],
        ccd: CCD,
        pretrained_embedding: dict,
        featurization_args: dict,
        safe_load: bool = True,
        max_chains: int = 20,
        max_tokens: int = 384,
    ) -> None:
        """
        Parameters
        ----------
        configs : list[TrainingDatasetConfig]
            List of dataset configurations.
        ccd: CCD
            CCD database
        pretrained_embedding : dict
            Pretrained embedding configuration.
        featurization_args : dict
            Additional arguments for featurization.
        safe_load : bool
            Whether to retry loading on failure.
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample. Must be a multiple of 64 for
            LocalAtomAttention.

        Notes
        -----
        This dataset implements AF3-style sampling (chain/interface-based).
        1. Samples are generated based on chains/interfaces in the structures.
        2. During data loading, samples are cropped to fit within `max_tokens`
           using the provided `cropper`.
        """
        self.datasets: list[TrainingDataset] = [
            TrainingDataset(
                config,
                ccd,
                pretrained_embedding,
                featurization_args,
                safe_load,
                max_chains,
                max_tokens,
            )
            for config in configs
        ]
        self.cumulative_sizes: np.ndarray = np.cumsum([len(ds) for ds in self.datasets])
        self.weights: np.ndarray = np.concatenate(
            [
                config.weight * (ds.weights / ds.weights.sum())
                for ds, config in zip(self.datasets, configs, strict=True)
            ]
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> tuple[FoldingInput, SymmetryInfo]:
        """Get the folding input for the given index."""
        # Find the dataset index
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side="right")
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ValidationDataset(LMDBDataset):
    """Validation dataset without sampling and cropping."""

    def __init__(
        self,
        config: ValidationDatasetConfig,
        ccd: CCD,
        pretrained_embedding: dict,
        featurization_args: dict,
        safe_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        config : ValidationDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        pretrained_embedding : dict
            Pretrained embedding configuration.
        featurization_args : dict
            Additional arguments for featurization.
        """
        super().__init__(
            config,
            ccd,
            pretrained_embedding,
            featurization_args,
            return_symmetry=True,
            return_structure=True,
            safe_load=safe_load,
        )

    def setup(self) -> None:
        """Additional setup for subclasses."""
        self.metadatas.sort(key=lambda m: m.num_residues)
