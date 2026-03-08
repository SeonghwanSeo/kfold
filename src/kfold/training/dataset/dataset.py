"""
Dataset structures:

(Shared across datasets)
ccd-train.pkl

(For each dataset)
rcsb-train/
    manifest.json
    structure.lmdb
    apo.lmdb            # apo structures for each chain.
      - seq: np.ndarray of shape (L,), dtype S1
      - coords: np.ndarray of shape (L, 37, 3), dtype float32
    apo_unitok.lmdb     # pre-computed structure tokens for apo structures.
    apo_lookup.json     # mapping from each chain to apo structure(s).
afdb-distillation/ ...  # no apo.lmdb or apo_lookup.json (label=apo)
    manifest.json
    structure.lmdb/
    apo_unitok.lmdb     # pre-computed structure tokens for apo structures.
rcsb-val/ ...


* apo_lookup.json format
```json
{
  "6oim": {
    "1": [
      {
        "source": "esmfold"
        "name": "uniq_protein_000020-esmfold",
        "residue_map": "1:250->1:250",
      },
      {
        "source": "afdb"
        "name": "AF-P01116-F1-model_v6",
        "residue_map": "1:235->11:245",
      },
      {
        "source": "pdb"
        "name": "51d6-A",
        "residue_map": "5:250->5:250",
      }
    ],
    "2": [...]
  },
  "1a2c": {...}
}
```
"""

import dataclasses
import io
import json
import logging
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch
from omegaconf import OmegaConf
from typing_extensions import override

import kfold.constants as C
from kfold.data.pipelines import (
    apo_initialization,
    featurization,
    prior_sampling,
    sequence_masking,
    tokenization,
)
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.misc import hash_seq
from kfold.utils.registry import Registry

from .cropper import BaseCropper
from .sampler import BaseSampler, Sample
from .utils import pre_crop, symmetry


def _open_lmdb(lmdb_path: str | Path) -> lmdb.Environment:
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found.")
    return lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )


def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example:
        "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, apo_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    apo_st, apo_end = map(int, apo_range.split(":"))
    if (res_end - res_st) != (apo_end - apo_st):
        return -1, -1, -1, -1  # invalid mapping
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    return res_st - 1, res_end, apo_st - 1, apo_end


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
    manifest_path : str | Path | None
        Optional path to the custom manifest file.
    seed : int | None
        Random seed for data loading.
    is_protein_monomer_distillation : bool
        Whether this is large-scale protein monomer synthetic data,
        such as AFDB or ESMAtlas. This flag can be used to enable
        specific handling for monomer distillation data, such as
        feeding apo structures from labeled monomer structures.
    apo_init : ApoInitializerConfig
        Configuration for apo structure initialization.
    """

    name: str
    data_path: str | Path
    manifest_path: str | Path | None = None
    seed: int | None = None
    is_protein_monomer_distillation: bool = False
    apo_init: apo_initialization.ApoInitializerConfig
    prior_sampler: prior_sampling.PriorSamplerConfig | None

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
    sampler : BaseSampler.Config | None
        Sampler configuration for generating samples.
    cropper : BaseCropper.Config | None
        Cropper configuration for cropping structures.
    """

    weight: float = 1.0
    sampler: BaseSampler.Config | None
    cropper: BaseCropper.Config | None


@dataclasses.dataclass(kw_only=True)
class ValidationDatasetConfig(DatasetConfig): ...


# Type alias
SymmetryInfo = dict


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class SafeLoadingDataset(torch.utils.data.Dataset):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        config: DatasetConfig,
        ccd: CCD,
        return_symmetry: bool,
        return_structure: bool,
        safe_load: bool,
        train: bool,
    ) -> None:
        """
        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
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
        self.data_root: Path = Path(config.data_path)
        self.seed: int | None = config.seed
        self.return_symmetry: bool = return_symmetry
        self.return_structure: bool = return_structure
        self.safe_load: bool = safe_load

        if train:
            self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")
        else:
            self.logger = logging.getLogger(f"[Validation Dataset:{self.name}]")

        # Sanity check on dataset files and configurations
        self.sanity_check()

        # Flag for specific handling of protein monomer distillation datasets
        self.is_protein_monomer_distillation: bool = (
            config.is_protein_monomer_distillation
        )

        # === Load dataset components === #
        # CCD (shared across datasets)
        self.ccd: CCD = ccd

        # Metadata list
        # NOTE: For AFDB distillation, which includes a lot of samples,
        # we keep a dict instead of Metadata obj to save memory.
        self.metadatas: list[dict] = self.load_manifest(
            custom_manifest=config.manifest_path  # optional custom manifest path
        )

        # Lookup table
        self.lookup_table: dict = self.load_lookup_table()

        # === Initialize modules === #
        self.apo_initializer = apo_initialization.ApoInitializer(
            config.apo_init, self.ccd, self.is_protein_monomer_distillation
        )

        if config.prior_sampler is not None:
            # For diffusion bridge model, we may want to sample prior structures
            # from apo structures with ot-permutation.
            self.prior_sampler = prior_sampling.PriorSampler(
                config.prior_sampler, self.ccd
            )
        else:
            # For regular edm, we don't need to sample prior structures since
            # the prior distribution is gaussian.
            self.prior_sampler = None

        self.tokenizer = tokenization.Tokenizer(self.ccd, self.prior_sampler)
        self.featurizer = featurization.InputFeaturizer()

        # Additional setup can be done in subclasses
        self.setup()

    def __len__(self) -> int:
        return len(self.metadatas)

    # === Setup === #
    def load_manifest(self, custom_manifest: str | Path | None = None) -> list[dict]:
        if custom_manifest is not None:
            manifest_path = Path(custom_manifest)
        else:
            manifest_path = self.data_root / "manifest.msgpack"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest_path} not found.")

        if manifest_path.suffix == ".msgpack":
            with open(manifest_path, "rb") as f:
                metadata_dicts: list[dict] = msgpack.unpack(f)
        else:
            with open(manifest_path) as f:
                metadata_dicts: list[dict] = json.load(f)
        return metadata_dicts

    def load_lookup_table(self) -> dict:
        if self.is_protein_monomer_distillation:
            # For protein monomer distillation datasets, we directly
            # feed apo structures from labeled monomer structures.
            self.logger.info(
                "Protein monomer distillation dataset detected. "
                "Skipping apo lookup table loading."
            )
            return {}

        lookup_path = self.data_root / "apo_lookup.msgpack"
        if not lookup_path.exists():
            # NOTE: For protein monomer distillation datasets,
            # we can directly feed apo structures from labeled monomer structures.
            raise FileNotFoundError(f"Apo lookup file {lookup_path} not found.")
        with open(lookup_path, "rb") as f:
            lookup_table: dict = msgpack.unpack(f)
        return lookup_table

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        pass

    def setup(self) -> None:
        """Additional setup for subclasses."""
        pass

    @property
    def lmdb_env(self) -> lmdb.Environment:
        if not hasattr(self, "_lmdb_env"):
            self._lmdb_env = _open_lmdb(self.data_root / "structure.lmdb")
        return self._lmdb_env

    @property
    def apo_lmdb_env(self) -> lmdb.Environment:
        """Get the LMDB environment for apo structures."""
        if not hasattr(self, "_apo_lmdb_env"):
            self._apo_lmdb_env = _open_lmdb(self.data_root / "apo.lmdb")
        return self._apo_lmdb_env

    @property
    def unitok_lmdb_env(self) -> lmdb.Environment:
        """Get the LMDB environment for structure tokens of apo structures."""
        if not hasattr(self, "_unitok_lmdb_env"):
            self._unitok_lmdb_env = _open_lmdb(self.data_root / "apo_unitok.lmdb")
        return self._unitok_lmdb_env

    def __del__(self):
        if hasattr(self, "_apo_lmdb_env"):
            self._apo_lmdb_env.close()
        if hasattr(self, "_unitok_lmdb_env"):
            self._unitok_lmdb_env.close()
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()

    # === Core dataset methods === #
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
        # If there is no problem, the cluster ID (for training) and
        # low_homology flag (for validation) will be missing in npz
        ref_metadata = ref_struct.metadata
        assert ref_metadata.id == name, (
            f"Loaded ID {ref_metadata.id} does not match requested ID {name}."
        )
        assert ref_metadata.num_chains == metadata.num_chains, (
            f"Loaded num_chains {ref_metadata.num_chains} does not match "
            f"requested num_chains {metadata.num_chains}."
        )
        assert ref_metadata.num_residues == metadata.num_residues, (
            f"Loaded num_residues {ref_metadata.num_residues} does not match "
            f"requested num_residues {metadata.num_residues}."
        )
        assert ref_metadata.num_interfaces == metadata.num_interfaces, (
            f"Loaded num_interfaces {ref_metadata.num_interfaces} does not match "
            f"requested num_interfaces {metadata.num_interfaces}."
        )
        # Copy metadata (to update cluster_id if needed)
        ref_struct.metadata = metadata.copy()
        return ref_struct

    def load_apo_structure(
        self,
        ref_struct: RefStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Populate the apo structure for the given reference structure."""
        self.apo_initializer(ref_struct, apo_lookup, rng)

    def tokenize(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        # only use cached conformer tokens to avoid
        # ETKDG conformer generation during training/validation.
        use_cached_conformer_only = True
        # TODO: add apo info
        return self.tokenizer(ref_struct, {}, rng, use_cached_conformer_only)

    # === Optional to-override in subclasses === #
    def extract_substructure(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
        **kwargs,
    ) -> RefStructure:
        """Pre-crop the folding input structure as needed.
        See Section 2.5.4 of AlphaFold3 SI

        In contrast to `crop_structure`, this method is intended for
        sampling sub-complexes from the original structure before applying
        the main cropping strategy.
        """
        return ref_struct

    def crop_structure(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> TokenizedStructure:
        """Crop the folding input structure as needed."""
        return struct

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 32 for LocalAtomAttention."""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 16)
        # Pad num_seq_tokens for CUDA efficiency.
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 64)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 32)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence_tokens=num_sequence_tokens,
        )

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
            sample: Metadata = Metadata.from_dict(self.metadatas[index])
            try:
                return self.get_item(sample)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                sample_id = sample.id
                self.logger.error(
                    f"Error loading index {sample_id}({index}): {e}. Retrying..."
                )
                if not self.safe_load:
                    raise e
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
            offset = int(hash_seq(metadata.id), 16)
            rng = np.random.default_rng((self.seed + offset) % (1 << 32))
        else:
            rng = np.random.default_rng()

        # Load structure (NOTE: ref_struct.metadata == metadata)
        ref_struct: RefStructure = self.load_ref_structure(metadata)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        ref_struct = self.extract_substructure(ref_struct, rng=rng, **kwargs)
        metadata = ref_struct.metadata  # update metadata after extraction

        # Get apo lookup for the structure
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Populate apo structure (in-place)
        self.load_apo_structure(ref_struct, apo_lookup, rng)

        # Tokenization
        struct: TokenizedStructure = self.tokenize(ref_struct, rng=rng)

        # Populate structure tokens for apo structure (in-place)
        self.populate_structure_tokens(struct, apo_lookup, rng)

        # Cropping
        cropped_struct = self.crop_structure(struct, metadata, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped_struct, metadata, rng=rng)

        struct_info = {}
        struct_info["id"] = metadata_id
        if self.return_structure:
            struct_info["structure"] = ref_struct
        if self.return_symmetry:
            # WARN: symmetry computation should be done before padding
            struct_info["symmetry"] = symmetry.get_symmetries(
                ref_struct,
                self.ccd,
                max_chain_permutations=1000,
                rng=rng,
            )

        # Pad the folding input to multiple of 64 for LocalAtomAttention
        f_input = self.pad_input(f_input)

        return f_input, struct_info

    def featurize(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
    ) -> FoldingInput:
        """Featurize the given tokenized structure."""
        # Featurization
        f_input = self.featurizer(struct, rng=rng)
        return f_input

    # === Helper methods for apo structure handling === #
    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        if self.is_protein_monomer_distillation:
            # For protein monomer distillation datasets, we directly
            # feed apo structures from labeled monomer structures, so
            # we don't have to load apo structures.

            # NOTE: we still need the lookup table to load structure
            # tokens for apo structures.
            # monomer: always have a chain with entity_id=1
            return {1: {"key": ref_struct.id}}

        entry_id: str = ref_struct.id
        entry_lookup: dict[str, list[dict[str, str]]] = self.lookup_table[entry_id]

        # Match apo structure for each protein entries.
        apo_lookup: dict[int, dict] = {}  # entity_id -> apo_info dict
        visited_entity_ids: set[int] = set()
        for c in ref_struct.chains:
            if not c.ctype.is_protein:
                continue
            if c.entity_id in visited_entity_ids:
                continue  # already populated from another chain with same entity_id
            entity_id = c.entity_id
            visited_entity_ids.add(entity_id)

            entity_apo_infos: list[dict[str, str]] = entry_lookup[str(c.entity_id)]
            num_apos = len(entity_apo_infos)
            # Select apo structure (randomly if multiple)
            if num_apos == 0:
                self.logger.warning(
                    f"No apo info found for entity {entry_id}:{entity_id}"
                )
                continue
            apo_info = entity_apo_infos[rng.integers(0, num_apos)].copy()

            name = apo_info["name"]
            source = apo_info["source"]
            key = f"{source}:{name}"
            apo_info["key"] = key

            # Load apo coordinates from LMDB
            with self.apo_lmdb_env.begin(write=False) as txn:
                value_bytes = txn.get(key.encode("utf-8"))
                if value_bytes is None:
                    self.logger.warning(
                        f"Apo structure {key} not found in LMDB for entity "
                        f"{entry_id}:{entity_id}"
                    )
                    continue
                with io.BytesIO(value_bytes) as byte_stream:
                    with np.load(byte_stream) as data:
                        apo_info["seq"] = "".join(data["seq"].astype(str).tolist())
                        apo_info["coords"] = data["coords"].copy()

            apo_lookup[entity_id] = apo_info
        return apo_lookup

    def populate_structure_tokens(
        self,
        struct: TokenizedStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Populate the structure tokens for the given tokenized structure."""

        bb_struct_token_id = struct.sequence.bb_struct_token_id
        fa_struct_token_id = struct.sequence.fa_struct_token_id

        visited_entity_ids: set[int] = set()
        with self.unitok_lmdb_env.begin(write=False) as txn:
            for c_i in range(struct.num_chains):
                if struct.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                    continue  # only populate structure tokens for protein chains

                eid = struct.chain.entity_id[c_i]
                if eid in visited_entity_ids:
                    continue  # already populated from another chain with same entity_id
                visited_entity_ids.add(eid)

                if eid not in apo_lookup:
                    self.logger.warning(
                        f"No apo info for entity_id {eid} in apo lookup."
                        f"Skipping structure token population for this entity."
                    )
                    continue
                apo_info = apo_lookup[eid]
                key = apo_info["key"]
                v = txn.get(key.encode("utf-8"))
                if v is None:
                    self.logger.warning(
                        f"Apo structure tokens {key} not found in LMDB for "
                        f"entity_id {eid}"
                    )
                    continue
                # Load pre-computed structure tokens for apo structure from LMDB
                apo_unitok = np.frombuffer(v, dtype=np.uint16).reshape(2, -1)
                bb_tok, fa_tok = apo_unitok
                toklen = len(bb_tok)

                # Find the corresponding sequence token indices
                seq_token_i = np.where(struct.sequence.entity_id == eid)[0]
                # Remove bos/eos
                seq_token_i = seq_token_i[1:-1]

                if "residue_map" not in apo_info:
                    # If residue map is not provided, we assume the entire
                    # sequence can be aligned.
                    if len(bb_tok) != len(seq_token_i):
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot be aligned "
                            f"with sequence tokens (len={len(seq_token_i)}) for "
                            f"entity_id {eid} without residue mapping."
                        )
                        continue
                    # Populate the structure tokens for the aligned residues
                    bb_struct_token_id[seq_token_i] = bb_tok
                    fa_struct_token_id[seq_token_i] = fa_tok
                else:
                    residue_map = apo_info["residue_map"]
                    res_st, res_end, apo_st, apo_end = parse_residue_map(residue_map)
                    if res_st == -1:
                        self.logger.warning(
                            f"Invalid residue map {residue_map} for entity_id {eid}."
                        )
                        continue
                    if toklen < (apo_end - apo_st) or (len(seq_token_i) < res_end):
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot cover the "
                            f"residue mapping for entity_id {eid}: {residue_map}."
                        )
                        continue
                    seq_token_i_mapped = seq_token_i[res_st:res_end]
                    # Populate the structure tokens for the mapped residues
                    bb_struct_token_id[seq_token_i_mapped] = bb_tok[apo_st:apo_end]
                    fa_struct_token_id[seq_token_i_mapped] = fa_tok[apo_st:apo_end]


class TrainingDataset(SafeLoadingDataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        ccd: CCD,
        safe_load: bool,
        max_chains: int,
        max_tokens: int,
        max_sequence_tokens: int,
    ) -> None:
        """
        Parameters
        ----------
        config : TrainingDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample.
        max_sequence_tokens : int
            Maximum number of sequence tokens per sample,
            limiting the entire input size of PLM module.
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
            return_symmetry=False,
            return_structure=False,
            safe_load=safe_load,
            train=True,
        )
        self.config: TrainingDatasetConfig = config
        if self.seed is not None:
            # Warn about fixed seed affecting randomness
            self.logger.warning(
                "Seed is set for TrainingDataset, which may affect randomness."
            )

        # For pre-cropping (RefStructure)
        self.max_chains: int = max_chains
        # For main cropping (TokenizedStructure)
        self.max_tokens: int = max_tokens
        self.max_sequence_tokens: int = max_sequence_tokens

        assert max_sequence_tokens >= max_tokens + (max_chains * 2), (
            "max_sequence_tokens should be greater than max_tokens to accommodate "
            "additional sequence tokens for PLM input."
        )  # +2 tokens per chain for [CLS] and [SEP]

        assert config.cropper is not None, "Cropper config must be provided."
        self.cropper: BaseCropper = Registry.instantiate(config.cropper)

        if config.sampler is None:
            # Uniform sampling (complex-level)
            self.sampler: BaseSampler = BaseSampler()
        else:
            # AF3-style sampling (chain/interface-based)
            self.sampler: BaseSampler = Registry.instantiate(config.sampler)

        samples, weights = self.sampler.get_samples(self.metadatas)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray = weights

        # Sequence masking for training
        # TODO: do we have to configurize this?
        self.seq_masking = sequence_masking.SequenceMasking(
            mask_prob=0.9, mask_ratio=0.15
        )

        self.setup()

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        # Check if perturbation is enabled for training set, and validate files.
        if cfg.apo_init.protein_perturbation is None:
            self.logger.warning("Protein perturbation is disabled for training set.")
        elif cfg.apo_init.protein_perturbation.rieprody is not None:
            rieprody_lmdb_path = self.data_root / "rieprody_metric.lmdb"
            if not rieprody_lmdb_path.exists():
                raise FileNotFoundError(
                    f"RieProDy LMDB path {rieprody_lmdb_path} not found "
                    f"while rieprody is enabled."
                )
            # If rieprody perturbation is enabled, we need to provide the LMDB path
            cfg.apo_init.protein_perturbation.rieprody.metric_lmdb_path = (
                rieprody_lmdb_path
            )
        if cfg.apo_init.ligand_perturbation is None:
            self.logger.warning("Ligand perturbation is disabled for training set.")

    @override
    def __len__(self) -> int:
        return len(self.samples)

    def tokenize(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        # Tokenize the structure
        tok_struct = super().tokenize(ref_struct, rng)
        # Then apply sequence masking for training
        self.seq_masking(tok_struct, rng)
        return tok_struct

    @override
    def extract_substructure(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
        **kwargs,
    ) -> RefStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_chains < ref_struct.num_chains:
            # Get sub-complex with limited number of chains
            ref_struct = pre_crop.extract_substructure(
                ref_struct,
                max_chains=self.max_chains,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return ref_struct

    @override
    def crop_structure(
        self,
        struct: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_tokens < struct.num_tokens:
            # Crop the tokenized structure
            struct = self.cropper.crop(
                struct,
                metadata,
                max_tokens=self.max_tokens,
                max_sequence_tokens=self.max_sequence_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return struct

    @override
    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        max_chains = self.max_chains
        max_tokens = self.max_tokens
        max_sequence_tokens = self.max_sequence_tokens
        max_atoms = max_tokens * 24  # max 24 atoms per token
        max_bonds = max_tokens * 10  # max 10 bonds per token
        return f_input.pad(
            max_tokens=max_tokens,
            max_chains=max_chains,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
            max_sequence_tokens=max_sequence_tokens,
        )

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
            metadata = Metadata.from_dict(sample.metadata)
            try:
                return self.get_item(metadata, asym_ids=sample.asym_id)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                sample_id = sample.metadata["id"]
                self.logger.error(
                    f"Error loading index {sample_id}({index}): {e}. Retrying..."
                )
                if not self.safe_load:
                    raise e
                index = np.random.randint(0, len(self))
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
        safe_load: bool = True,
        max_chains: int = 20,
        max_tokens: int = 384,
        max_sequence_tokens: int = 768,
    ) -> None:
        """
        Parameters
        ----------
        configs : list[TrainingDatasetConfig]
            List of dataset configurations.
        ccd: CCD
            CCD database
        safe_load : bool
            Whether to retry loading on failure.
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample.
        max_sequence_tokens : int
            Maximum number of sequence tokens per sample,
            limiting the entire input size of PLM module.

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
                safe_load,
                max_chains,
                max_tokens,
                max_sequence_tokens,
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


class ValidationDataset(SafeLoadingDataset):
    """Validation dataset without sampling and cropping."""

    def __init__(
        self,
        config: ValidationDatasetConfig,
        ccd: CCD,
        safe_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        config : ValidationDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        """
        super().__init__(
            config,
            ccd,
            return_symmetry=True,
            return_structure=True,
            safe_load=safe_load,
            train=False,
        )
        self.config: ValidationDatasetConfig = config

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        if cfg.is_protein_monomer_distillation:
            raise ValueError(
                "Protein monomer distillation dataset should be training dataset"
            )
        # Check if perturbation is enabled for validation set, which is not expected.
        if cfg.apo_init.protein_perturbation is not None:
            self.logger.warning("Protein perturbation is enabled for validation set.")
        if cfg.apo_init.ligand_perturbation is not None:
            self.logger.warning("Ligand perturbation is enabled for validation set.")

    def setup(self) -> None:
        """Additional setup for subclasses."""
        self.metadatas.sort(key=lambda m: Metadata.from_dict(m).num_tokens)
