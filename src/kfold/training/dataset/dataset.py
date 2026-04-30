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
from typing_extensions import override

import kfold.constants as C
from kfold.data.pipelines import (
    apo_initialization,
    featurization,
    prior_sampling,
    sequence_masking,
    structure_cleaning,
    tokenization,
)
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.utils.permutation_alignment.symmetry import get_symmetries
from kfold.utils.misc import hash_seq
from kfold.utils.registry import Registry

from .cropper import BaseCropper
from .sampler import BaseSampler, Sample
from .utils import constraint_sampling, pre_crop


# === Helper functions === #
def _open_lmdb(lmdb_path: str | Path) -> lmdb.Environment:
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found.")
    return lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )


def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example: "1:100->5:104" -> (0, 100, 4, 104)
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
    apo_init : ApoInitializerConfig
        Configuration for apo structure initialization.
    """

    name: str
    data_path: str | Path
    manifest_path: str | Path | None = None
    seed: int | None = None
    apo_init: apo_initialization.ApoInitializerConfig
    prior_sampler: prior_sampling.PriorSamplerConfig | None


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
StructInfo = dict


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class SafeLoadingDataset(torch.utils.data.Dataset):
    """A dataset that safely retries loading data on failure."""

    def __init__(
        self,
        config: DatasetConfig,
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
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
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
        safe_load : bool
            Whether to retry loading on failure.
        """
        # === Initialize parameters === #
        self.config: DatasetConfig = config
        self.name: str = config.name
        self.data_root: Path = Path(config.data_path)
        self.seed: int | None = config.seed
        self.safe_load: bool = safe_load
        self.train: bool = train

        if train:
            self.logger = logging.getLogger(f"[Training Dataset:{self.name}]")
        else:
            self.logger = logging.getLogger(f"[Validation Dataset:{self.name}]")

        # Sanity check on dataset files and configurations
        self.sanity_check()

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
            config.apo_init, self.ccd
        )
        self.tokenizer: tokenization.Tokenizer = tokenizer
        self.featurizer: featurization.InputFeaturizer = featurizer
        self.prior_sampler: prior_sampling.PriorSampler | None = prior_sampler
        self.num_priors: int = 4 if train else 5  # default number of prior samples

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
        lookup_path = self.data_root / "apo_lookup.msgpack"
        if not lookup_path.exists():
            # NOTE: For protein monomer distillation datasets,
            # we can directly feed apo structures from labeled monomer structures.
            raise FileNotFoundError(f"Apo lookup file {lookup_path} not found.")
        with open(lookup_path, "rb") as f:
            lookup_table: dict = msgpack.unpack(f)
        # Convert entity IDs from string to int for easier handling later
        for entry_id, entry_lookup in lookup_table.items():
            lookup_table[entry_id] = {
                int(eid): infos for eid, infos in entry_lookup.items()
            }
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

    def cleanup_structure(self, ref_struct: RefStructure) -> RefStructure:
        """Clean up the reference structure as needed."""
        # NOTE: Right now, we simply filter out the unrealistic bonds.
        return structure_cleaning.clean_up_ref_structure(ref_struct)

    def load_apo_structure(
        self,
        ref_struct: RefStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Populate the apo structure for the given reference structure."""
        self.apo_initializer(ref_struct, apo_lookup, rng)

    def tokenize(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        return self.tokenizer(ref_struct, rng, num_priors=self.num_priors)

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> None:
        """Populate the prior coordinates for the given reference structure."""
        num_priors = self.num_priors
        if self.prior_sampler is not None:
            prior_coords = self.prior_sampler(ref_struct, num_priors, rng)
            prior_coords = prior_coords.transpose(1, 0, 2)
            mask = tokenized.atom.pad_mask
            tokenized.atom.prior_coords[mask] = prior_coords

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
        tokenized: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> TokenizedStructure:
        """Crop the tokenized structure as needed to fit within the model input size."""
        return tokenized

    def featurize(self, tokenized: TokenizedStructure) -> FoldingInput:
        """Featurize the given tokenized structure."""
        return self.featurizer(tokenized)

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        # Pad num_tokens for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 32)
        # Pad num_seq_tokens for CUDA efficiency.
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 64)
        # Pad num_atoms for local attention.
        num_atoms = next_multiple(f_input.num_atoms, 64)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence_tokens=num_sequence_tokens,
        )

    def __getitem__(self, index: int) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given index, with retry on failure."""
        return self.get_item_safe(index, num_trials=100)

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
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
                trials.append(sample_id)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(self, metadata: Metadata, **kwargs) -> tuple[FoldingInput, StructInfo]:
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

        # Clean up structure
        ref_struct = self.cleanup_structure(ref_struct)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        ref_struct = self.extract_substructure(ref_struct, rng=rng, **kwargs)
        metadata = ref_struct.metadata  # update metadata after extraction

        # Get apo lookup for the structure
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Populate apo structure (in-place)
        self.load_apo_structure(ref_struct, apo_lookup, rng)

        # Tokenization
        tokenized: TokenizedStructure = self.tokenize(ref_struct, rng=rng)

        # Sample prior coordinates for diffusion bridge model (in-place)
        self.sample_prior_coords(ref_struct, tokenized, rng)

        # Populate structure tokens for apo structure (in-place)
        self.populate_structure_tokens(tokenized, apo_lookup, rng)

        # Cropping
        cropped = self.crop_structure(tokenized, metadata, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped)

        # Pad the features.
        f_input = self.pad_input(f_input)

        struct_info = {
            "id": metadata_id,
            "structure": ref_struct,
            "symmetry": get_symmetries(ref_struct, self.ccd),
        }

        return f_input, struct_info

    # === Helper methods for apo structure handling === #
    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        entry_id: str = ref_struct.id
        entry_lookup: dict[int, list[dict[str, str]]] = self.lookup_table[entry_id]

        # Match apo structure for each protein entries.
        apo_lookup: dict[int, dict] = {}  # entity_id -> apo_info dict
        visited_entity_ids: set[int] = set()
        for c in ref_struct.chains:
            if not c.ctype.is_protein:
                # Currently we only provide apo structures for protein chains.
                # For ligand, we use ETKDG conformers as apo.
                continue

            eid: int = c.entity_id
            ek: str = f"{entry_id}:{eid}"  # For logging purpose

            if eid in visited_entity_ids:
                continue  # already populated from another chain with same entity_id
            visited_entity_ids.add(eid)

            if eid not in entry_lookup:
                self.logger.warning(f"No apo info found for entity '{ek}' in lookup.")
                continue

            entity_apo_infos: list[dict[str, str]] = entry_lookup[eid]
            num_apos = len(entity_apo_infos)
            # Select apo structure (randomly if multiple)
            if num_apos == 0:
                self.logger.warning(f"Empty apo info found for entity '{ek}' in lookup.")
                continue
            apo_info = entity_apo_infos[rng.integers(0, num_apos)].copy()

            name = apo_info["name"]
            source = apo_info["source"]
            apo_key = f"{source}:{name}"
            apo_info["key"] = apo_key

            # Load apo coordinates from LMDB
            with self.apo_lmdb_env.begin(write=False) as txn:
                value_bytes = txn.get(apo_key.encode("utf-8"))
                if value_bytes is None:
                    self.logger.warning(
                        f"Apo '{apo_key}' not found in LMDB for entity {ek}"
                    )
                    continue
                with io.BytesIO(value_bytes) as byte_stream:
                    with np.load(byte_stream) as data:
                        apo_info["seq"] = "".join(data["seq"].astype(str).tolist())
                        apo_info["coords"] = data["coords"].copy()

            apo_lookup[eid] = apo_info
        return apo_lookup

    def populate_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> None:
        """Populate the structure tokens for the given tokenized structure."""

        bb_struct_token_id = tokenized.sequence.bb_struct_token_id
        fa_struct_token_id = tokenized.sequence.fa_struct_token_id

        visited_entity_ids: set[int] = set()
        with self.unitok_lmdb_env.begin(write=False) as txn:
            for c_i in range(tokenized.num_chains):
                if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                    continue  # only populate structure tokens for protein chains

                eid = tokenized.chain.entity_id[c_i]
                ek = f"{tokenized.id}:{eid}"  # For logging purpose

                if eid in visited_entity_ids:
                    continue  # already populated from another chain with same entity_id
                visited_entity_ids.add(eid)

                if eid not in apo_lookup:
                    self.logger.warning(
                        f"No apo info for entity `{ek}` in apo lookup. "
                        f"Skipping this entry"
                    )
                    continue
                apo_info = apo_lookup[eid]
                key = apo_info["key"]
                v = txn.get(key.encode("utf-8"))
                if v is None:
                    self.logger.warning(
                        f"Apo structure tokens {key} not found in LMDB for "
                        f"entity `{ek}`. Skipping this entry"
                    )
                    continue
                # Load pre-computed structure tokens for apo structure from LMDB
                apo_unitok = np.frombuffer(v, dtype=np.uint16).reshape(2, -1)
                bb_tok, fa_tok = apo_unitok
                toklen = len(bb_tok)

                # Find the corresponding sequence token indices
                seq_token_i = np.where(tokenized.sequence.entity_id == eid)[0]
                # Remove bos/eos
                seq_token_i = seq_token_i[1:-1]

                if "residue_map" not in apo_info:
                    # If residue map is not provided, we assume the entire
                    # sequence can be aligned.
                    if len(bb_tok) != len(seq_token_i):
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot be aligned "
                            f"with sequence tokens (len={len(seq_token_i)}) for "
                            f"entity `{ek}` without residue map. Skipping this entry."
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
                            f"Invalid residue map {residue_map} for entity {ek}. "
                            f"Skipping this entry."
                        )
                        continue
                    if toklen < (apo_end - apo_st) or (len(seq_token_i) < res_end):
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot cover the "
                            f"residue mapping for entity {ek}: {residue_map}."
                            f" Skipping this entry."
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
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
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
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
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
            tokenizer,
            featurizer,
            prior_sampler,
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
            f"max_sequence_tokens should be greater than max_tokens to accommodate "
            f"additional sequence tokens for PLM input."
            f" (max_sequence_tokens={max_sequence_tokens}, max_tokens={max_tokens}, "
            f"max_chains={max_chains})"
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

        # Constraint sampling for training
        # TODO: configurize the parameters
        self.constraint_sampling = constraint_sampling.ConstraintSampling(
            min_dist=3.0,
            max_dist=22.0,
            prob_constraint=0.05,
            max_constraints=5,
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
        # Sample the constraints
        constraints = self.constraint_sampling(ref_struct, rng)
        # Tokenize the structure
        tok_struct = self.tokenizer(
            ref_struct, rng, num_priors=self.num_priors, constraints=constraints
        )
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
        tokenized: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> TokenizedStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_tokens < tokenized.num_tokens:
            # Crop the tokenized structure
            tokenized = self.cropper.crop(
                tokenized,
                metadata,
                max_tokens=self.max_tokens,
                max_sequence_tokens=self.max_sequence_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return tokenized

    @override
    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        max_chains = self.max_chains
        max_tokens = self.max_tokens
        max_sequence_tokens = self.max_sequence_tokens
        max_atoms = max_tokens * 24  # max 24 atoms per token
        max_bonds = max_tokens * 10  # max 10 bonds per token
        num_constraints = max_tokens  # max 1 constraint per token
        return f_input.pad(
            max_tokens=max_tokens,
            max_chains=max_chains,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
            max_sequence_tokens=max_sequence_tokens,
            max_constraints=num_constraints,
        )

    @override
    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
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
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        max_chains: int,
        max_tokens: int,
        max_sequence_tokens: int,
        safe_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        configs : list[TrainingDatasetConfig]
            List of dataset configurations.
        ccd: CCD
            CCD database
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample.
        max_sequence_tokens : int
            Maximum number of sequence tokens per sample,
            limiting the entire input size of PLM module.
        safe_load : bool
            Whether to retry loading on failure.

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
                tokenizer,
                featurizer,
                prior_sampler,
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

    def __getitem__(self, index: int) -> tuple[FoldingInput, StructInfo]:
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
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
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
            tokenizer,
            featurizer,
            prior_sampler,
            safe_load=safe_load,
            train=False,
        )
        self.config: ValidationDatasetConfig = config

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        # Check if perturbation is enabled for validation set, which is not expected.
        if cfg.apo_init.protein_perturbation is not None:
            self.logger.warning("Protein perturbation is enabled for validation set.")
        if cfg.apo_init.ligand_perturbation is not None:
            self.logger.warning("Ligand perturbation is enabled for validation set.")

    def setup(self) -> None:
        """Additional setup for subclasses."""
        self.metadatas.sort(key=lambda m: Metadata.from_dict(m).num_tokens)
