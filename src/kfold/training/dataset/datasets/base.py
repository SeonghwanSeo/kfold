"""
Dataset structures:

(Shared across datasets)
ccd-train.pkl

(For each dataset)
rcsb-train/
    manifest.json
    structure.lmdb
    apo_lmdb/{chain_type}/{source}.lmdb
    apo_tok_lmdb/protein/{source}.lmdb
    prior_lmdb/{chain_type}.lmdb
    apo_lookup.json     # mapping from each chain to apo structure(s).
af2-long/ ...           # simple dataset with AF2 structures.
    manifest.json
    structure.lmdb/
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
import logging
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch

import kfold.constants as C
from kfold.data.pipelines import (
    apo_initialization,
    featurization,
    prior_sampling,
    tokenization,
)
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.utils.apo_io import (
    unpack_apo_record,
    unpack_prior_stack_record,
)
from kfold.training.utils.permutation_alignment.symmetry import get_symmetries
from kfold.utils.misc import hash_seq

# Type alias
StructInfo = dict


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
    apo_init : ApoInitializerConfig | None
        Configuration for apo structure initialization.
    prob_use_complex_apo : float
        RCSB-specific multimer apo selection probability. Ignored by base datasets.
    prob_use_complex_prior : float
        RCSB-specific multimer prior selection probability. Ignored by base datasets.
    """

    name: str
    data_path: str | Path | None = None
    manifest_path: str | Path | None = None
    seed: int | None = None
    apo_init: apo_initialization.ApoInitializerConfig | None
    prior_sampler: prior_sampling.PriorSamplerConfig | None
    prob_use_complex_apo: float = 0.0
    prob_use_complex_prior: float = 0.0


def next_multiple(n: int, divisor: int) -> int:
    """Return the next integer greater than or equal to n that is divisible by divisor."""
    return ((n + divisor - 1) // divisor) * divisor


class BaseLMDBDataset(torch.utils.data.Dataset):
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
        assert config.data_path is not None, "DatasetConfig.data_path must be specified."
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
        self.apo_initializer: apo_initialization.ApoInitializer | None = None
        if config.apo_init is not None:
            self.apo_initializer = apo_initialization.ApoInitializer(config.apo_init)
        self.tokenizer: tokenization.Tokenizer = tokenizer
        self.featurizer: featurization.InputFeaturizer = featurizer
        self.prior_sampler: prior_sampling.PriorSampler | None = prior_sampler
        self.num_priors: int = 4 if train else 5  # default number of prior samples

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
        if manifest_path.suffix != ".msgpack":
            raise ValueError(f"Manifest must be msgpack: {manifest_path}")
        with open(manifest_path, "rb") as f:
            return msgpack.unpack(f)

    def load_lookup_table(self) -> dict:
        lookup_path = self.data_root / "apo_lookup.msgpack"
        if not lookup_path.exists():
            # NOTE: For protein monomer distillation datasets,
            # we can directly feed apo structures from labeled monomer structures.
            raise FileNotFoundError(f"Apo lookup file {lookup_path} not found.")
        with open(lookup_path, "rb") as f:
            lookup_table: dict = msgpack.unpack(f, raw=False)

        # RCSB preprocessing writes entity IDs as msgpack/json-compatible strings.
        # Runtime code indexes them by integer entity_id from RefStructure chains.
        for entry_id, entry_lookup in lookup_table.items():
            lookup_table[entry_id] = {
                int(eid): list(infos) for eid, infos in entry_lookup.items()
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

    def _get_source_lmdb_env(
        self,
        cache_name: str,
        root_name: str,
        chain_type: str,
        source: str,
    ) -> lmdb.Environment:
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)
        cache_key = (chain_type, source)
        if cache_key not in cache:
            # Workers open LMDB handles lazily after fork; do not share handles
            # from the parent dataset construction path.
            cache[cache_key] = _open_lmdb(
                self.data_root / root_name / chain_type / f"{source}.lmdb"
            )
        return cache[cache_key]

    def _get_apo_source_lmdb_env(self, chain_type: str, source: str) -> lmdb.Environment:
        return self._get_source_lmdb_env(
            "_apo_source_lmdb_envs", "apo_lmdb", chain_type, source
        )

    def _get_apo_tok_source_lmdb_env(
        self, chain_type: str, source: str
    ) -> lmdb.Environment:
        return self._get_source_lmdb_env(
            "_apo_tok_source_lmdb_envs", "apo_tok_lmdb", chain_type, source
        )

    def _get_prior_stack_lmdb_env(self, chain_type: str) -> lmdb.Environment:
        cache = getattr(self, "_prior_stack_lmdb_envs", None)
        if cache is None:
            cache = {}
            self._prior_stack_lmdb_envs = cache
        if chain_type not in cache:
            cache[chain_type] = _open_lmdb(
                self.data_root / "prior_lmdb" / f"{chain_type}.lmdb"
            )
        return cache[chain_type]

    def __del__(self):
        if hasattr(self, "_prior_stack_lmdb_envs"):
            for env in self._prior_stack_lmdb_envs.values():
                env.close()
        if hasattr(self, "_apo_tok_source_lmdb_envs"):
            for env in self._apo_tok_source_lmdb_envs.values():
                env.close()
        if hasattr(self, "_apo_source_lmdb_envs"):
            for env in self._apo_source_lmdb_envs.values():
                env.close()
        if hasattr(self, "_lmdb_env"):
            self._lmdb_env.close()

    # === Core dataset methods === #
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

        # Get apo lookup for the structure
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Fetch apo structure
        apo_dict = self.fetch_apo_structures(ref_struct, apo_lookup, rng)

        # Fetch prior coordinates
        prior_coords = self.sample_prior_coords(ref_struct, rng)

        # Tokenization
        tokenized = self.tokenize(ref_struct, apo_dict, prior_coords, rng)

        # Populate structure tokens for apo structure (in-place)
        # NOTE: For inference, this will be done on-the-fly.
        self.populate_structure_tokens(tokenized, apo_lookup)

        # Featurization
        f_input = self.featurize(tokenized)

        # Pad the features.
        f_input = self.pad_input(f_input)

        struct_info = {
            "id": metadata_id,
            "structure": ref_struct,
            "symmetry": get_symmetries(ref_struct, self.ccd),
            "train_confidence_head": False,
        }

        return f_input, struct_info

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

    def fetch_apo_structures(
        self,
        ref_struct: RefStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Return the apo coordinates for the given reference structure.
        Key: asym_id, Value: apo coordinates of shape [Natoms, 3]
        """
        if not apo_lookup:
            return {}
        assert self.apo_initializer is not None, (
            f"Dataset '{self.name}' has apo lookup records but apo_init is null."
        )
        return self.apo_initializer(ref_struct, apo_lookup, rng)

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior coordinates for the given structure."""
        if self.num_priors <= 0 or self.prior_sampler is None:
            return np.empty((0, ref_struct.num_atoms, 3), dtype=np.float32)

        prior_coords = self.get_prior_coords(ref_struct, rng)
        return self.prior_sampler.sample(ref_struct, prior_coords, self.num_priors, rng)

    def tokenize(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        prior_coords: np.ndarray,
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        return self.tokenizer(
            ref_struct,
            rng,
            apo_coords=apo_dict,
            prior_coords=prior_coords,
        )

    def featurize(self, tokenized: TokenizedStructure) -> FoldingInput:
        """Featurize the given tokenized structure."""
        return self.featurizer(tokenized)

    # === Optional to-override in subclasses === #
    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        """Pad the folding input to multiple of 64 for LocalAtomAttention."""
        # Pad inputs for CUDA efficiency.
        num_tokens = next_multiple(f_input.num_tokens, 32)
        num_sequence_tokens = next_multiple(f_input.num_sequence_tokens, 64)
        num_atoms = next_multiple(f_input.num_atoms, 64)
        return f_input.pad(
            max_tokens=num_tokens,
            max_atoms=num_atoms,
            max_sequence_tokens=num_sequence_tokens,
        )

    # === Helper methods for apo structure handling === #
    @staticmethod
    def _chain_type_name(chain) -> str:
        if chain.ctype.is_protein:
            return "protein"
        if chain.ctype.is_rna:
            return "rna"
        return "dna"

    def _load_apo_info_from_lmdb(
        self,
        apo_info: dict,
        *,
        context: str,
    ) -> dict:
        """Attach `seq` and `coords` from source-specific apo LMDB to a lookup record."""
        loaded = apo_info.copy()
        source = loaded["source"]
        chain_type = loaded["chain_type"]
        lmdb_key = loaded["name"]

        env = self._get_apo_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            value_bytes = txn.get(lmdb_key.encode("utf-8"))
        if value_bytes is None:
            raise KeyError(f"Apo '{source}:{lmdb_key}' not found for {context}")

        loaded["key"] = f"{source}:{lmdb_key}"
        loaded["lmdb_key"] = lmdb_key
        loaded.update(unpack_apo_record(value_bytes))
        return loaded

    def _load_prior_stack_info_from_lmdb(
        self,
        entry_id: str,
        entity_id: int,
        chain_type: str,
    ) -> np.ndarray | None:
        """Load an entity-level stacked prior LMDB record."""
        entity_key = f"{entry_id}_{entity_id}"
        prior_lmdb_path = self.data_root / "prior_lmdb" / f"{chain_type}.lmdb"
        if not prior_lmdb_path.exists():
            return None

        env = self._get_prior_stack_lmdb_env(chain_type)
        with env.begin(write=False) as txn:
            value_bytes = txn.get(entity_key.encode("utf-8"))
        if value_bytes is None:
            return None

        record = unpack_prior_stack_record(value_bytes)
        coords = record["coords"]
        if coords.ndim != 4:
            raise ValueError(
                f"Prior stack {entity_key} has shape {coords.shape}; expected "
                "(N, L, A, 3)."
            )
        return coords

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        entry_id: str = ref_struct.id
        chain_lookup: dict[int, list[dict]] = self.lookup_table[entry_id]

        # Match apo structure for each protein entries.
        apo_lookup: dict[int, dict] = {}  # asym_id -> apo_info dict
        # Symmetric chains share an entity-level apo choice.
        selected_by_entity: dict[int, dict] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}

        for c in ref_struct.chains:
            if c.asym_id in metadata_by_asym_id:
                metadata_by_asym_id[c.asym_id].apo_uid = c.asym_id
            if c.is_ligand:
                # For ligand, we use ETKDG conformers as apo.
                continue

            eid: int = c.entity_id
            ek: str = f"{entry_id}:{eid}"  # For logging purpose
            ctype_str = str(c.ctype)

            if eid not in chain_lookup:
                self.logger.warning(
                    f"No apo info found for {ctype_str} entity '{ek}' in lookup."
                )
                continue

            if eid not in selected_by_entity:
                entity_apo_infos: list[dict] = chain_lookup[eid]
                num_apos = len(entity_apo_infos)
                # Select apo structure (randomly if multiple)
                assert num_apos > 0, (
                    f"Empty apo info found for {ctype_str} entity '{ek}' in lookup."
                )
                apo_info = entity_apo_infos[rng.integers(0, num_apos)]
                loaded = self._load_apo_info_from_lmdb(apo_info, context=f"entity {ek}")
                selected_by_entity[eid] = loaded

            apo_info = selected_by_entity[eid].copy()
            # ApoInitializer consumes asym_id-keyed records after sub-complex
            # extraction; keep the sampled entity-level payload but bind it to
            # this physical chain.
            apo_info["asym_id"] = c.asym_id
            apo_info["apo_uid"] = c.asym_id
            apo_info["is_multimer_apo"] = False
            apo_lookup[c.asym_id] = apo_info

        return apo_lookup

    def get_prior_coords(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Load per-chain priors as stacked residue-order coordinates."""
        entry_id: str = ref_struct.id
        prior_coords: dict[int, np.ndarray] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}

        for c in ref_struct.chains:
            if c.asym_id in metadata_by_asym_id:
                metadata_by_asym_id[c.asym_id].prior_uid = c.asym_id

        for c in ref_struct.chains:
            if not (c.ctype.is_protein or c.ctype.is_nucleic_acid):
                continue

            if c.asym_id in prior_coords:
                continue

            eid: int = c.entity_id
            chain_type = self._chain_type_name(c)
            loaded = self._load_prior_stack_info_from_lmdb(entry_id, eid, chain_type)
            if loaded is None:
                continue

            prior_coords[c.asym_id] = loaded

        return prior_coords

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        """Populate the structure tokens for the given tokenized structure.

        Parameters
        ----------
        tokenized : TokenizedStructure
            The tokenized structure to populate tokens for in-place.
        apo_lookup : dict[int, dict]
            Mapping from asym_id to apo structure dictionary metadata.
        """
        for c_i in range(tokenized.num_chains):
            if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                continue  # only populate structure tokens for protein chains

            asym_id = int(tokenized.chain.asym_id[c_i])
            ek = f"{tokenized.id}:{asym_id}"  # For logging purpose

            apo_info = apo_lookup.get(asym_id)
            if apo_info is None:
                self.logger.warning(
                    f"No apo info for protein chain `{ek}` in apo lookup. "
                    f"Skipping this entry"
                )
                continue

            source = apo_info["source"]
            key = apo_info["name"]

            env = self._get_apo_tok_source_lmdb_env("protein", source)
            with env.begin(write=False) as txn:
                v = txn.get(key.encode("utf-8"))
            if v is None:
                self.logger.warning(
                    f"Apo structure tokens {source}:{key} not found in LMDB for "
                    f"chain `{ek}`. Skipping this entry"
                )
                continue
            # Load pre-computed structure tokens for apo structure from LMDB
            apo_tok = np.frombuffer(v, dtype=np.int16).reshape(2, -1)
            self._insert_structure_tokens(
                tokenized, c_i, apo_info, apo_tok, source=source, key=key, ek=ek
            )

    def _insert_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        c_i: int,
        apo_info: dict,
        apo_tok: np.ndarray,
        *,
        source: str,
        key: str,
        ek: str,
    ) -> None:
        """Insert one chain's apo structure tokens into a tokenized structure."""
        bb_struct_token_id = tokenized.sequence.bb_struct_token_id
        fa_struct_token_id = tokenized.sequence.fa_struct_token_id
        seq_lens = tokenized.chain.num_residues + 2
        seq_starts = np.cumsum(seq_lens) - seq_lens
        bb_tok, fa_tok = apo_tok
        toklen = len(bb_tok)

        seq_start = int(seq_starts[c_i])
        seq_end = seq_start + int(seq_lens[c_i])
        seq_token_len = int(tokenized.chain.num_residues[c_i])

        if "residue_map" not in apo_info:
            if len(bb_tok) != seq_token_len:
                self.logger.warning(
                    f"Apo tokens ({source}:{key}, len={toklen}) cannot be "
                    f"aligned with sequence tokens (len={seq_token_len}) for "
                    f"chain {c_i} (entity `{ek}`) without residue map. Skipping."
                )
                return
            bb_struct_token_id[seq_start + 1 : seq_end - 1] = bb_tok
            fa_struct_token_id[seq_start + 1 : seq_end - 1] = fa_tok
            return

        residue_map = apo_info["residue_map"]
        res_st, res_end, apo_st, apo_end = parse_residue_map(residue_map)
        if res_st == -1:
            self.logger.warning(
                f"Invalid residue map {residue_map} for chain {c_i} "
                f"(entity {ek}). Skipping."
            )
            return
        if toklen < (apo_end - apo_st) or (seq_token_len < res_end):
            self.logger.warning(
                f"Apo tokens ({source}:{key}, len={toklen}) cannot cover "
                f"the residue mapping for chain {c_i} (entity {ek}): "
                f"{residue_map}. Skipping."
            )
            return
        _st, _end = seq_start + 1 + res_st, seq_start + 1 + res_end
        bb_struct_token_id[_st:_end] = bb_tok[apo_st:apo_end]
        fa_struct_token_id[_st:_end] = fa_tok[apo_st:apo_end]
