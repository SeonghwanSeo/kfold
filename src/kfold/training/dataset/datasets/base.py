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
    apo_tok.lmdb     # pre-computed structure tokens for apo structures.
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
import json
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
from kfold.data.utils.io.apo import unpack_apo_complex_record, unpack_apo_record
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
    apo_init : ApoInitializerConfig
        Configuration for apo structure initialization.
    prob_use_complex_apo : float
        Probability of replacing chain apo records with grouped complex apo records
        when the lookup entry provides complex groups.
    """

    name: str
    data_path: str | Path | None = None
    manifest_path: str | Path | None = None
    seed: int | None = None
    apo_init: apo_initialization.ApoInitializerConfig
    prior_sampler: prior_sampling.PriorSamplerConfig | None
    prob_use_complex_apo: float = 0.0


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
        self.config: DatasetConfig = config
        self.name: str = config.name
        self.data_root: Path = Path(config.data_path)
        self.seed: int | None = config.seed
        self.prob_use_complex_apo: float = config.prob_use_complex_apo
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
            lookup_table: dict = msgpack.unpack(f, raw=False, strict_map_key=False)

        # Normalize legacy and extended lookup formats to one internal shape.
        for entry_id, entry_lookup in lookup_table.items():
            lookup_table[entry_id] = self.normalize_apo_lookup_entry(entry_lookup)
        return lookup_table

    @staticmethod
    def normalize_apo_lookup_entry(entry_lookup: dict) -> dict:
        """Normalize one apo lookup entry to the extended internal format."""
        if "chains" in entry_lookup or "complex_groups" in entry_lookup:
            chains_raw = entry_lookup.get("chains", {})
            complex_groups_raw = entry_lookup.get("complex_groups", [])
        else:
            # Legacy format: entity_id -> [apo_info]
            chains_raw = entry_lookup
            complex_groups_raw = []

        chains: dict[int, list[dict]] = {
            int(eid): list(infos) for eid, infos in chains_raw.items()
        }

        complex_groups: list[dict] = []
        for group in complex_groups_raw:
            members_raw = group.get("members", {})
            members = {int(aid): info for aid, info in members_raw.items()}
            if "asym_ids" in group:
                asym_ids = [int(aid) for aid in group["asym_ids"]]
            else:
                asym_ids = list(members.keys())
            normalized = {
                "kind": group.get("kind", "complex"),
                "chain_type": group.get("chain_type", "protein"),
                "apo_uid": int(group["apo_uid"]),
                "asym_ids": asym_ids,
                "group_id": group.get("group_id", ""),
            }
            if "label_asym_ids" in group:
                normalized["label_asym_ids"] = [
                    str(aid) for aid in group["label_asym_ids"]
                ]
            if members:
                normalized["members"] = members
            if "candidates" in group:
                normalized["candidates"] = list(group["candidates"])
            complex_groups.append(normalized)

        return {"chains": chains, "complex_groups": complex_groups}

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
    def apo_complex_lmdb_env(self) -> lmdb.Environment:
        """Get the LMDB environment for multimer apo structures."""
        if not hasattr(self, "_apo_complex_lmdb_env"):
            self._apo_complex_lmdb_env = _open_lmdb(self.data_root / "apo_complex.lmdb")
        return self._apo_complex_lmdb_env

    @property
    def apo_tok_lmdb_env(self) -> lmdb.Environment:
        """Get the LMDB environment for structure tokens of apo structures."""
        if not hasattr(self, "_apo_tok_lmdb_env"):
            self._apo_tok_lmdb_env = _open_lmdb(self.data_root / "apo_tok.lmdb")
        return self._apo_tok_lmdb_env

    def __del__(self):
        if hasattr(self, "_apo_complex_lmdb_env"):
            self._apo_complex_lmdb_env.close()
        if hasattr(self, "_apo_lmdb_env"):
            self._apo_lmdb_env.close()
        if hasattr(self, "_apo_tok_lmdb_env"):
            self._apo_tok_lmdb_env.close()
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

        # Tokenization
        tokenized = self.tokenize(ref_struct, apo_dict, rng)

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
        return self.apo_initializer(ref_struct, apo_lookup, rng)

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample the prior coordinates for the given structure and apo coordinates."""
        if self.num_priors <= 0 or self.prior_sampler is None:
            return np.empty((0, ref_struct.num_atoms, 3), dtype=np.float32)
        else:
            return self.prior_sampler(ref_struct, apo_dict, self.num_priors, rng)

    def tokenize(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        # Sample prior coordinates for diffusion bridge model
        prior_coords = self.sample_prior_coords(ref_struct, apo_dict, rng)
        # Tokenization
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
    def _get_apo_key(apo_info: dict) -> str:
        if "key" in apo_info:
            return apo_info["key"]
        return f"{apo_info['source']}:{apo_info['name']}"

    @staticmethod
    def _decode_apo_sequence(seq: np.ndarray) -> str:
        if seq.ndim == 0:
            return str(seq.astype(str).item())
        return "".join(seq.astype(str).tolist())

    def _load_apo_info_from_lmdb(
        self,
        apo_info: dict,
        *,
        context: str,
        strict: bool,
    ) -> dict | None:
        """Attach `seq` and `coords` from apo.lmdb to an apo lookup record."""
        loaded = apo_info.copy()
        apo_key = self._get_apo_key(loaded)
        loaded["key"] = apo_key

        with self.apo_lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(apo_key.encode("utf-8"))
        if value_bytes is None:
            msg = f"Apo '{apo_key}' not found in LMDB for {context}"
            if strict:
                raise KeyError(msg)
            self.logger.warning(msg)
            return None

        loaded.update(unpack_apo_record(value_bytes))
        return loaded

    def _load_apo_complex_info_from_lmdb(
        self,
        apo_info: dict,
        *,
        context: str,
        strict: bool,
    ) -> dict | None:
        """Attach multimer chain payloads from apo_complex.lmdb."""
        loaded = apo_info.copy()
        apo_key = self._get_apo_key(loaded)
        loaded["key"] = apo_key

        with self.apo_complex_lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(apo_key.encode("utf-8"))
        if value_bytes is None:
            msg = f"Apo complex '{apo_key}' not found in LMDB for {context}"
            if strict:
                raise KeyError(msg)
            self.logger.warning(msg)
            return None

        loaded["chains"] = unpack_apo_complex_record(value_bytes)
        return loaded

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        entry_id: str = ref_struct.id
        entry_lookup: dict = self.lookup_table[entry_id]
        chain_lookup: dict[int, list[dict]] = entry_lookup["chains"]
        complex_groups: list[dict] = entry_lookup["complex_groups"]

        # Match apo structure for each protein entries.
        apo_lookup: dict[int, dict] = {}  # asym_id -> apo_info dict
        selected_by_entity: dict[int, dict] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        label_by_asym_id = {
            c.asym_id: c.label_asym_id or str(c.asym_id)
            for c in ref_struct.metadata.chains
        }
        protein_asym_ids = {c.asym_id for c in ref_struct.chains if c.ctype.is_protein}

        for c in ref_struct.chains:
            if c.asym_id in metadata_by_asym_id:
                metadata_by_asym_id[c.asym_id].apo_uid = c.asym_id
            if not (c.ctype.is_protein or c.ctype.is_nucleic_acid):
                # Currently we only provide apo structures for polymer chains.
                # For ligand, we use ETKDG conformers as apo.
                continue

            eid: int = c.entity_id
            ek: str = f"{entry_id}:{eid}"  # For logging purpose

            if eid not in chain_lookup:
                self.logger.warning(f"No apo info found for entity '{ek}' in lookup.")
                continue

            if eid not in selected_by_entity:
                entity_apo_infos: list[dict] = chain_lookup[eid]
                num_apos = len(entity_apo_infos)
                # Select apo structure (randomly if multiple)
                if num_apos == 0:
                    self.logger.warning(
                        f"Empty apo info found for entity '{ek}' in lookup."
                    )
                    continue
                apo_info = entity_apo_infos[rng.integers(0, num_apos)]
                loaded = self._load_apo_info_from_lmdb(
                    apo_info, context=f"entity {ek}", strict=False
                )
                if loaded is None:
                    continue
                selected_by_entity[eid] = loaded

            apo_info = selected_by_entity[eid].copy()
            apo_info["asym_id"] = c.asym_id
            apo_info["apo_uid"] = c.asym_id
            apo_info["skip_perturbation"] = c.ctype.is_nucleic_acid
            apo_info["use_struct_token"] = c.ctype.is_protein
            apo_lookup[c.asym_id] = apo_info

        if complex_groups and rng.random() < self.prob_use_complex_apo:
            for group in complex_groups:
                apo_uid = int(group["apo_uid"])
                group_asym_ids = [int(aid) for aid in group["asym_ids"]]
                active_asym_ids = [
                    asym_id for asym_id in group_asym_ids if asym_id in protein_asym_ids
                ]
                if not active_asym_ids:
                    continue

                if group.get("candidates"):
                    candidates = group["candidates"]
                    candidate = candidates[rng.integers(0, len(candidates))]
                    context = f"complex group {entry_id}:{apo_uid}"
                    complex_info = self._load_apo_complex_info_from_lmdb(
                        candidate, context=context, strict=True
                    )
                    assert complex_info is not None
                    complex_chains: dict[str, dict] = complex_info["chains"]
                    label_asym_ids = group.get("label_asym_ids", [])
                    label_by_group_asym_id = {
                        int(asym_id): str(label)
                        for asym_id, label in zip(
                            group_asym_ids, label_asym_ids, strict=False
                        )
                    }
                    for asym_id in active_asym_ids:
                        label_asym_id = label_by_group_asym_id.get(
                            asym_id, label_by_asym_id.get(asym_id, str(asym_id))
                        )
                        if label_asym_id not in complex_chains:
                            raise KeyError(
                                f"Apo complex {complex_info['key']} for entry "
                                f"{entry_id} does not contain chain "
                                f"{label_asym_id} for asym_id {asym_id}."
                            )
                        loaded = candidate.copy()
                        loaded.update(complex_chains[label_asym_id])
                        loaded["key"] = complex_info["key"]
                        loaded["complex_key"] = complex_info["key"]
                        loaded["complex_chain_id"] = label_asym_id
                        loaded["asym_id"] = asym_id
                        loaded["apo_uid"] = apo_uid
                        loaded["skip_perturbation"] = True
                        loaded["use_struct_token"] = False
                        apo_lookup[asym_id] = loaded
                        if asym_id in metadata_by_asym_id:
                            metadata_by_asym_id[asym_id].apo_uid = apo_uid
                    continue

                members: dict[int, dict] = group.get("members", {})
                for asym_id in active_asym_ids:
                    asym_id = int(asym_id)
                    if asym_id not in members:
                        raise KeyError(
                            f"Complex apo group for entry {entry_id} is missing "
                            f"member asym_id {asym_id}."
                        )

                    context = f"complex group {entry_id}:{apo_uid}/{asym_id}"
                    loaded = self._load_apo_info_from_lmdb(
                        members[asym_id], context=context, strict=True
                    )
                    assert loaded is not None
                    loaded["asym_id"] = asym_id
                    loaded["apo_uid"] = apo_uid
                    loaded["skip_perturbation"] = True
                    loaded["use_struct_token"] = False
                    apo_lookup[asym_id] = loaded
                    if asym_id in metadata_by_asym_id:
                        metadata_by_asym_id[asym_id].apo_uid = apo_uid

        return apo_lookup

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
        rng : np.random.Generator
            Random number generator.
        """
        bb_struct_token_id = tokenized.sequence.bb_struct_token_id
        fa_struct_token_id = tokenized.sequence.fa_struct_token_id

        # Precompute start and end indices of sequence tokens for all chains
        seq_lens = tokenized.chain.num_residues + 2
        seq_starts = np.cumsum(seq_lens) - seq_lens

        with self.apo_tok_lmdb_env.begin(write=False) as txn:
            for c_i in range(tokenized.num_chains):
                if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                    continue  # only populate structure tokens for protein chains

                asym_id = int(tokenized.chain.asym_id[c_i])
                eid = tokenized.chain.entity_id[c_i]
                ek = f"{tokenized.id}:{asym_id}"  # For logging purpose

                apo_info = apo_lookup.get(asym_id, apo_lookup.get(int(eid)))
                if apo_info is None:
                    self.logger.warning(
                        f"No apo info for chain `{ek}` in apo lookup. Skipping this entry"
                    )
                    continue
                if not apo_info.get("use_struct_token", True):
                    continue
                key = apo_info["key"]
                v = txn.get(key.encode("utf-8"))
                if v is None:
                    self.logger.warning(
                        f"Apo structure tokens {key} not found in LMDB for "
                        f"chain `{ek}`. Skipping this entry"
                    )
                    continue
                # Load pre-computed structure tokens for apo structure from LMDB
                apo_tok = np.frombuffer(v, dtype=np.int16).reshape(2, -1)
                bb_tok, fa_tok = apo_tok
                toklen = len(bb_tok)

                # Compute the sequence token slice for this chain c_i safely
                seq_start = int(seq_starts[c_i])
                seq_end = seq_start + int(seq_lens[c_i])
                seq_token_len = int(tokenized.chain.num_residues[c_i])

                if "residue_map" not in apo_info:
                    # If residue map is not provided, we assume the entire
                    # sequence can be aligned.
                    if len(bb_tok) != seq_token_len:
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot be aligned "
                            f"with sequence tokens (len={seq_token_len}) for "
                            f"chain {c_i} (entity `{ek}`) without residue map. Skipping."
                        )
                        continue
                    # Populate the structure tokens for the aligned residues
                    bb_struct_token_id[seq_start + 1 : seq_end - 1] = bb_tok
                    fa_struct_token_id[seq_start + 1 : seq_end - 1] = fa_tok
                else:
                    residue_map = apo_info["residue_map"]
                    res_st, res_end, apo_st, apo_end = parse_residue_map(residue_map)
                    if res_st == -1:
                        self.logger.warning(
                            f"Invalid residue map {residue_map} for chain {c_i} "
                            f"(entity {ek}). Skipping."
                        )
                        continue
                    if toklen < (apo_end - apo_st) or (seq_token_len < res_end):
                        self.logger.warning(
                            f"Apo tokens ({key}, len={toklen}) cannot cover the "
                            f"residue mapping for chain {c_i} (entity {ek}): "
                            f"{residue_map}. Skipping."
                        )
                        continue
                    # Populate the structure tokens for the mapped residues
                    _st, _end = seq_start + 1 + res_st, seq_start + 1 + res_end
                    bb_struct_token_id[_st:_end] = bb_tok[apo_st:apo_end]
                    fa_struct_token_id[_st:_end] = fa_tok[apo_st:apo_end]
