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
from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.utils import apo_io, apo_perturbation
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
    """

    name: str
    data_path: str | Path | None = None
    manifest_path: str | Path | None = None
    seed: int | None = None
    prob_perturbation: float = 0.0
    apo_perturb: apo_perturbation.ApoPerturbationConfig | None = None


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
        self.tokenizer: tokenization.Tokenizer = tokenizer
        self.featurizer: featurization.InputFeaturizer = featurizer
        self.prior_sampler: prior_sampling.PriorSampler | None = prior_sampler
        self.num_priors: int = 4 if train else 5  # default number of prior samples

        # Data augmentation
        self.apo_perturb: apo_perturbation.ApoPerturbation | None = None
        if config.apo_perturb is not None:
            self.apo_perturb: apo_perturbation.ApoPerturbation = (
                apo_perturbation.ApoPerturbation(config.apo_perturb)
            )

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

    def __del__(self):
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

        # Get apo lookup for the structure (asym_id to apo)
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Fetch apo structure
        apo_dict = self.fetch_apo_structures(ref_struct, apo_lookup, rng)

        # Sample prior coordinates for diffusion bridge model.
        prior_coords = self.sample_prior_coords(ref_struct, apo_dict, rng)

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

    # === Helper methods for apo structure handling === #
    @staticmethod
    def _chain_type_name(chain) -> str:
        if chain.ctype.is_protein:
            return "protein"
        if chain.ctype.is_rna:
            return "rna"
        return "dna"

    def _load_apo_info_from_lmdb(self, apo_info: dict) -> bool:
        """Attach `seq` and `coords` from source-specific apo LMDB to a lookup record."""
        source = apo_info["source"]
        chain_type = apo_info["chain_type"]
        name = apo_info["name"]
        env = self._get_apo_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            v = txn.get(name.encode("utf-8"))
        if v is None:
            apo_key = apo_info["key"]
            entity_key = apo_info["entity_key"]
            self.logger.warning(
                f"Apo structure {apo_key} not found in LMDB for "
                f"entity `{entity_key}`. Skipping this entry"
            )
            return False
        else:
            apo_info.update(apo_io.unpack_apo_record(v))
            return True

    def _load_apo_tok_from_lmdb(self, apo_info: dict) -> bool:
        """Load the pre-computed structure tokens for the apo structure from LMDB."""
        source = apo_info["source"]
        chain_type = apo_info["chain_type"]
        name = apo_info["name"]
        if chain_type != "protein":
            # NOTE: Currently, we only use structure tokens for protein.
            return True

        env = self._get_apo_tok_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            v = txn.get(name.encode("utf-8"))
        if v is None:
            apo_key = apo_info["key"]
            entity_key = apo_info["entity_key"]
            self.logger.warning(
                f"Apo structure token {apo_key} not found in LMDB for "
                f"entity `{entity_key}`. Skipping this entry"
            )
            return False
        else:
            # Load pre-computed structure tokens for apo structure from LMDB
            apo_tok = np.frombuffer(v, dtype=np.int16).reshape(2, -1)
            apo_info["tokens"] = apo_tok
            return True

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        entry_id: str = ref_struct.id
        entry_lookup: dict[int, list[dict]] = self.lookup_table[entry_id]

        # Match apo structure for each protein entries.
        apo_lookup: dict[int, dict] = {}  # asym_id -> apo_info dict
        cache: dict[int, dict] = {}  # entity_id -> apo_info dict
        for c in ref_struct.chains:
            if not c.is_polymer:
                # We do not save ligand etkdg conformers in apo_lookup.
                continue
            if c.entity_id in cache:
                # NOTE: This is essential: for prior sampling, we need to
                # conduct OT permutation alignment between apo and holo structures.
                # If we use different apo structures for the same entity,
                # the alignment will be inconsistent.
                apo_info = cache[c.entity_id].copy()
                apo_info["chain_key"] = f"{entry_id}_{c.asym_id}"
                apo_lookup[c.asym_id] = apo_info
                continue

            eid: int = c.entity_id
            ek: str = f"{entry_id}:{eid}"  # For logging purpose

            if eid not in entry_lookup:
                self.logger.warning(f"No apo info found for entity '{ek}' in lookup.")
                continue

            # Select one apo structure randomly
            entity_apo_infos: list[dict] = entry_lookup[eid]
            num_apos = len(entity_apo_infos)
            assert num_apos > 0, f"No apo info for entity '{ek}' in lookup."
            apo_info = entity_apo_infos[rng.integers(0, num_apos)].copy()

            # Add key
            apo_info["key"] = f"{apo_info['source']}:{apo_info['name']}"
            apo_info["chain_key"] = f"{entry_id}_{c.asym_id}"
            apo_info["entity_key"] = f"{entry_id}:{c.entity_id}"

            # Load the apo structure from LMDB (if not already loaded)
            success = self._load_apo_info_from_lmdb(apo_info)
            if not success:
                continue

            # Load the apo structure tokens
            _ = self._load_apo_tok_from_lmdb(apo_info)

            apo_lookup[c.asym_id] = apo_info
            cache[eid] = apo_info  # cache for other chains of the same entity

        return apo_lookup

    def _get_polymer_apo_coords(
        self,
        chain: Chain,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Fetch the apo coordinates for a protein chain."""
        assert chain.is_polymer

        def fallback():
            return chain.map_atom_coords_to_residue_coords(chain.atom.coords)

        # If the apo structure is not found, use the original coordinates as apo
        # for training. For validation, raise an error.
        if chain.asym_id not in apo_lookup:
            if self.train:
                self.logger.warning(
                    f"Apo structure not found for chain {chain.asym_id} in lookup."
                    f"Use the original coordinates as apo."
                )
                return fallback()
            else:
                raise KeyError(
                    f"Apo structure not found for chain {chain.asym_id} in lookup."
                )

        # Get the apo coordinates from the lookup
        apo_info = apo_lookup[chain.asym_id]
        key = apo_info["key"]
        seq = apo_info["seq"]
        coords = apo_info["coords"]
        res_map = apo_info.get("residue_map", None)
        if not self.train:
            # NOTE: While res mapping is also allowed for inference,
            # we enforce that there is no missing residues in the apo for simplicity.
            assert res_map is None, (
                f"Residue map {res_map} found for chain {chain.asym_id} in lookup."
                f"Residue mapping is only allowed for training."
            )

        # Perturb the apo coordinates
        # NOTE: We only perturb protein apo structures right now.
        if (
            chain.is_protein
            and self.apo_perturb is not None
            and rng.random() < self.config.prob_perturbation
        ):
            coords = self.apo_perturb.run_protein_perturbation(
                seq, coords, mask=None, rng=rng, rieprody_key=key
            )

        # Map the apo coordinates to the chain's coordinates
        length = chain.num_residues
        if res_map is not None:
            res_st, res_end, apo_st, apo_end = parse_residue_map(res_map)
            if res_st == -1:
                self.logger.warning(
                    f"Invalid residue map {res_map} for chain {key}. "
                    f"Use the original coordinates as apo."
                )
                return fallback()
            if res_st == 0 and res_end == length:
                # Simply crop apo_coords without padding
                coords = coords[apo_st:apo_end]
            else:
                # Need to pad coords to match the full sequence length
                padded_coords = np.full(
                    (length, coords.shape[1], 3), np.nan, dtype=np.float32
                )
                padded_coords[res_st:res_end] = coords[apo_st:apo_end]
                coords = padded_coords

        assert coords.shape[0] == chain.num_residues, (
            f"Apo coordinates length {coords.shape[0]} does not match "
            f"chain length {length} for chain {key}."
        )
        return coords

    def _get_ligand_apo_coords(
        self, chain: Chain, rng: np.random.Generator, key: str
    ) -> np.ndarray:
        """Fetch the apo coordinates for a ligand chain."""
        assert chain.is_ligand
        if chain.smiles is not None:
            ref_comp = Component.from_smiles("LIG", chain.smiles)
            coords = ref_comp.get_ref_conformer(rng, train=True)
        else:
            coords = np.full_like(chain.atom.coords, np.nan)
            ccd_sequence = chain.get_ccd_sequence()
            for res_i in range(chain.num_residues):
                res_idx = res_i + 1  # 1-based
                code = ccd_sequence[res_i]
                if code not in self.ccd:
                    self.logger.warning(
                        f"CCD code {code} not found for ligand {key}."
                        f"Filling with NaN coordinates."
                    )
                    continue
                ref_comp = self.ccd[code]
                ref_pos = ref_comp.get_ref_conformer(rng, train=True)
                ref_atom_order = ref_comp.get_atom_index_map()
                # Map reference conformer to chain's atom order
                src_atom_indices: list[int] = []
                dst_atom_indices: list[int] = []
                for atom_i in chain.residue.iter_residue_atoms(res_idx):
                    an = chain.atom.name[atom_i]
                    if an in ref_atom_order:
                        src_atom_indices.append(ref_atom_order[an])
                        dst_atom_indices.append(atom_i)
                coords[dst_atom_indices] = ref_pos[src_atom_indices]
        coords = np.expand_dims(coords, axis=1)  # [Natom, 1, 3]
        return coords

    def fetch_apo_structures(
        self,
        ref_struct: RefStructure,
        apo_lookup: dict[int, dict],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Return the apo coordinates for the given reference structure.
        Key: asymmetric chain ID (asym_id)
        Value: apo coordinates of shape:
            - Protein: [L, 37, 3]
            - RNA/DNA: [L, 29, 3]
            - Ligand: [Natom, 1, 3]
        """
        chain_coords: dict[int, np.ndarray] = {}
        for c in ref_struct.chains:
            key = f"{ref_struct.id}_{c.asym_id}"
            if c.is_polymer:
                coords = self._get_polymer_apo_coords(c, apo_lookup, rng)
            else:
                # For ligand chains, use etkdg conformer.
                coords = self._get_ligand_apo_coords(c, rng, key)
            if c.is_protein:
                assert coords.shape == (c.num_residues, 37, 3)
            elif c.is_nucleic_acid:
                assert coords.shape == (c.num_residues, 29, 3)
            else:
                assert coords.shape == (c.num_atoms, 1, 3)
            chain_coords[c.asym_id] = coords
        return chain_coords

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample prior coordinates for the given structure."""
        if self.num_priors <= 0 or self.prior_sampler is None:
            return np.empty((0, ref_struct.num_atoms, 3), dtype=np.float32)

        return self.prior_sampler(ref_struct, apo_dict, self.num_priors, rng)

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

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        """Populate the structure tokens for the given tokenized structure."""
        for c_i in range(tokenized.num_chains):
            if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                continue  # only populate structure tokens for protein chains

            asym_id = int(tokenized.chain.asym_id[c_i])
            k = f"{tokenized.id}-{asym_id}"  # For logging purpose

            apo_info = apo_lookup.get(asym_id)
            if apo_info is None:
                self.logger.warning(
                    f"No apo info for chain `{k}` in apo lookup. Skipping."
                )
                continue

            tok = apo_info.get("tokens", None)
            res_map = apo_info.get("residue_map", None)

            if tok is not None:
                key = apo_info["key"]
                self._insert_structure_tokens(tokenized, c_i, tok, res_map, key=key)

    def _insert_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        c_i: int,
        apo_tok: np.ndarray,
        res_map: str | None = None,
        *,
        key: str,
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

        if res_map is None:
            if len(bb_tok) != seq_token_len:
                self.logger.warning(
                    f"Apo tokens ({key}, len={toklen}) cannot be "
                    f"aligned with sequence tokens (len={seq_token_len}) for "
                    f"chain {c_i} without residue map. Skipping."
                )
                return
            bb_struct_token_id[seq_start + 1 : seq_end - 1] = bb_tok
            fa_struct_token_id[seq_start + 1 : seq_end - 1] = fa_tok
        else:
            res_st, res_end, apo_st, apo_end = parse_residue_map(res_map)
            if res_st == -1:
                self.logger.warning(
                    f"Invalid residue map {res_map} for chain {c_i}. Skipping."
                )
                return
            if toklen < (apo_end - apo_st) or (seq_token_len < res_end):
                self.logger.warning(
                    f"Apo tokens ({key}, len={toklen}) cannot cover "
                    f"the residue mapping for chain {c_i}: {res_map}. Skipping."
                )
                return
            _st, _end = seq_start + 1 + res_st, seq_start + 1 + res_end
            bb_struct_token_id[_st:_end] = bb_tok[apo_st:apo_end]
            fa_struct_token_id[_st:_end] = fa_tok[apo_st:apo_end]
