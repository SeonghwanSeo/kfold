"""Dataset classes for training with AF2 monomer distillation data.

Since monomer distillation data is very large, we use a simplified
data format and metadata format.

Structure: each sample is stored as a npz file containing:
    'sequence': np.ndarray(shape=(L,), dtype='S')
    'coordinates': np.ndarray(shape=(Natom, 3), dtype=np.float32)
    'b_factors': np.ndarray(shape=(Natom,), dtype=np.float32)

Metadata: each sample has a simple metadata dict containing:
    'id': str, the unique identifier of the sample
    'pred': dict, the prediction record containing:
        'model': str, same to 'AF2'.
        'plddt': float, the average pLDDT-Cα of the structure.

There is major difference between monomer distillation dataset and other datasets:
    - No apo coordinates (trunk) -> Fill to NaN.
    - No apo structure tokens (trunk) -> Fill to 0.0.
    - Use perturbed label coordinates as prior (diffusion bridge)
"""

import io

import numpy as np

import kfold.constants as C
from kfold.data.pipelines import structure_preparation
from kfold.data.types.metadata import ChainInfo, Metadata, PredictionRecord
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure

from .distillation import DistillationDataset

StructInfo = dict


def get_chain_info() -> ChainInfo:
    """Get the ChainInfo for the monomer"""
    return ChainInfo(
        name="A",
        type=C.ChainType.PROTEIN.value,
        entity_id=1,
        asym_id=1,
        sym_id=1,
        num_residues=0,
        num_atoms=0,
        num_tokens=0,
    )


class MonomerDistillationDataset(DistillationDataset):
    """Training dataset for monomer distillation."""

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.metadatas`.
        """
        trials = []
        for _ in range(num_trials):
            sample = self.samples[index]
            metadata_dict = sample.metadata
            metadata = Metadata(
                id=metadata_dict["id"],
                source="pred",
                pred=PredictionRecord(**metadata_dict["pred"]),
                chains=[get_chain_info()],
            )
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

    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""
        name = metadata.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB.")

        with io.BytesIO(value_bytes) as byte_stream:
            with np.load(byte_stream) as data:
                seq = data["sequence"].item().decode("utf-8")
                coords = data["coordinates"]  # [Natom, 3]

        ccd_sequence = [C.residue.PROTEIN_ONE_TO_THREE[aa] for aa in seq]
        chain = structure_preparation.prepare_ref_chain(
            C.ChainType.PROTEIN,
            ccd_sequence,
            ccd=self.ccd,
            entity_id=1,
            asym_id=1,
            sym_id=1,
        )

        # The coordinates are already ordered
        chain.atom.coords[:] = coords

        metadata.chains[0].num_residues = chain.num_residues
        metadata.chains[0].num_atoms = chain.num_atoms
        metadata.chains[0].num_tokens = chain.num_tokens

        ref_struct = RefStructure(
            chains=[chain], connections=[], metadata=metadata.copy()
        )
        return ref_struct

    def load_lookup_table(self) -> dict:
        return {}

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Get the apo lookup for the given reference structure."""
        c = ref_struct.chains[0]
        seq = c.get_sequence()
        ccd_sequence = c.get_ccd_sequence()

        res_atom_dict = C.atom.residue_atoms
        atom37_order = C.atom.protein_atom37_order

        # Convert to atom37 format
        apo_coords = c.atom.coords  # [Natom, 3]
        apo_coords_37 = np.full((c.num_residues, 37, 3), np.nan, dtype=np.float32)
        g_atom_i = 0
        for i, restype in enumerate(ccd_sequence):
            atoms = res_atom_dict[restype]
            natoms = len(atoms)
            st, end = g_atom_i, g_atom_i + natoms
            atom_indices = [atom37_order[a] for a in atoms]
            apo_coords_37[i, atom_indices] = apo_coords[st:end]
            g_atom_i += natoms
        assert g_atom_i == c.num_atoms, (
            f"Total atom counts {g_atom_i} does not match chain.num_atoms {c.num_atoms}."
        )

        return {c.asym_id: {"coords": apo_coords_37, "seq": seq}}

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        return  # skip populating structure tokens for monomer distillation dataset


class RNAMonomerDistillationDataset(DistillationDataset):
    """Training dataset for rna monomer distillation."""

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.metadatas`.
        """
        trials = []
        for _ in range(num_trials):
            sample = self.samples[index]
            metadata_dict = sample.metadata
            metadata = Metadata(
                id=metadata_dict["id"],
                source="pred",
                pred=PredictionRecord(**metadata_dict["pred"]),
                chains=[get_chain_info()],
            )
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

    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""
        name = metadata.id
        key_bytes = name.encode("utf-8")
        with self.lmdb_env.begin(write=False) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Record {name} not found in LMDB.")

        with io.BytesIO(value_bytes) as byte_stream:
            with np.load(byte_stream) as data:
                seq = data["sequence"].item().decode("utf-8")
                coords = data["coordinates"]  # [Natom, 3]

        ccd_sequence = list(seq)
        chain = structure_preparation.prepare_ref_chain(
            C.ChainType.RNA,
            ccd_sequence,
            ccd=self.ccd,
            entity_id=1,
            asym_id=1,
            sym_id=1,
        )

        # The coordinates are already ordered
        chain.atom.coords[:] = coords

        metadata.chains[0].num_residues = chain.num_residues
        metadata.chains[0].num_atoms = chain.num_atoms
        metadata.chains[0].num_tokens = chain.num_tokens

        ref_struct = RefStructure(
            chains=[chain], connections=[], metadata=metadata.copy()
        )
        return ref_struct

    def load_lookup_table(self) -> dict:
        return {}

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        return {}

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        return
