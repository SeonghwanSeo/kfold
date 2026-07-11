"""Dataset classes for training with Large-scale Monomer Distillation data.

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

Apo coordinates: we do not pre-compute apo coordinates for monomer distillation dataset.
Instead, we use the perturbed label coordinates as prior for diffusion bridge.

# For RNA, which is relatively small and there is no perturbation code, we provide
the apo structure distillation source for prior.

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


def get_chain_info(ctype: C.ChainType) -> ChainInfo:
    """Get the ChainInfo for the monomer"""
    return ChainInfo(
        name="A",
        type=ctype.value,
        entity_id=1,
        asym_id=1,
        sym_id=1,
        num_residues=0,
        num_atoms=0,
        num_tokens=0,
    )


class MonomerDistillationDataset(DistillationDataset):
    """Training dataset for monomer distillation."""

    ctype: C.ChainType = C.ChainType.PROTEIN

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        super().sanity_check()
        cfg = self.config
        if cfg.prob_perturbation != 1.0:
            self.logger.warning(
                "prob_perturbation is not 1.0 for monomer distillation dataset."
            )
        if cfg.prob_drop_apo != 1.0:
            self.logger.warning(
                "prob_drop_apo is not 1.0 for monomer distillation dataset."
            )
        if cfg.prob_drop_struct_token != 1.0:
            self.logger.warning(
                "prob_drop_struct_token is not 1.0 for monomer distillation dataset."
            )

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
                chains=[get_chain_info(self.ctype)],
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
        raise NotImplementedError(
            "load_ref_structure is not implemented for monomer distillation dataset."
        )

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        return  # skip populating structure tokens for monomer distillation dataset


class ProteinMonomerDistillationDataset(MonomerDistillationDataset):
    """Training dataset for large-scale protein monomer distillation.
    Since it is very large, we directly use the apo coordinates as label structure.
    """

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        super().sanity_check()
        assert self.config.apo_perturb is not None, (
            "apo_perturb must be provided for protein monomer distillation dataset."
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
        """Use the label monomer structure as the synthetic apo/prior source."""
        del rng
        chain = ref_struct.chains[0]
        return {
            chain.asym_id: {
                "key": f"{ref_struct.id}:{chain.asym_id}",
                "seq": chain.get_sequence(map_to_standard=True),
                "coords": chain.map_atom_coords_to_residue_coords(chain.atom.coords),
            }
        }

    def get_prior_coords(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Use the label monomer structure as the prior apo-like source."""
        del apo_dict
        chain = ref_struct.chains[0]
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        if chain.asym_id in metadata_by_asym_id:
            metadata_by_asym_id[chain.asym_id].prior_uid = chain.asym_id

        coords = chain.map_atom_coords_to_residue_coords(chain.atom.coords)

        # Perturb the label coordinates to apply harsh perturbation.
        assert self.apo_perturb is not None
        seq = chain.get_sequence(map_to_standard=True)
        coords = self.apo_perturb.run_protein_perturbation(
            seq, coords, mask=None, rng=rng
        )
        return {chain.asym_id: coords}


class RNAMonomerDistillationDataset(MonomerDistillationDataset):
    """Training dataset for rna monomer distillation with apo/prior LMDB."""

    ctype: C.ChainType = C.ChainType.RNA

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

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        return
