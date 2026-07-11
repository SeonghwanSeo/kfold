"""Dataset classes for training with AF-multimer distillation data.

Since multimer distillation data quite large, we use a simplified
data format and metadata format.

Structure: each sample is stored as a npz file containing:
    '{asym_id}.sequence': np.ndarray(shape=(L,), dtype='S')
    '{asym_id}.coordinates': np.ndarray(shape=(Natom, 3), dtype=np.float32)
    '{asym_id}.b_factors': np.ndarray(shape=(Natom,), dtype=np.float32)

Metadata: each sample has a simple metadata dict containing:
    'id': str, the unique identifier of the sample
    'pred': dict, the prediction record containing:
        'model': str, same to 'AF2'.
        'plddt': float, the average pLDDT-Cα of the structure.
    'chains': list of dict, each dict contains:
        'asym_id': str, the asym_id of the chain
        'entity_id': int, the entity_id of the chain

There is major difference between monomer distillation dataset and other datasets:
    - No apo coordinates (trunk) -> Fill to NaN.
    - No apo structure tokens (trunk) -> Fill to 0.0.
    - Use perturbed label coordinates as prior (diffusion bridge)
"""

import io

import numpy as np

import kfold.constants as C
from kfold.data.pipelines import structure_preparation
from kfold.data.types.metadata import ChainInfo, InterfaceInfo, Metadata, PredictionRecord
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure

from .distillation import DistillationDataset

StructInfo = dict


class HomodimerDistillationDataset(DistillationDataset):
    """Training dataset for homodimer distillation."""

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        super().sanity_check()
        cfg = self.config
        assert self.config.apo_perturb is not None, (
            "apo_perturb must be set for homodimer distillation dataset.",
        )
        if cfg.prob_perturbation != 1.0:
            self.logger.warning(
                "prob_perturbation is not 1.0 for homodimer distillation dataset."
            )
        if cfg.prob_drop_apo != 1.0:
            self.logger.warning(
                "prob_drop_apo is not 1.0 for homodimer distillation dataset."
            )
        if cfg.prob_drop_struct_token != 1.0:
            self.logger.warning(
                "prob_drop_struct_token is not 1.0 for homodimer distillation dataset."
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
            assert len(metadata_dict["chains"]) == 2, (
                f"Expected 2 chains for homodimer, "
                f"but got {len(metadata_dict['chains'])}."
            )
            chain_infos = []
            for i, name in enumerate(["A", "B"], start=1):
                chain_info = ChainInfo(
                    name=name,
                    type=C.ChainType.PROTEIN.value,
                    entity_id=1,
                    asym_id=i,
                    sym_id=i,
                    num_residues=0,
                    num_atoms=0,
                    num_tokens=0,
                )
                chain_infos.append(chain_info)

            metadata = Metadata(
                id=metadata_dict["id"],
                source="pred",
                pred=PredictionRecord(**metadata_dict["pred"]),
                chains=chain_infos,
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
                seq1 = data["0.sequence"].item().decode("utf-8")
                coords1 = data["0.coordinates"]  # [Natom, 3]
                seq2 = data["1.sequence"].item().decode("utf-8")
                coords2 = data["1.coordinates"]  # [Natom, 3]
        assert seq1 == seq2, (
            f"Expected identical sequences for homodimer, but got {seq1} and {seq2}."
        )

        ccd_sequence = [C.residue.PROTEIN_ONE_TO_THREE[aa] for aa in seq1]
        chain = structure_preparation.prepare_ref_chain(
            C.ChainType.PROTEIN, ccd_sequence, ccd=self.ccd, entity_id=1
        )
        chain1 = chain.copy_with(asym_id=1, sym_id=1)
        chain2 = chain.copy_with(deepcopy=True, asym_id=2, sym_id=2)

        # The coordinates are already ordered
        chain1.atom.coords[:] = coords1
        chain2.atom.coords[:] = coords2

        for cm in metadata.chains:
            cm.num_residues = chain.num_residues
            cm.num_atoms = chain.num_atoms
            cm.num_tokens = chain.num_tokens

        ref_struct = RefStructure(
            chains=[chain1, chain2], connections=[], metadata=metadata.copy()
        )
        return ref_struct

    def load_lookup_table(self) -> dict:
        return {}

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, dict]:
        """Use one randomly selected holo chain as the shared synthetic apo source."""
        src_chain = ref_struct.chains[rng.integers(0, 2)]
        apo_info = {
            "key": f"{ref_struct.id}:{src_chain.asym_id}",
            "seq": src_chain.get_sequence(map_to_standard=True),
            "coords": self._center_label_residue_coords(src_chain),
        }
        return {chain.asym_id: apo_info.copy() for chain in ref_struct.chains}

    def get_prior_coords(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Use one randomly selected label chain as the shared prior source."""
        del apo_dict
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        for chain in ref_struct.chains:
            if chain.asym_id in metadata_by_asym_id:
                metadata_by_asym_id[chain.asym_id].prior_uid = chain.asym_id

        prior_coords_dict = {}
        for chain in ref_struct.chains:
            c = ref_struct.chains[0] if rng.random() < 0.5 else ref_struct.chains[1]
            coords = self._center_label_residue_coords(c)

            # Apply harsh perturbation to the prior coordinates.
            assert self.apo_perturb is not None
            seq = c.get_sequence(map_to_standard=True)
            coords = self.apo_perturb.run_protein_perturbation(
                seq, coords, mask=None, rng=rng
            )

            prior_coords_dict[chain.asym_id] = coords

        return prior_coords_dict

    def populate_structure_tokens(
        self, tokenized: TokenizedStructure, apo_lookup: dict[int, dict]
    ) -> None:
        return  # skip populating structure tokens for monomer distillation dataset


class HeterodimerDistillationDataset(DistillationDataset):
    """Training dataset for heterodimer distillation."""

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
            assert len(metadata_dict["chains"]) == 2, (
                f"Expected 2 chains for heterodimer, "
                f"but got {len(metadata_dict['chains'])}."
            )

            chain_infos = []
            for i, name in enumerate(["A", "B"], start=1):
                chain_info = ChainInfo(
                    name=name,
                    type=C.ChainType.PROTEIN.value,
                    entity_id=i,
                    asym_id=i,
                    sym_id=1,
                    num_residues=0,
                    num_atoms=0,
                    num_tokens=0,
                )
                chain_infos.append(chain_info)
            iface_infos = [InterfaceInfo((1, 2))]

            metadata = Metadata(
                id=metadata_dict["id"],
                source="pred",
                pred=PredictionRecord(**metadata_dict["pred"]),
                chains=chain_infos,
                interfaces=iface_infos,
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
                seq1 = data["0.sequence"].item().decode("utf-8")
                coords1 = data["0.coordinates"]  # [Natom, 3]
                seq2 = data["1.sequence"].item().decode("utf-8")
                coords2 = data["1.coordinates"]  # [Natom, 3]
        assert seq1 != seq2, (
            f"Expected different sequences for heterodimer, but got {seq1} and {seq2}."
        )

        ctype = C.ChainType.PROTEIN
        ccd_seq1 = [C.residue.PROTEIN_ONE_TO_THREE[aa] for aa in seq1]
        chain1 = structure_preparation.prepare_ref_chain(
            ctype, ccd_seq1, self.ccd, entity_id=1, asym_id=1, sym_id=1
        )
        ccd_seq2 = [C.residue.PROTEIN_ONE_TO_THREE[aa] for aa in seq2]
        chain2 = structure_preparation.prepare_ref_chain(
            ctype, ccd_seq2, self.ccd, entity_id=2, asym_id=2, sym_id=1
        )

        # The coordinates are already ordered
        chain1.atom.coords[:] = coords1
        chain2.atom.coords[:] = coords2

        for c, cm in zip([chain1, chain2], metadata.chains, strict=True):
            cm.num_residues = c.num_residues
            cm.num_atoms = c.num_atoms
            cm.num_tokens = c.num_tokens

        ref_struct = RefStructure(
            chains=[chain1, chain2], connections=[], metadata=metadata.copy()
        )
        return ref_struct
