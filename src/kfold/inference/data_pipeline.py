import itertools
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch

import kfold.constants as C
from kfold.data.pipelines import (
    featurization,
    prior_sampling,
    structure_preparation,
    tokenization,
)
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, CovalentConnection, RefStructure

from . import query


class InferenceInput(NamedTuple):
    """One query and its unbatched, fully constructed model input."""

    query: query.Query
    ref_struct: RefStructure
    f_input: FoldingInput


class InputDataPipeline:
    def __init__(
        self,
        ccd: CCD,
        num_prior_samples: int = 5,
    ) -> None:
        """Initialize the input data pipeline.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary for residue information.
        num_prior_samples : int, optional
            Number of diffusion priors to create. Default is 5.
        """

        self.ccd: CCD = ccd

        self.prior_sampler = prior_sampling.PriorSampler.inference_mode()
        if num_prior_samples <= 0:
            raise ValueError("num_prior_samples must be positive.")
        self.num_prior_samples = num_prior_samples

        # Initialize tokenizer
        self.tokenizer = tokenization.Tokenizer(self.ccd)

        # Initialize featurizer
        self.featurizer: featurization.InputFeaturizer = featurization.InputFeaturizer()

    def build_input(self, input: query.Query, seed: int) -> InferenceInput:
        """Build an unbatched FoldingInput from a prepared query."""

        # Extract all protein sequences
        entries: list[query.ProteinSequence | query.ProteinMultimerSequence] = [
            seq for seq in input.sequences if isinstance(seq, query.ProteinSequence)
        ] + input.multimer_sequences

        # Validate that the query has been prepared
        if any(
            entry._apo_coords is None or entry._prior_coords is None for entry in entries
        ):
            raise ValueError("Call runner.prepare_query() before building the input.")

        # Initialize RNGs
        prior_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        tokenizer_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

        # Prepare reference structure
        ref_struct = self.read_query(input)

        # Prepare apo coordinates
        num_apo = max((len(entry._apo_coords[0]) for entry in entries), default=1)
        asym_by_name = {chain.name: chain.asym_id for chain in ref_struct.metadata.chains}
        apo_coords: dict[int, np.ndarray] = {}
        for entry in entries:
            multimer = isinstance(entry, query.ProteinMultimerSequence)
            component_ids = (
                list(zip(*entry.ids, strict=True)) if multimer else [entry.ids]
            )
            for ids, apos in zip(component_ids, entry._apo_coords, strict=True):
                padded = np.full((num_apo, *apos.shape[1:]), np.nan, dtype=np.float32)
                padded[: len(apos)] = apos
                for chain_id in ids:
                    asym_id = asym_by_name[chain_id]
                    apo_coords[asym_id] = padded

        # Sample prior coordinates
        prior_candidates: dict[int, np.ndarray] = {}
        for entry in entries:
            multimer = isinstance(entry, query.ProteinMultimerSequence)
            component_ids = (
                list(zip(*entry.ids, strict=True)) if multimer else [entry.ids]
            )
            for ids, priors in zip(component_ids, entry._prior_coords, strict=True):
                for chain_id in ids:
                    asym_id = asym_by_name[chain_id]
                    prior_candidates[asym_id] = priors
        prior_coords = self.prior_sampler.sample(
            ref_struct, prior_candidates, self.num_prior_samples, prior_rng
        )

        # Tokenize structure
        tokenized = self.tokenizer(
            ref_struct,
            tokenizer_rng,
            apo_coords=apo_coords,
            num_apo=num_apo,
            prior_coords=prior_coords,
        )

        f_input = self.featurizer(tokenized)
        for entry in entries:
            if entry._apo_token is None:
                continue
            component_ids = (
                list(zip(*entry.ids, strict=True))
                if isinstance(entry, query.ProteinMultimerSequence)
                else [entry.ids]
            )
            for ids, tokens in zip(component_ids, entry._apo_token, strict=True):
                for chain_id in ids:
                    asym_id = asym_by_name[chain_id]
                    st = int(torch.where(f_input.sequence.asym_id == asym_id)[0][0]) + 1
                    end = st + len(tokens[0]["bb_token_id"])
                    for apo_i, token_ids in enumerate(tokens):
                        f_input.sequence.bb_struct_token_id[st:end, apo_i] = (
                            torch.from_numpy(token_ids["bb_token_id"])
                        )
                        f_input.sequence.fa_struct_token_id[st:end, apo_i] = (
                            torch.from_numpy(token_ids["fa_token_id"])
                        )

        f_input = f_input.pad(
            max_tokens=((f_input.num_tokens + 31) // 32) * 32,
            max_atoms=((f_input.num_atoms + 63) // 64) * 64,
            max_sequence_tokens=((f_input.num_sequence_tokens + 63) // 64) * 64,
        )
        return InferenceInput(input, ref_struct, f_input)

    def read_query(self, input: query.Query) -> RefStructure:
        """Prepare the reference structure from the input file.

        Parameters
        ----------
        input : Query
            The input query file.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        entity_id_iter = itertools.count(1)
        asym_id_iter = itertools.count(1)
        chain_id_to_asym_id: dict[str, int] = {}

        # Collect bonded atoms
        chain_bonded_atoms: dict[str, dict[int, set[str]]] = defaultdict(dict)
        for bond in input.bonds:
            chain_id1, res_idx1, atom1 = bond.atom1
            chain_id2, res_idx2, atom2 = bond.atom2
            chain_bonded_atoms[chain_id1].setdefault(res_idx1, set()).add(atom1)
            chain_bonded_atoms[chain_id2].setdefault(res_idx2, set()).add(atom2)

        for seq in input.sequences:
            entity_id = next(entity_id_iter)
            # Prepare chain ids
            chain_names: list[str] = seq.ids
            num_chains = len(chain_names)
            asym_ids: list[int] = [next(asym_id_iter) for _ in range(num_chains)]
            sym_ids: list[int] = [i for i in range(1, num_chains + 1)]
            for name, asym_id in zip(chain_names, asym_ids, strict=True):
                chain_id_to_asym_id[name] = asym_id

            num_residues = len(seq)

            # Parse sequence
            entity_chain: Chain = self.parse_sequence(seq, entity_id)

            # Create copies for multiple chains
            for i in range(num_chains):
                # Assign chain ids
                chain_name: str = chain_names[i]
                asym_id: int = asym_ids[i]
                sym_id: int = sym_ids[i]

                # Create chain copy
                if entity_chain.is_ligand and chain_name in chain_bonded_atoms:
                    # If it's a ligand with covalent bonds, create a new chain
                    bonded_atoms = chain_bonded_atoms[chain_name]
                    chain = self.parse_sequence(seq, entity_id, bonded_atoms).copy_with(
                        asym_id=asym_id, sym_id=sym_id
                    )
                else:
                    # Otherwise, create a copy of the original chain with new ids
                    chain = entity_chain.copy_with(
                        asym_id=asym_id, sym_id=sym_id, deepcopy=(i > 0)
                    )
                chains.append(chain)

                # Add chain metadata
                chain_meta = ChainInfo(
                    type=entity_chain.ctype,
                    name=chain_name,
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    num_residues=num_residues,
                    num_tokens=chain.num_tokens,
                    num_atoms=chain.num_atoms,
                    description=seq.description,
                )
                if entity_chain.smiles is not None:
                    chain_meta.smiles = entity_chain.smiles
                chain_metas.append(chain_meta)

        # A multimer sequence describes two entities and one or more physical
        # copies of their shared-frame pair.
        for sequence_group in input.multimer_sequences:
            entity_id1 = next(entity_id_iter)
            entity_id2 = next(entity_id_iter)
            entity_chain1 = structure_preparation.prepare_ref_chain(
                chain_type=sequence_group.ctype,
                ccd_sequences=sequence_group.ccd_sequence1,
                ccd=self.ccd,
                entity_id=entity_id1,
            )
            entity_chain2 = structure_preparation.prepare_ref_chain(
                sequence_group.ctype,
                sequence_group.ccd_sequence2,
                ccd=self.ccd,
                entity_id=entity_id2,
            )

            for sym_id, (chain_name1, chain_name2) in enumerate(
                sequence_group.ids, start=1
            ):
                asym_id1 = next(asym_id_iter)
                asym_id2 = next(asym_id_iter)
                rigid_group_uid = asym_id1
                chain_id_to_asym_id[chain_name1] = asym_id1
                chain_id_to_asym_id[chain_name2] = asym_id2

                chain1 = entity_chain1.copy_with(
                    asym_id=asym_id1, sym_id=sym_id, deepcopy=(sym_id > 1)
                )
                chain2 = entity_chain2.copy_with(
                    asym_id=asym_id2, sym_id=sym_id, deepcopy=(sym_id > 1)
                )
                chains.extend((chain1, chain2))

                chain_metas.extend(
                    (
                        ChainInfo(
                            type=chain1.ctype,
                            name=chain_name1,
                            entity_id=entity_id1,
                            asym_id=asym_id1,
                            sym_id=sym_id,
                            num_residues=chain1.num_residues,
                            num_tokens=chain1.num_tokens,
                            num_atoms=chain1.num_atoms,
                            description=sequence_group.description,
                            apo_uid=rigid_group_uid,
                            prior_uid=rigid_group_uid,
                        ),
                        ChainInfo(
                            type=chain2.ctype,
                            name=chain_name2,
                            entity_id=entity_id2,
                            asym_id=asym_id2,
                            sym_id=sym_id,
                            num_residues=chain2.num_residues,
                            num_tokens=chain2.num_tokens,
                            num_atoms=chain2.num_atoms,
                            description=sequence_group.description,
                            apo_uid=rigid_group_uid,
                            prior_uid=rigid_group_uid,
                        ),
                    )
                )

        # Prepare metadata
        metadata = Metadata(id=input.name, source="query", chains=chain_metas)

        # Add covalent bonds
        connections: list[CovalentConnection] = []
        for bond in input.bonds:
            chain_id1, res_idx1, atom1 = bond.atom1
            chain_id2, res_idx2, atom2 = bond.atom2
            asym_id1 = chain_id_to_asym_id[chain_id1]
            asym_id2 = chain_id_to_asym_id[chain_id2]
            connections.append(
                CovalentConnection(
                    (asym_id1, asym_id2), (res_idx1, res_idx2), (atom1, atom2)
                )
            )

        # Return RefStructure
        ref_struct = structure_preparation.prepare_structure(
            chains, connections, metadata
        )
        return ref_struct

    def parse_sequence(
        self,
        seq: query.BaseSequence,
        entity_id: int,
        bonded_atoms: dict[int, set[str]] | None = None,
    ) -> Chain:
        """Parse a chain from the sequence input.

        Parameters
        ----------
        seq : Sequence
            The sequence input.
        entity_id : int
            The entity_id to assign to the chain.
        bonded_atoms : dict[int, set[str]], optional
            A dictionary mapping residue index to a set of atom names that are involved
            in covalent bonds.

        Returns
        -------
        chain: Chain
            The reference chain.

        Notes
        -----
        The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
        """
        if isinstance(seq, query.PolymerSequence):
            return structure_preparation.prepare_ref_chain(
                seq.ctype,
                seq.ccd_sequence,
                ccd=self.ccd,
                entity_id=entity_id,
                bonded_atoms=bonded_atoms,
            )
        elif isinstance(seq, query.LigandSequence):
            # Load ccd or smiles
            ctype = C.ChainType.LIGAND
            if seq.ccd_ids is not None:
                ccd_ids = seq.ccd_ids
                smiles = None
            else:
                assert seq.smiles is not None, (
                    "Either CCD code or SMILES must be provided."
                )
                ccd_ids = [f"LIG{entity_id}"]
                smiles = seq.smiles

            return structure_preparation.prepare_ref_chain(
                ctype,
                ccd_ids,
                ccd=self.ccd,
                smiles=smiles,
                entity_id=entity_id,
                bonded_atoms=bonded_atoms,
            )
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")
