"""Build model inputs from validated queries and prepared apo/prior candidates."""

import itertools
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch

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
from kfold.inference.apo_runner import ApoChain, ApoMultimer
from kfold.inference.query import LigandSequence, PolymerSequence, ProteinPair, Query


class InferenceInput(NamedTuple):
    """A query, its reference structure, and the prepared model input.

    apos and priors contain one candidate list per entry in query.protein_entries
    order. Each candidate holds one monomer chain or both protein-pair components.
    """

    query: Query
    seed: int  # Seed used for prior sampling and tokenization.
    ref_struct: RefStructure
    f_input: FoldingInput
    apos: list[list[ApoChain] | list[ApoMultimer]]
    priors: list[list[ApoChain] | list[ApoMultimer]]


class InputDataPipeline:
    """Build reference structures and model features from queries and candidates."""

    def __init__(self, ccd: CCD) -> None:
        self.ccd = ccd
        self.prior_sampler = prior_sampling.PriorSampler.inference_mode()
        self.tokenizer = tokenization.Tokenizer(ccd)
        self.featurizer = featurization.InputFeaturizer()

    def build_input(
        self,
        query: Query,
        seed: int,
        num_samples: int = 5,
        *,
        apos: list[list[ApoChain] | list[ApoMultimer]],
        priors: list[list[ApoChain] | list[ApoMultimer]],
    ) -> InferenceInput:
        """Build model features from a query and its prepared candidates.

        Parameters
        ----------
        query : Query
            Validated query describing chains and bonds.
        seed : int
            Seed for prior sampling and tokenization.
        num_samples : int
            Number of prior coordinate samples to prepare for prediction.
        apos: list[list[ApoChain] | list[ApoMultimer]]
            Aligned, tokenized apo candidates in query.protein_entries order.
        priors: list[list[ApoChain] | list[ApoMultimer]]
            Aligned prior candidates in the same entry order.

        Returns
        -------
        InferenceInput
            Unbatched model input with the query, reference structure, and candidates.
        """
        # Validate candidate lists.
        if any(not candidates for candidates in [*apos, *priors]):
            raise ValueError(
                "Each protein entry requires non-empty apo and prior candidate lists."
            )

        # Use independent random streams for prior sampling and tokenization.
        prior_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        tokenizer_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

        # Build reference chains and bonds from the query.
        ref_struct = self.build_reference_structure(query)
        num_apo = max((len(candidates) for candidates in apos), default=1)

        # Place aligned apo and prior coordinates onto query chain copies.
        apo_coords, prior_candidates, apo_chains_by_asym = self._map_candidates_to_chains(
            query, ref_struct, apos, priors, num_apo
        )

        # Sample initial coordinates for each prediction.
        prior_coords = self.prior_sampler.sample(
            ref_struct, prior_candidates, num_samples, prior_rng
        )

        # Tokenize the reference structure.
        tokenized = self.tokenizer(
            ref_struct,
            tokenizer_rng,
            apo_coords=apo_coords,
            num_apo=num_apo,
            prior_coords=prior_coords,
        )
        # Featurize the tokenized structure into model input tensors.
        f_input = self.featurizer(tokenized)

        # Insert the precomputed apo structure tokens at aligned residue positions.
        self._insert_apo_structure_tokens(f_input, apo_chains_by_asym)

        # Pad feature dimensions to the model's required multiples.
        f_input = f_input.pad(
            max_tokens=((f_input.num_tokens + 31) // 32) * 32,
            max_atoms=((f_input.num_atoms + 63) // 64) * 64,
            max_sequence_tokens=((f_input.num_sequence_tokens + 63) // 64) * 64,
        )
        return InferenceInput(query, seed, ref_struct, f_input, apos, priors)

    @staticmethod
    def _map_candidates_to_chains(
        query: Query,
        ref_struct: RefStructure,
        apos: list[list[ApoChain] | list[ApoMultimer]],
        priors: list[list[ApoChain] | list[ApoMultimer]],
        num_apo: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, list[ApoChain]]]:
        """Place aligned candidates on query sequences and map them to chain copies."""
        asym_by_name = {chain.name: chain.asym_id for chain in ref_struct.metadata.chains}
        apo_coords: dict[int, np.ndarray] = {}
        prior_candidates: dict[int, np.ndarray] = {}
        apo_chains_by_asym: dict[int, list[ApoChain]] = {}
        for entry, entry_apos, entry_priors in zip(
            query.protein_entries, apos, priors, strict=True
        ):
            # Separate pair components while preserving candidate order across chains.
            if isinstance(entry, ProteinPair):
                components = [
                    (
                        [pair[0] for pair in entry.id],
                        entry.sequence1,
                        [apo.chain1 for apo in entry_apos],
                        [prior.chain1 for prior in entry_priors],
                    ),
                    (
                        [pair[1] for pair in entry.id],
                        entry.sequence2,
                        [apo.chain2 for apo in entry_apos],
                        [prior.chain2 for prior in entry_priors],
                    ),
                ]
            else:
                components = [(entry.id, entry.sequence, entry_apos, entry_priors)]
            # Validate each component before assembling arrays for its chain copies.
            for chain_ids, sequence, apo_chains, prior_chains in components:
                for chain in itertools.chain(apo_chains, prior_chains):
                    if chain.sequence != sequence:
                        raise ValueError(
                            f"Candidate sequence does not match query chains {chain_ids}."
                        )
                    if chain.coordinates.shape != (len(sequence), 37, 3):
                        raise ValueError(
                            f"Candidate coordinates for chains {chain_ids} must have "
                            f"shape ({len(sequence)}, 37, 3)."
                        )

                # Apo features share a candidate dimension across all entries.
                apo_array = np.full(
                    (num_apo, len(sequence), 37, 3), np.nan, dtype=np.float32
                )
                for apo_index, chain in enumerate(apo_chains):
                    apo_array[apo_index] = chain.coordinates

                # Priors keep every candidate for sampling, without candidate padding.
                prior_array = np.stack(
                    [chain.coordinates for chain in prior_chains]
                ).astype(np.float32, copy=False)

                # Copies share source coordinates and encoded structure tokens.
                for chain_id in chain_ids:
                    asym_id = asym_by_name[chain_id]
                    apo_coords[asym_id] = apo_array
                    prior_candidates[asym_id] = prior_array
                    apo_chains_by_asym[asym_id] = apo_chains

        return apo_coords, prior_candidates, apo_chains_by_asym

    @staticmethod
    def _insert_apo_structure_tokens(
        f_input: FoldingInput, apo_chains_by_asym: dict[int, list[ApoChain]]
    ) -> None:
        """Insert precomputed BB/FA tokens into model input features in place."""
        for asym_id, apo_chains in apo_chains_by_asym.items():
            # The first sequence position is the chain's beginning-of-sequence token.
            residue_start = (
                int(torch.where(f_input.sequence.asym_id == asym_id)[0][0]) + 1
            )
            for apo_index, chain in enumerate(apo_chains):
                if chain.bb_tokens is None:
                    continue
                start = residue_start
                end = start + len(chain.sequence)
                f_input.sequence.bb_struct_token_id[start:end, apo_index] = (
                    torch.from_numpy(chain.bb_tokens)
                )
                f_input.sequence.fa_struct_token_id[start:end, apo_index] = (
                    torch.from_numpy(chain.fa_tokens)
                )

    def build_reference_structure(self, query: Query) -> RefStructure:
        """Build chains for each query entry and copy, then add the specified bonds.

        Parameters
        ----------
        query : Query
            Validated query describing sequences, copies, and bonds.

        Returns
        -------
        RefStructure
            Reference structure in query entry, copy, then component order.
            Each protein-pair copy shares an apo/prior coordinate frame.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        entity_ids = itertools.count(1)
        asym_ids = itertools.count(1)
        asym_by_name: dict[str, int] = {}
        # Collect bonded atoms before preparing copy-specific ligand chemistry.
        bonded_atoms: dict[str, dict[int, set[str]]] = defaultdict(dict)
        for bond in query.bonds:
            for chain_id, residue_index, atom in (bond.atom1, bond.atom2):
                bonded_atoms[chain_id].setdefault(residue_index, set()).add(atom)

        # Build entity chains once, then expand them into the requested copies.
        for entry in query.sequences:
            if isinstance(entry, ProteinPair):
                entity_chains = [
                    structure_preparation.prepare_ref_chain(
                        entry.ctype, codes, ccd=self.ccd, entity_id=next(entity_ids)
                    )
                    for codes in (entry.ccd_sequence1, entry.ccd_sequence2)
                ]
                copies = entry.id
            else:
                entity_chains = [self._prepare_chain(entry, next(entity_ids))]
                copies = [[chain_id] for chain_id in entry.id]

            for sym_id, copy_ids in enumerate(copies, start=1):
                copy_asym_ids = [next(asym_ids) for _ in copy_ids]
                # Both chains in a pair move together; each copy is independent.
                rigid_group_uid = copy_asym_ids[0]
                for name, asym_id, entity_chain in zip(
                    copy_ids, copy_asym_ids, entity_chains, strict=True
                ):
                    asym_by_name[name] = asym_id
                    if isinstance(entry, LigandSequence) and name in bonded_atoms:
                        # Bonded atoms can change which leaving atoms this copy retains.
                        chain = self._prepare_chain(
                            entry, entity_chain.entity_id, bonded_atoms[name]
                        ).copy_with(asym_id=asym_id, sym_id=sym_id)
                    else:
                        chain = entity_chain.copy_with(
                            asym_id=asym_id, sym_id=sym_id, deepcopy=(sym_id > 1)
                        )
                    chains.append(chain)
                    chain_metas.append(
                        ChainInfo(
                            type=chain.ctype,
                            name=name,
                            entity_id=chain.entity_id,
                            asym_id=asym_id,
                            sym_id=sym_id,
                            num_residues=chain.num_residues,
                            num_tokens=chain.num_tokens,
                            num_atoms=chain.num_atoms,
                            smiles=chain.smiles,
                            description=entry.description,
                            apo_uid=rigid_group_uid,
                            prior_uid=rigid_group_uid,
                        )
                    )

        # Translate named query bonds into reference-structure chain indices.
        connections = []
        for bond in query.bonds:
            chain_id1, residue_index1, atom1 = bond.atom1
            chain_id2, residue_index2, atom2 = bond.atom2
            connections.append(
                CovalentConnection(
                    (asym_by_name[chain_id1], asym_by_name[chain_id2]),
                    (residue_index1, residue_index2),
                    (atom1, atom2),
                )
            )
        # Assemble chains, connections, and metadata into one reference structure.
        return structure_preparation.prepare_structure(
            chains,
            connections,
            Metadata(id=query.name, source="query", chains=chain_metas),
        )

    def _prepare_chain(
        self,
        entry: PolymerSequence | LigandSequence,
        entity_id: int,
        bonded_atoms: dict[int, set[str]] | None = None,
    ) -> Chain:
        if isinstance(entry, PolymerSequence):
            codes, smiles = entry.ccd_sequence, None
        else:
            codes = entry.ccd if entry.ccd is not None else [f"LIG{entity_id}"]
            smiles = entry.smiles
        return structure_preparation.prepare_ref_chain(
            entry.ctype,
            codes,
            ccd=self.ccd,
            smiles=smiles,
            entity_id=entity_id,
            bonded_atoms=bonded_atoms,
        )
