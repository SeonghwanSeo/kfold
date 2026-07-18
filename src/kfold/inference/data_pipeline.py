import itertools
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

import kfold.constants as C
from kfold.data.pipelines import (
    featurization,
    prior_sampling,
    structure_preparation,
    tokenization,
)
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.constraint import Constraint
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, CovalentConnection, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.io.structure import (
    read_protein_multimer_structure,
    read_protein_structure,
    read_rna_structure,
)

from . import query


@dataclass
class ResolvedStructureSources:
    """Normalized apo/prior inputs for one inference query."""

    apo_coords: dict[int, np.ndarray]
    prior_sources: list[dict[int, np.ndarray]]
    struct_token_records: list[dict]


@dataclass
class _AlignedStructureSource:
    """One parsed structure aligned to its query sequence."""

    sequence: str
    coords: np.ndarray
    target_coords: np.ndarray
    mappings: list[tuple[int, int, int, int]]


def _best_sequence_mapping(
    target_sequence: str,
    source_sequence: str,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Globally align two sequences and return contiguous diagonal runs.

    Each returned tuple is ``(target_start, target_end, source_start,
    source_end)``. Insertions in the source are omitted, while deletions in the
    source become gaps between target ranges.
    """
    diagonal, target_gap, source_gap = 1, 2, 3
    match_score, mismatch_score, gap_score = 2, -1, -2
    target_length = len(target_sequence)
    source_length = len(source_sequence)

    trace = np.zeros((target_length + 1, source_length + 1), dtype=np.uint8)
    trace[1:, 0] = target_gap
    trace[0, 1:] = source_gap
    previous_scores = [source_i * gap_score for source_i in range(source_length + 1)]

    for target_i, target_residue in enumerate(target_sequence, start=1):
        current_scores = [target_i * gap_score] + [0] * source_length
        for source_i, source_residue in enumerate(source_sequence, start=1):
            diagonal_score = previous_scores[source_i - 1] + (
                match_score if target_residue == source_residue else mismatch_score
            )
            target_gap_score = previous_scores[source_i] + gap_score
            source_gap_score = current_scores[source_i - 1] + gap_score
            if diagonal_score >= target_gap_score and diagonal_score >= source_gap_score:
                current_scores[source_i] = diagonal_score
                trace[target_i, source_i] = diagonal
            elif target_gap_score >= source_gap_score:
                current_scores[source_i] = target_gap_score
                trace[target_i, source_i] = target_gap
            else:
                current_scores[source_i] = source_gap_score
                trace[target_i, source_i] = source_gap
        previous_scores = current_scores

    target_i = target_length
    source_i = source_length
    aligned_pairs: list[tuple[int, int]] = []
    while target_i > 0 or source_i > 0:
        direction = trace[target_i, source_i]
        if direction == diagonal:
            target_i -= 1
            source_i -= 1
            aligned_pairs.append((target_i, source_i))
        elif direction == target_gap:
            target_i -= 1
        elif direction == source_gap:
            source_i -= 1
        else:
            raise RuntimeError("Sequence alignment traceback reached an invalid state.")
    aligned_pairs.reverse()

    mappings: list[tuple[int, int, int, int]] = []
    for mapped_target_i, mapped_source_i in aligned_pairs:
        if mappings and (
            mapped_target_i == mappings[-1][1] and mapped_source_i == mappings[-1][3]
        ):
            target_start, _, source_start, _ = mappings[-1]
            mappings[-1] = (
                target_start,
                mapped_target_i + 1,
                source_start,
                mapped_source_i + 1,
            )
        else:
            mappings.append(
                (
                    mapped_target_i,
                    mapped_target_i + 1,
                    mapped_source_i,
                    mapped_source_i + 1,
                )
            )

    num_matches = sum(
        target_sequence[target_i] == source_sequence[source_i]
        for target_i, source_i in aligned_pairs
    )
    return mappings, num_matches


class InputDataPipeline:
    def __init__(self, ccd: CCD, num_prior_samples: int = 5) -> None:
        """Initialize the input data pipeline.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary for residue information.
        num_prior_samples : int, optional
            Number of heuristic DNA/ligand priors to create. Default is 5.
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

        self.logger = logging.getLogger("InputDataPipeline")
        self.logger.setLevel(logging.INFO)

    def __call__(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, dict[int, dict]]:
        """Process an Query into model-ready inputs.

        Parameters
        ----------
        input : Query
            The input file containing sequences and metadata.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        tokenized_struct : TokenizedStructure
            The tokenized structure.
        f_input : FoldingInput
            The featurized model input.
        struct_tok_input : dict[int, dict]
            Raw protein structures and target sequence mappings for learned
            structure tokenization.
        """
        return self.run(input)

    def run(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, dict[int, dict]]:
        """Process an Query into model-ready inputs.

        Parameters
        ----------
        input : Query
            The input file containing sequences and metadata.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        tokenized_struct : TokenizedStructure
            The tokenized structure.
        f_input : FoldingInput
            The featurized model input.
        struct_tok_input : dict[int, dict]
            Raw protein structures and target sequence mappings for learned
            structure tokenization.
        """
        # Prepare structure from input file
        ref_struct, constraints = self.read_query(input)

        source_rng = np.random.default_rng(np.random.SeedSequence([input.seed, 0]))
        sources = self.resolve_structure_sources(ref_struct, input, source_rng)
        return self.prepare_model_input(input.seed, ref_struct, constraints, sources)

    def prepare_model_input(
        self,
        seed: int,
        ref_struct: RefStructure,
        constraints: list[Constraint],
        sources: ResolvedStructureSources,
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, dict[int, dict]]:
        """Convert normalized custom or generated sources into model input."""
        prior_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        tokenizer_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))

        # Each resolved source contributes exactly one prior. Perturbation,
        # relaxation, and rigid augmentation stay inside PriorSampler.
        prior_coords = np.stack(
            [
                self.prior_sampler.sample(ref_struct, source, 1, prior_rng)[0]
                for source in sources.prior_sources
            ],
            axis=0,
        )

        # Tokenize structure
        # Learned structure token IDs are added later on the model device.
        tokenized = self.tokenizer(
            ref_struct,
            tokenizer_rng,
            apo_coords=sources.apo_coords,
            prior_coords=prior_coords,
            constraints=constraints,
        )

        # Featurize input
        f_input = self.featurizer(tokenized)

        # Prepare structure tokenization input for later use in model inference
        struct_tok_input = self.prepare_struct_tok_input(
            f_input, sources.struct_token_records
        )
        return ref_struct, tokenized, f_input, struct_tok_input

    def read_query(self, input: query.Query) -> tuple[RefStructure, list[Constraint]]:
        """Prepare the reference structure from the input file.

        Parameters
        ----------
        input : Query
            The input query file.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        constraints : list[Constraint]
            The list of distance constraints specified in the input.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        entity_id_iter = itertools.count(1)
        asym_id_iter = itertools.count(1)
        chain_id_to_asym_id: dict[str, int] = {}

        # Collect bonded atoms
        chain_bonded_atoms: dict[str, dict[int, set[str]]] = defaultdict(dict)
        for constraint in input.constraints:
            if constraint.type == "bond":
                chain_id1, res_idx1, atom1 = constraint.atom1
                chain_id2, res_idx2, atom2 = constraint.atom2
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

            for copy_i, (chain_name1, chain_name2) in enumerate(
                sequence_group.ids, start=1
            ):
                asym_id1 = next(asym_id_iter)
                asym_id2 = next(asym_id_iter)
                rigid_group_uid = asym_id1
                chain_id_to_asym_id[chain_name1] = asym_id1
                chain_id_to_asym_id[chain_name2] = asym_id2

                chain1 = entity_chain1.copy_with(
                    asym_id=asym_id1,
                    sym_id=copy_i,
                    deepcopy=(copy_i > 1),
                )
                chain2 = entity_chain2.copy_with(
                    asym_id=asym_id2,
                    sym_id=copy_i,
                    deepcopy=(copy_i > 1),
                )
                chains.extend((chain1, chain2))

                chain_metas.extend(
                    (
                        ChainInfo(
                            type=chain1.ctype,
                            name=chain_name1,
                            entity_id=entity_id1,
                            asym_id=asym_id1,
                            sym_id=copy_i,
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
                            sym_id=copy_i,
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
        metadata = Metadata(
            id=input.name,
            source="query",
            chains=chain_metas,
        )

        # Add covalent bond and constraint
        connections: list[CovalentConnection] = []
        constraints: list[Constraint] = []
        for constraint in input.constraints:
            if constraint.type == "bond":
                chain_id1, res_idx1, atom1 = constraint.atom1
                chain_id2, res_idx2, atom2 = constraint.atom2
                asym_id1 = chain_id_to_asym_id[chain_id1]
                asym_id2 = chain_id_to_asym_id[chain_id2]
                connections.append(
                    CovalentConnection(
                        (asym_id1, asym_id2), (res_idx1, res_idx2), (atom1, atom2)
                    )
                )
            elif constraint.type == "distance":
                chain_id1, res_idx1, atom1 = constraint.atom1
                chain_id2, res_idx2, atom2 = constraint.atom2
                lower_bound, upper_bound = constraint.range  # type: ignore
                asym_id1 = chain_id_to_asym_id[chain_id1]
                asym_id2 = chain_id_to_asym_id[chain_id2]
                constraints.append(
                    Constraint(
                        (asym_id1, asym_id2),
                        (res_idx1, res_idx2),
                        (atom1, atom2),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                )
            else:
                raise ValueError(f"Unsupported constraint type: {constraint.type}")

        # Return RefStructure
        ref_struct = structure_preparation.prepare_structure(
            chains, connections, metadata
        )
        return ref_struct, constraints

    def resolve_structure_sources(
        self,
        ref_struct: RefStructure,
        input: query.Query,
        rng: np.random.Generator,
    ) -> ResolvedStructureSources:
        """Load custom sources and normalize them by physical ``asym_id``."""
        metadata_by_name = {chain.name: chain for chain in ref_struct.metadata.chains}
        apo_coords: dict[int, np.ndarray] = {}
        prior_groups: list[list[dict[int, np.ndarray]]] = []
        struct_token_records: list[dict] = []

        for sequence in input.sequences:
            if not isinstance(sequence, (query.ProteinSequence, query.RNASequence)):
                continue
            apo, priors, token_record = self._resolve_polymer_sources(
                sequence, metadata_by_name, input.name
            )
            apo_coords.update(apo)
            if priors:
                prior_groups.append(priors)
            if token_record is not None:
                struct_token_records.append(token_record)

        for sequence_group in input.multimer_sequences:
            apo, priors, token_records = self._resolve_multimer_sources(
                sequence_group, metadata_by_name, input.name
            )
            apo_coords.update(apo)
            prior_groups.append(priors)
            struct_token_records.extend(token_records)

        # DNA uses NaN apo coordinates. The tokenization pipeline derives its
        # apo mask, while PriorSampler fills missing prior coordinates heuristically.
        for chain in ref_struct.chains:
            if chain.is_polymer and chain.asym_id not in apo_coords:
                apo_coords[chain.asym_id] = self._empty_polymer_source(chain)

        prior_sources = self._align_prior_sources(ref_struct, prior_groups, rng)
        return ResolvedStructureSources(
            apo_coords=apo_coords,
            prior_sources=prior_sources,
            struct_token_records=struct_token_records,
        )

    def _resolve_polymer_sources(
        self,
        sequence: query.ProteinSequence | query.RNASequence,
        metadata_by_name: dict[str, ChainInfo],
        entry_id: str,
    ) -> tuple[dict[int, np.ndarray], list[dict[int, np.ndarray]], dict | None]:
        label = f"{entry_id}:{','.join(sequence.ids)}"
        apo_path, prior_paths = self._resolve_source_paths(
            sequence.apo, sequence.prior, label
        )

        apo_source = self._read_polymer_source(
            apo_path, sequence.sequence, sequence.ctype, label
        )
        asym_ids = [metadata_by_name[name].asym_id for name in sequence.ids]
        apo_coords = {asym_id: apo_source.target_coords.copy() for asym_id in asym_ids}

        prior_sources: list[dict[int, np.ndarray]] = []
        for prior_path in prior_paths:
            prior_source = self._read_polymer_source(
                prior_path, sequence.sequence, sequence.ctype, label
            )
            prior_sources.append(
                {asym_id: prior_source.target_coords.copy() for asym_id in asym_ids}
            )

        token_record = None
        if isinstance(sequence, query.ProteinSequence):
            token_record = {
                "seq": apo_source.sequence,
                "coords": apo_source.coords,
                "chains": [
                    (asym_id, *mapping)
                    for asym_id in asym_ids
                    for mapping in apo_source.mappings
                ],
            }
        return apo_coords, prior_sources, token_record

    def _resolve_multimer_sources(
        self,
        sequence_group: query.ProteinMultimerSequence,
        metadata_by_name: dict[str, ChainInfo],
        entry_id: str,
    ) -> tuple[dict[int, np.ndarray], list[dict[int, np.ndarray]], list[dict]]:
        physical_ids = [chain_id for pair in sequence_group.ids for chain_id in pair]
        label = f"{entry_id}:{','.join(physical_ids)}"
        apo_path, prior_paths = self._resolve_source_paths(
            sequence_group.apo, sequence_group.prior, label
        )

        apo_components = self._read_multimer_source(
            apo_path,
            (sequence_group.sequence1, sequence_group.sequence2),
            label,
        )
        apo_coords: dict[int, np.ndarray] = {}
        component_asym_ids: tuple[list[int], list[int]] = ([], [])
        for id_pair in sequence_group.ids:
            for component_i, chain_name in enumerate(id_pair):
                asym_id = metadata_by_name[chain_name].asym_id
                component_asym_ids[component_i].append(asym_id)
                apo_coords[asym_id] = apo_components[component_i].target_coords.copy()

        prior_sources: list[dict[int, np.ndarray]] = []
        for prior_path in prior_paths:
            prior_components = self._read_multimer_source(
                prior_path,
                (sequence_group.sequence1, sequence_group.sequence2),
                label,
            )
            source: dict[int, np.ndarray] = {}
            for component_i, asym_ids in enumerate(component_asym_ids):
                coords = prior_components[component_i].target_coords
                source.update({asym_id: coords.copy() for asym_id in asym_ids})
            prior_sources.append(source)

        token_records = []
        for component_i, asym_ids in enumerate(component_asym_ids):
            component = apo_components[component_i]
            token_records.append(
                {
                    "seq": component.sequence,
                    "coords": component.coords,
                    "chains": [
                        (asym_id, *mapping)
                        for asym_id in asym_ids
                        for mapping in component.mappings
                    ],
                }
            )
        return apo_coords, prior_sources, token_records

    def _resolve_source_paths(
        self,
        apo_path: str | None,
        prior_paths: list[str] | None,
        label: str,
    ) -> tuple[str, list[str]]:
        priors = list(prior_paths or [])
        if apo_path is None and priors:
            apo_path = priors[0]
            self.logger.warning(
                "No custom apo was provided for %s; using the first prior as apo: %s",
                label,
                apo_path,
            )
        if apo_path is None:
            raise ValueError(
                f"No apo/prior source was provided for {label}. "
                "The external apo sampler is not integrated yet."
            )
        return apo_path, priors

    def _read_polymer_source(
        self,
        path: str,
        expected_sequence: str,
        chain_type: C.ChainType,
        label: str,
    ) -> _AlignedStructureSource:
        if chain_type.is_protein:
            sequence, coords = read_protein_structure(path)
            atom_width = 37
        elif chain_type.is_rna:
            sequence, coords = read_rna_structure(path)
            atom_width = 29
        else:
            raise ValueError(f"Unsupported custom polymer source type: {chain_type}")

        return self._align_structure_source(
            path,
            label,
            expected_sequence,
            sequence,
            coords,
            atom_width,
        )

    def _align_structure_source(
        self,
        path: str,
        label: str,
        expected_sequence: str,
        source_sequence: str,
        source_coords: np.ndarray,
        atom_width: int,
    ) -> _AlignedStructureSource:
        source_shape = (len(source_sequence), atom_width, 3)
        if source_coords.shape != source_shape:
            raise ValueError(
                f"Structure coordinates for {label} in {path} have shape "
                f"{source_coords.shape}; expected {source_shape} for the parsed "
                "structure sequence."
            )

        source_coords = source_coords.astype(np.float32, copy=True)
        if source_sequence == expected_sequence:
            mappings = [(0, len(expected_sequence), 0, len(source_sequence))]
            return _AlignedStructureSource(
                sequence=source_sequence,
                coords=source_coords,
                target_coords=source_coords.copy(),
                mappings=mappings,
            )

        mappings, num_matches = _best_sequence_mapping(expected_sequence, source_sequence)
        if num_matches == 0:
            raise ValueError(
                f"Structure sequence mismatch for {label} in {path}: no matching "
                f"residues between expected {expected_sequence} and got "
                f"{source_sequence}."
            )

        target_coords = np.full(
            (len(expected_sequence), atom_width, 3), np.nan, dtype=np.float32
        )
        for target_start, target_end, source_start, source_end in mappings:
            target_coords[target_start:target_end] = source_coords[
                source_start:source_end
            ]

        num_mapped = sum(
            target_end - target_start for target_start, target_end, _, _ in mappings
        )
        self.logger.warning(
            "Aligned structure sequence for %s in %s: query length %d, source "
            "length %d, %d/%d mapped residues match, %d substitutions, %d "
            "query residues without coordinates, and %d ignored source residues.",
            label,
            path,
            len(expected_sequence),
            len(source_sequence),
            num_matches,
            num_mapped,
            num_mapped - num_matches,
            len(expected_sequence) - num_mapped,
            len(source_sequence) - num_mapped,
        )
        return _AlignedStructureSource(
            sequence=source_sequence,
            coords=source_coords,
            target_coords=target_coords,
            mappings=mappings,
        )

    def _read_multimer_source(
        self,
        path: str,
        expected_sequences: tuple[str, str],
        label: str,
    ) -> tuple[_AlignedStructureSource, _AlignedStructureSource]:
        chain_records = list(read_protein_multimer_structure(pathlib.Path(path)).values())
        if len(chain_records) != 2:
            raise ValueError(
                f"Protein multimer source for {label} in {path} must contain "
                f"exactly two non-empty protein chains, found {len(chain_records)}."
            )

        components: list[_AlignedStructureSource] = []
        for component_i, (record, expected_sequence) in enumerate(
            zip(chain_records, expected_sequences, strict=True), start=1
        ):
            sequence = record["seq"]
            coords = record["coords"]
            component_label = f"Protein multimer component {component_i} for {label}"
            components.append(
                self._align_structure_source(
                    path,
                    component_label,
                    expected_sequence,
                    sequence,
                    coords,
                    37,
                )
            )
        return components[0], components[1]

    def _align_prior_sources(
        self,
        ref_struct: RefStructure,
        prior_groups: list[list[dict[int, np.ndarray]]],
        rng: np.random.Generator,
    ) -> list[dict[int, np.ndarray]]:
        """Shuffle logical prior groups and cycle them to a global ensemble."""
        num_priors = (
            max(len(group) for group in prior_groups)
            if prior_groups
            else self.num_prior_samples
        )
        global_sources: list[dict[int, np.ndarray]] = [{} for _ in range(num_priors)]

        for group in prior_groups:
            order = rng.permutation(len(group))
            for prior_i, global_source in enumerate(global_sources):
                source = group[int(order[prior_i % len(order)])]
                overlap = global_source.keys() & source.keys()
                if overlap:
                    raise ValueError(
                        f"Multiple prior sources target asym_id(s): {sorted(overlap)}"
                    )
                global_source.update(
                    {asym_id: coords.copy() for asym_id, coords in source.items()}
                )

        for global_source in global_sources:
            for chain in ref_struct.chains:
                if chain.asym_id in global_source:
                    continue
                if chain.is_polymer:
                    global_source[chain.asym_id] = self._empty_polymer_source(chain)
                else:
                    global_source[chain.asym_id] = self._get_ligand_prior_source(
                        chain, rng
                    )
        return global_sources

    @staticmethod
    def _empty_polymer_source(chain: Chain) -> np.ndarray:
        atom_width = 37 if chain.is_protein else 29
        return np.full((chain.num_residues, atom_width, 3), np.nan, dtype=np.float32)

    def _get_ligand_prior_source(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate one ligand conformer in residue-major prior-source format."""
        assert chain.is_ligand
        if chain.smiles is not None:
            ref_comp = Component.from_smiles("LIG", chain.smiles)
            coords = ref_comp.get_ref_conformer(rng, train=False)
        else:
            coords = np.full_like(chain.atom.coords, np.nan)
            ccd_sequence = chain.get_ccd_sequence()
            for res_i, code in enumerate(ccd_sequence):
                if code not in self.ccd:
                    self.logger.warning(
                        f"CCD code {code} not found for ligand chain "
                        f"{chain.asym_id}. Filling with NaN coordinates."
                    )
                    continue

                ref_comp = self.ccd[code]
                ref_pos = ref_comp.get_ref_conformer(rng, train=False)
                ref_atom_order = ref_comp.get_atom_index_map()
                src_atom_indices: list[int] = []
                dst_atom_indices: list[int] = []
                res_idx = res_i + 1
                for atom_i in chain.residue.iter_residue_atoms(res_idx):
                    atom_name = chain.atom.name[atom_i]
                    if atom_name in ref_atom_order:
                        src_atom_indices.append(ref_atom_order[atom_name])
                        dst_atom_indices.append(atom_i)
                coords[dst_atom_indices] = ref_pos[src_atom_indices]

        return np.expand_dims(coords, axis=1).astype(np.float32)

    def prepare_struct_tok_input(
        self,
        f_input: FoldingInput,
        struct_token_records: list[dict],
    ) -> dict[int, dict]:
        """Add target sequence slices to raw apo structure-token records."""
        struct_tok_input: dict[int, dict] = {}
        for record_i, record in enumerate(struct_token_records):
            mappings: list[tuple[int, int, int, int]] = []
            for (
                asym_id,
                target_offset_st,
                target_offset_end,
                source_st,
                source_end,
            ) in record["chains"]:
                indices = torch.where(f_input.sequence.asym_id == asym_id)[0]
                if len(indices) == 0:
                    raise ValueError(f"No sequence indices found for asym_id {asym_id}.")

                st, end = int(indices[0]), int(indices[-1]) + 1
                expected_indices = torch.arange(st, end, device=indices.device)
                if not torch.equal(indices, expected_indices):
                    raise ValueError(
                        f"Sequence indices for asym_id {asym_id} are not contiguous."
                    )

                # sequence.asym_id includes BOS and EOS. The target slice only
                # covers residue tokens, matching the raw structure-token length.
                residue_st, residue_end = st + 1, end - 1
                target_st = residue_st + target_offset_st
                target_end = residue_st + target_offset_end
                if not residue_st <= target_st <= target_end <= residue_end:
                    raise ValueError(
                        f"Structure-token target mapping for asym_id {asym_id} is "
                        f"outside its residue range: {target_st}:{target_end} not in "
                        f"{residue_st}:{residue_end}."
                    )
                if target_end - target_st != source_end - source_st:
                    raise ValueError(
                        f"Structure-token mapping length mismatch for asym_id "
                        f"{asym_id}: target {target_st}:{target_end}, "
                        f"source {source_st}:{source_end}."
                    )
                mappings.append((target_st, target_end, source_st, source_end))

            struct_tok_input[record_i] = {
                "seq": record["seq"],
                "coords": torch.as_tensor(record["coords"], dtype=torch.float32),
                "mappings": mappings,
            }
        return struct_tok_input

    # ================================================================================
    # Chain Parsing Functions
    # ================================================================================
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
