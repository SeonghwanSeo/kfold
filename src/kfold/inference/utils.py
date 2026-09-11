"""PDB loading, sequence alignment, and apo structure-token encoding."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np

from kfold.data.utils.io.structure import _read_protein_chain, read_gemmi_structure
from kfold.inference.query import ProteinMultimerSequence, ProteinSequence, Query

if TYPE_CHECKING:
    from kfold.model.modules.prot_struct_encoder import StructureEncoder

logger = logging.getLogger(__name__)


def read_pdbs(paths: list[str]) -> tuple[list[str], list[gemmi.Structure]]:
    """Return matching lists of single-model PDB text and original structures."""
    pdbs, structures = [], []
    for source in paths:
        path = Path(source).resolve()
        structure = read_gemmi_structure(path)
        if not len(structure) or any(not len(model) for model in structure):
            raise ValueError(f"Structure file contains an empty model: {path}")
        for model in structure:
            single = gemmi.Structure()
            single.add_model(model.clone())
            pdbs.append(single.make_pdb_string())
            structures.append(single)
    return pdbs, structures


def _best_sequence_mapping(
    target_sequence: str,
    source_sequence: str,
) -> tuple[tuple[int, int, int, int], int]:
    """Return the ungapped overlap with the most matching residues."""
    target_length = len(target_sequence)
    source_length = len(source_sequence)
    best_mapping = (0, 0, 0, 0)
    best_score = (-1, -1)

    for offset in range(1 - target_length, source_length):
        target_start = max(0, -offset)
        source_start = max(0, offset)
        overlap = min(
            target_length - target_start,
            source_length - source_start,
        )
        num_matches = sum(
            target_sequence[target_start + i] == source_sequence[source_start + i]
            for i in range(overlap)
        )
        score = (num_matches, overlap)
        if score > best_score:
            best_score = score
            best_mapping = (
                target_start,
                target_start + overlap,
                source_start,
                source_start + overlap,
            )

    return best_mapping, best_score[0]


def resolve_structure_chains(query: Query, structures: list) -> list:
    """Resolve source chains in entry -> apo/prior -> model -> component order.

    Each component is (source sequence, source coordinates, target slice).
    Structures follow the same entry and model order as the query's PDB lists.
    """

    def resolve_chain(model, component_i, target_sequence):
        chain = model.subchains()[0] if component_i is None else model[component_i]
        sequence, coords = _read_protein_chain(chain)
        if sequence == target_sequence:
            start, end, source_start, source_end = 0, len(sequence), 0, len(sequence)
        else:
            (start, end, source_start, source_end), matches = _best_sequence_mapping(
                target_sequence, sequence
            )
            if not matches:
                raise ValueError(
                    f"No matching residues for {query.name} in model {model.num}."
                )
            logger.info(
                "Mapped %s model %s: target %s:%s, source %s:%s",
                query.name,
                model.num,
                start,
                end,
                source_start,
                source_end,
            )
        return (
            sequence[source_start:source_end],
            coords[source_start:source_end],
            slice(start, end),
        )

    chains = []
    entries = [seq for seq in query.sequences if isinstance(seq, ProteinSequence)]
    entries += query.multimer_sequences
    for entry, ensembles in zip(entries, structures, strict=True):
        multimer = isinstance(entry, ProteinMultimerSequence)
        sequences = [entry.sequence1, entry.sequence2] if multimer else [entry.sequence]
        entry_chains = []
        for ensemble in ensembles:
            model_chains = []
            for structure in ensemble:
                if len(structure) != 1 or not len(structure[0]):
                    raise ValueError(f"Expected one nonempty model for {query.name}.")
                model = structure[0]
                if multimer and (
                    len(model) != 2 or any(not len(chain) for chain in model)
                ):
                    raise ValueError("Expected two nonempty multimer chains.")
                model_chains.append(
                    [
                        resolve_chain(model, i if multimer else None, sequence)
                        for i, sequence in enumerate(sequences)
                    ]
                )
            entry_chains.append(model_chains)
        chains.append(entry_chains)
    return chains


def encode_apo_tokens(chains: list, encoder: "StructureEncoder") -> list:
    """Return CPU tokens in entry -> apo model -> component order."""
    tokens = []
    for apo_chains, _ in chains:
        entry_tokens = []
        for components in apo_chains:
            model_tokens = []
            for sequence, coords, _ in components:
                encoded = encoder.tokenize(sequence, coords)
                model_tokens.append(
                    {
                        name: encoded[name].detach().cpu().numpy().copy()
                        for name in ("bb_token_id", "fa_token_id")
                    }
                )
            entry_tokens.append(model_tokens)
        tokens.append(entry_tokens)
    return tokens


def align_structures(query: Query, chains: list, tokens: list | None) -> None:
    """Place coordinates and precomputed tokens on query residues, using list order."""
    entries = [seq for seq in query.sequences if isinstance(seq, ProteinSequence)]
    entries += query.multimer_sequences
    for entry_i, (entry, ensembles) in enumerate(zip(entries, chains, strict=True)):
        sequences = (
            [entry.sequence1, entry.sequence2]
            if isinstance(entry, ProteinMultimerSequence)
            else [entry.sequence]
        )
        entry._apo_token = [[] for _ in sequences] if tokens is not None else None
        for kind, model_chains in zip(("apo", "prior"), ensembles, strict=True):
            component_coords = [[] for _ in sequences]
            for model_i, components in enumerate(model_chains):
                for component_i, (sequence, source) in enumerate(
                    zip(sequences, components, strict=True)
                ):
                    _, coords, target = source
                    aligned = np.full((len(sequence), 37, 3), np.nan, dtype=np.float32)
                    aligned[target] = coords
                    component_coords[component_i].append(aligned)
                    if kind == "prior" or tokens is None:
                        continue
                    placed = {}
                    for name, values in tokens[entry_i][model_i][component_i].items():
                        placed[name] = np.full(len(sequence), -1, dtype=np.int64)
                        placed[name][target] = values
                    entry._apo_token[component_i].append(placed)
            setattr(entry, f"_{kind}_coords", [np.stack(c) for c in component_coords])
