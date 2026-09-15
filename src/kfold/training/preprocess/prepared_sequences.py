"""Export sequences with the final entity IDs from prepared RefStructures."""

from pathlib import Path

from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import write_fasta


def structure_sequences(structure: RefStructure) -> list[tuple[str, str]]:
    entities = {}
    for chain in structure.chains:
        if chain.entity_id in entities:
            continue
        sequence = (
            chain.get_sequence(map_to_standard=True)
            if chain.ctype.is_polymer
            else "-".join(chain.get_ccd_sequence())
        )
        header = f"{structure.id}|{chain.entity_id}|{chain.ctype.name.lower()}"
        entities[chain.entity_id] = (header, sequence)
    return [entities[eid] for eid in sorted(entities)]


def save_sequences(sequences: list[tuple[str, str]], dataset_dir: Path):
    sequences = sorted(sequences)
    directory = dataset_dir / "sequences"
    directory.mkdir(parents=True, exist_ok=True)
    write_fasta(sequences, directory / "all_sequences.fasta")
    unique = {}
    for header, sequence in sequences:
        if header.rsplit("|", 1)[1] == "protein":
            unique.setdefault(sequence, header)
    write_fasta(
        [(header, seq) for seq, header in unique.items()],
        directory / "unique_protein_sequences.fasta",
    )
