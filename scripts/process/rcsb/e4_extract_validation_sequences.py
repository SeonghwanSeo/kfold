"""Filter validation sequences to validation_ids.txt entries.

The initial validation sequence extraction is date-range based.  This script
narrows those FASTA files to the final validation IDs and writes polymer FASTA
records with per-entity IDs (`<pdb_id>_<entity_id>`) instead of `uniq_*` IDs.
"""

import argparse
import pathlib
from collections import Counter

import kfold.constants as C
from kfold.data.utils.io.fasta import read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validation-ID-filtered sequence FASTA files."
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to processed dataset root.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="RCSB split containing validation_ids.txt and sequences/.",
    )
    parser.add_argument(
        "--ids_file",
        type=pathlib.Path,
        default=None,
        help="Optional validation IDs file. Defaults to rcsb-*/validation_ids.txt.",
    )
    parser.add_argument(
        "--source_all_sequences",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional source all_sequences FASTA. Defaults to "
            "rcsb-*/sequences/all_sequences.fasta."
        ),
    )
    return parser.parse_args()


def parse_sequence_key(seq_id: str) -> tuple[str, str, C.ChainType]:
    pdb_id, entity_id, chain_type = seq_id.split("|")
    return pdb_id.lower(), entity_id, C.ChainType[chain_type.upper()]


def load_entry_ids(path: pathlib.Path) -> set[str]:
    with path.open() as f:
        entry_ids = {line.strip().lower() for line in f if line.strip()}
    if len(entry_ids) == 0:
        raise ValueError(f"No entry IDs found in {path}")
    return entry_ids


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    seq_dir = data_dir / "sequences"
    ids_file = args.ids_file or data_dir / "validation_ids.txt"
    source_all_sequences = args.source_all_sequences or seq_dir / "all_sequences.fasta"

    entry_ids = load_entry_ids(ids_file)
    all_records = read_fasta(source_all_sequences)

    selected: list[tuple[str, str, C.ChainType, str]] = []
    for seq_id, sequence in all_records:
        pdb_id, entity_id, chain_type = parse_sequence_key(seq_id)
        if pdb_id not in entry_ids:
            continue
        selected.append((pdb_id, entity_id, chain_type, sequence))

    selected_entry_ids = {pdb_id for pdb_id, _, _, _ in selected}
    missing_entry_ids = entry_ids - selected_entry_ids
    if missing_entry_ids:
        examples = ", ".join(sorted(missing_entry_ids)[:10])
        raise ValueError(
            f"{len(missing_entry_ids)} validation IDs were not found in "
            f"{source_all_sequences}: {examples}"
        )

    selected.sort(key=lambda x: (x[0], int(x[1]), x[2].value, x[3]))
    seq_dir.mkdir(parents=True, exist_ok=True)

    all_sequences = [
        (f"{pdb_id}|{entity_id}|{chain_type.name.lower()}", sequence)
        for pdb_id, entity_id, chain_type, sequence in selected
    ]
    write_fasta(all_sequences, seq_dir / "all_sequences.fasta")

    polymer_outputs = {
        C.ChainType.PROTEIN: seq_dir / "unique_protein_sequences.fasta",
    }
    counts = Counter()
    for chain_type, output_path in polymer_outputs.items():
        records = [
            (f"{pdb_id}_{entity_id}", sequence)
            for pdb_id, entity_id, ctype, sequence in selected
            if ctype is chain_type
        ]
        write_fasta(records, output_path)
        counts[chain_type.name.lower()] = len(records)

    ligand_count = sum(ctype.is_ligand for _, _, ctype, _ in selected)
    print(f"Validation IDs: {len(entry_ids)}")
    print(f"Selected sequence records: {len(selected)}")
    print(f"Proteins: {counts['protein']}")
    print(f"Ligands: {ligand_count}")
    print(f"Wrote sequences to {seq_dir}")


if __name__ == "__main__":
    main()
