"""Extract TPD protein/RNA sequences for apo sampling."""

import argparse
import pathlib

import pandas as pd

DATASET_NAME = "tpd"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The tpd/ folder is appended.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FASTA files.",
    )
    return parser.parse_args()


def iter_sequence_columns(df: pd.DataFrame, prefix: str):
    for col in sorted(c for c in df.columns if c.startswith(prefix)):
        for row in df[["entry_id", col]].itertuples(index=False):
            seq = getattr(row, col)
            if pd.isna(seq):
                continue
            seq = str(seq).strip().upper()
            if not seq:
                continue
            yield row.entry_id, col, seq


def write_fasta(records: list[tuple[str, str]], path: pathlib.Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    with path.open("w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def main():
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    metadata_path = data_dir / "metadata" / "filtered_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Prepare TPD cif/ and metadata first."
        )
    df = pd.read_csv(metadata_path)

    out_dir = data_dir / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[tuple[str, str]] = []
    unique_by_type: dict[str, dict[str, str]] = {"protein": {}, "rna": {}}
    mapping_rows: list[dict] = []

    for chain_type, prefix in (("protein", "protein_"), ("rna", "rna_")):
        for entry_id, chain_name, seq in iter_sequence_columns(df, prefix):
            record_id = f"{entry_id}|{chain_name}"
            all_records.append((record_id, seq))
            if seq not in unique_by_type[chain_type]:
                unique_id = f"uniq_{chain_type}_{len(unique_by_type[chain_type]) + 1}"
                unique_by_type[chain_type][seq] = unique_id
            mapping_rows.append(
                {
                    "entry_id": entry_id,
                    "chain_name": chain_name,
                    "chain_type": chain_type,
                    "unique_id": unique_by_type[chain_type][seq],
                    "sequence": seq,
                }
            )

    write_fasta(all_records, out_dir / "all_sequences.fasta", args.overwrite)
    for chain_type, seq_to_id in unique_by_type.items():
        records = [(unique_id, seq) for seq, unique_id in seq_to_id.items()]
        write_fasta(
            records,
            out_dir / f"unique_{chain_type}_sequences.fasta",
            args.overwrite,
        )

    mapping_path = out_dir / "sequence_mapping.tsv"
    if mapping_path.exists() and not args.overwrite:
        raise FileExistsError(f"{mapping_path} exists. Use --overwrite.")
    pd.DataFrame(mapping_rows).to_csv(mapping_path, sep="\t", index=False)

    print(f"All chain sequences: {len(all_records)}")
    print(f"Unique protein sequences: {len(unique_by_type['protein'])}")
    print(f"Unique RNA sequences: {len(unique_by_type['rna'])}")
    print(f"Wrote FASTA files to {out_dir}")


if __name__ == "__main__":
    main()
