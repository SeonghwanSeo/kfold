"""Extract JASPAR protein/DNA sequences for preprocessing and apo sampling."""

import argparse
import pathlib
import re

import pandas as pd

DATASET_NAME = "JASPAR"
CHAIN_TYPES = (("protein", "protein_"), ("dna", "dna_"))
CHAIN_COLUMN_RE = re.compile(r"^(protein|dna)_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The JASPAR/ folder is appended.",
    )
    parser.add_argument(
        "--metadata_path",
        type=pathlib.Path,
        default=None,
        help="Metadata CSV path. Defaults to JASPAR/metadata.csv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all metadata rows instead of distillation == 1 rows.",
    )
    parser.add_argument(
        "--processed_only",
        action="store_true",
        help="Only use rows with an existing JASPAR/npz/{data_idx}.npz file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FASTA and mapping files.",
    )
    return parser.parse_args()


def load_metadata(path: pathlib.Path, distillation_only: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if distillation_only:
        if "distillation" not in df.columns:
            raise KeyError(f"{path} does not contain a distillation column.")
        df = df[df["distillation"] == 1]
    return df


def filter_processed_rows(df: pd.DataFrame, data_dir: pathlib.Path) -> pd.DataFrame:
    if "data_idx" not in df.columns:
        raise KeyError("metadata does not contain a data_idx column.")
    npz_dir = data_dir / "npz"
    if not npz_dir.exists():
        raise FileNotFoundError(npz_dir)
    keep = df["data_idx"].apply(lambda data_idx: (npz_dir / f"{data_idx}.npz").exists())
    return df[keep]


def iter_sequence_columns(df: pd.DataFrame, prefix: str):
    def column_index(col: str) -> int | None:
        match = CHAIN_COLUMN_RE.match(col)
        if match is None or not col.startswith(prefix):
            return None
        return int(match.group(2))

    columns = sorted(
        (col for col in df.columns if column_index(col) is not None),
        key=lambda col: column_index(col),
    )
    for col in columns:
        for row in df[["data_idx", col]].itertuples(index=False):
            seq = getattr(row, col)
            if pd.isna(seq):
                continue
            seq = str(seq).strip().upper()
            if not seq:
                continue
            yield row.data_idx, col, seq


def write_fasta(records: list[tuple[str, str]], path: pathlib.Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    metadata_path = args.metadata_path or (data_dir / "metadata.csv")
    df = load_metadata(metadata_path, distillation_only=not args.all)
    if args.processed_only:
        df = filter_processed_rows(df, data_dir)

    out_dir = data_dir / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[tuple[str, str]] = []
    unique_by_type: dict[str, dict[str, str]] = {
        chain_type: {} for chain_type, _ in CHAIN_TYPES
    }
    mapping_rows: list[dict] = []

    for chain_type, prefix in CHAIN_TYPES:
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
    write_fasta(all_records, out_dir / "sequence.fasta", args.overwrite)
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

    print(f"Metadata rows used: {len(df)}")
    print(f"All chain sequences: {len(all_records)}")
    for chain_type, seq_to_id in unique_by_type.items():
        print(f"Unique {chain_type} sequences: {len(seq_to_id)}")
    print(f"Wrote sequence files to {out_dir}")


if __name__ == "__main__":
    main()
