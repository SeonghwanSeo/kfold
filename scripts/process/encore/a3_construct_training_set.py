"""Pack ENCORE NPZs and write manifests plus unique-sequence FASTA files."""

import argparse
import csv
import json
import multiprocessing
import pathlib
import shutil

import lmdb
import msgpack
from tqdm import tqdm

from kfold.data.types.structure import RefStructure

DATASET_NAME = "ENCORE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--map_size_gb", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_npz(path: pathlib.Path) -> tuple[str, bytes, dict, list[dict]]:
    ref_struct = RefStructure.load_npz(path)
    metadata = ref_struct.metadata.to_dict()
    seen_entities: set[int] = set()
    sequence_rows: list[dict] = []
    type_counts = {"protein": 0, "rna": 0}
    for chain in ref_struct.chains:
        if chain.entity_id in seen_entities:
            continue
        if chain.ctype.is_protein:
            chain_type = "protein"
        elif chain.ctype.is_rna:
            chain_type = "rna"
        else:
            continue
        seen_entities.add(chain.entity_id)
        chain_name = f"{chain_type}_{type_counts[chain_type]}"
        type_counts[chain_type] += 1
        sequence_rows.append(
            {
                "entry_id": ref_struct.id,
                "chain_name": chain_name,
                "chain_type": chain_type,
                "entity_id": chain.entity_id,
                "sequence": chain.get_sequence(map_to_standard=True),
            }
        )
    return path.stem, path.read_bytes(), metadata, sequence_rows


def check_output(path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")


def write_fasta(records: list[tuple[str, str]], path: pathlib.Path) -> None:
    with path.open("w") as f:
        for name, sequence in records:
            f.write(f">{name}\n{sequence}\n")


def write_sequences(dataset_dir: pathlib.Path, rows: list[dict]) -> None:
    rows.sort(key=lambda row: (row["entry_id"], int(row["entity_id"])))
    unique_by_type: dict[str, dict[str, str]] = {"protein": {}, "rna": {}}
    for row in rows:
        mapping = unique_by_type[row["chain_type"]]
        sequence = row["sequence"]
        if sequence not in mapping:
            mapping[sequence] = f"uniq_{row['chain_type']}_{len(mapping) + 1}"
        row["unique_id"] = mapping[sequence]

    sequence_dir = dataset_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    all_records = [
        (f"{row['entry_id']}|{row['chain_name']}", row["sequence"]) for row in rows
    ]
    write_fasta(all_records, sequence_dir / "all_sequences.fasta")
    write_fasta(all_records, sequence_dir / "sequence.fasta")
    for chain_type, sequence_to_id in unique_by_type.items():
        write_fasta(
            [(unique_id, sequence) for sequence, unique_id in sequence_to_id.items()],
            sequence_dir / f"unique_{chain_type}_sequences.fasta",
        )

    columns = [
        "entry_id",
        "chain_name",
        "chain_type",
        "entity_id",
        "unique_id",
        "sequence",
    ]
    with (sequence_dir / "sequence_mapping.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Entity sequences: {len(rows)}")
    for chain_type, mapping in unique_by_type.items():
        print(f"Unique {chain_type} sequences: {len(mapping)}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.data_dir / DATASET_NAME
    npz_paths = sorted((dataset_dir / "npz").glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files under {dataset_dir / 'npz'}")
    print(f"Found NPZ files: {len(npz_paths)}")

    lmdb_path = dataset_dir / "structure.lmdb"
    manifest_json = dataset_dir / "manifest.json"
    manifest_msgpack = dataset_dir / "manifest.msgpack"
    for path in (lmdb_path, manifest_json, manifest_msgpack):
        check_output(path, args.overwrite)
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)

    env = lmdb.open(str(lmdb_path), map_size=args.map_size_gb * 1024**3)
    metadata_dicts: list[dict] = []
    sequence_rows: list[dict] = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(process_npz, npz_paths, chunksize=16)
        txn = env.begin(write=True)
        try:
            for i, (key, value, metadata, seq_rows) in enumerate(
                tqdm(results, total=len(npz_paths), desc="Writing ENCORE LMDB"),
                start=1,
            ):
                txn.put(key.encode(), value)
                metadata_dicts.append(metadata)
                sequence_rows.extend(seq_rows)
                if i % 500 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except Exception:
            txn.abort()
            raise
    env.sync()
    env.close()

    metadata_dicts.sort(key=lambda row: row["id"])
    with manifest_json.open("w") as f:
        json.dump(metadata_dicts, f, indent=2)
    with manifest_msgpack.open("wb") as f:
        msgpack.pack(metadata_dicts, f)
    write_sequences(dataset_dir, sequence_rows)
    print(f"LMDB/manifest entries: {len(metadata_dicts)}")


if __name__ == "__main__":
    main()
