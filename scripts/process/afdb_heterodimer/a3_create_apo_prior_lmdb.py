"""Create AFDB heterodimer apo lookup, apo LMDBs, and prior LMDBs."""

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

import msgpack

sys.path.append("./")

from scripts.process.rcsb.f2_create_apo_prior_lmdb import (
    ApoTask,
    RankedRecord,
    collect_protein_source,
    ensure_overwrite,
    make_prior_tasks,
    summarize_ranked,
    write_apo_lmdbs,
    write_lookup,
    write_prior_lmdbs,
)

DATASET_NAME = "AFDB-heterodimer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help=f"Root preprocessed data directory. {DATASET_NAME}/ is appended.",
    )
    parser.add_argument(
        "--sampler_dir",
        default="apo/protein",
        help=f"Apo source directory under {DATASET_NAME}/.",
    )
    parser.add_argument(
        "--manifest_name",
        default="manifest.msgpack",
        help="Manifest used to enumerate apo/prior entries.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Number of structure parsing workers.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=128,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of low-ptm and long ranked structures to report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lookup/LMDB outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Collect inputs and print summary without writing outputs.",
    )
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> dict[str, dict]:
    with path.open("rb") as f:
        manifest = msgpack.unpack(f, raw=False)
    return {record["id"]: record for record in manifest}


def load_entity_sequences(
    data_dir: pathlib.Path,
    manifest_name: str,
) -> dict[str, dict[str, str]]:
    metadata_by_entry = load_manifest(data_dir / manifest_name)
    mapping_path = data_dir / "sequences" / "sequence_mapping.tsv"

    entity_sequences: dict[str, dict[str, str]] = {}
    with mapping_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["chain_type"] != "protein":
                continue
            record = metadata_by_entry.get(row["entry_id"])
            if record is None:
                continue

            protein_chains = [
                chain for chain in record["chains"] if int(chain["type"]) == 0
            ]
            protein_chains.sort(key=lambda chain: int(chain["entity_id"]))
            chain_i = int(row["chain_name"].removeprefix("protein_"))
            chain = protein_chains[chain_i]
            entity_key = f"{row['entry_id']}_{int(chain['entity_id'])}"
            entity_sequences[entity_key] = {
                "entity_key": entity_key,
                "chain_type": "protein",
                "sequence": row["sequence"],
            }
    return entity_sequences


def iter_apo_source_dirs(raw_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    raw_root = data_dir / args.sampler_dir
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    entity_sequences_by_key = load_entity_sequences(data_dir, args.manifest_name)
    apo_lookup: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    apo_tasks: list[ApoTask] = []
    prior_samples_by_type = defaultdict(lambda: defaultdict(list))
    ranked_records: list[RankedRecord] = []
    stats = defaultdict(int)

    for source_dir in iter_apo_source_dirs(raw_root):
        part = collect_protein_source(
            source_dir,
            source_dir.name,
            entity_sequences_by_key,
            apo_lookup,
            apo_tasks,
            prior_samples_by_type["protein"],
            ranked_records,
        )
        for key, value in part.items():
            stats[f"{source_dir.name}:{key}"] += value

    prior_tasks = make_prior_tasks(prior_samples_by_type)
    apo_lookup = {entry: dict(chains) for entry, chains in apo_lookup.items()}
    ranked_summary = summarize_ranked(ranked_records, args.top_n)

    print("Collection summary:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  entity_sequences: {len(entity_sequences_by_key)}")
    print(f"  apo_lmdb_tasks: {len(apo_tasks)}")
    print(f"  prior_stack_tasks: {len(prior_tasks)}")
    for chain_type, samples in sorted(prior_samples_by_type.items()):
        print(f"  prior_{chain_type}_entities: {len(samples)}")
        print(f"  prior_{chain_type}_samples: {sum(len(v) for v in samples.values())}")
    print(f"  apo_lookup_entries: {len(apo_lookup)}")

    print(f"Lowest ranked ptm structures with length >= 100 (top {args.top_n}):")
    for record in ranked_summary["low_ptm_len_ge_100"]:
        print(
            f"  {record['entity_key']} {record['chain_type']} {record['source']} "
            f"len={record['length']} ptm={record['ptm']}"
        )
    print(f"Longest ranked structures (top {args.top_n}):")
    for record in ranked_summary["longest"]:
        print(
            f"  {record['entity_key']} {record['chain_type']} {record['source']} "
            f"len={record['length']} ptm={record['ptm']}"
        )

    if args.dry_run:
        return

    for output_path in (
        data_dir / "apo_lookup.json",
        data_dir / "apo_lookup.msgpack",
        data_dir / "apo_ranked_quality_report.json",
    ):
        ensure_overwrite(output_path, args.overwrite)

    write_lookup(data_dir / "apo_lookup.msgpack", apo_lookup)
    with (data_dir / "apo_ranked_quality_report.json").open("w") as f:
        json.dump(ranked_summary, f, indent=2)

    apo_result = write_apo_lmdbs(
        data_dir,
        apo_tasks,
        args.num_workers,
        args.map_size_gb,
        args.overwrite,
    )
    prior_result = write_prior_lmdbs(
        data_dir,
        prior_tasks,
        args.num_workers,
        args.map_size_gb,
        args.overwrite,
    )
    print(f"apo_lmdb: {apo_result}")
    print(f"prior_lmdb: {prior_result}")


if __name__ == "__main__":
    main()
