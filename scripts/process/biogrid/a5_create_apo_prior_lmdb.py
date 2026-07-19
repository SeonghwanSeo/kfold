"""Create BioGRID apo lookup/LMDBs and entity-level prior stack LMDB."""

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.process.rcsb.f2_create_apo_prior_lmdb import (  # noqa: E402
    ApoTask,
    RankedRecord,
    collect_protein_source,
    ensure_overwrite,
    iter_source_dirs,
    make_prior_tasks,
    summarize_ranked,
    write_apo_lmdbs,
    write_lookup,
    write_prior_lmdbs,
)

DATASET_NAME = "Biogrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--sampler_dir", default="apo")
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--map_size_gb", type=int, default=128)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_entity_sequences(data_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    path = data_dir / "sequences" / "sequence_mapping.tsv"
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["chain_type"] != "protein":
                continue
            entity_key = f"{row['entry_id']}_{row['entity_id']}"
            result[entity_key] = {
                "entity_key": entity_key,
                "chain_type": "protein",
                "sequence": row["sequence"],
            }
    return result


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    raw_root = data_dir / args.sampler_dir
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    entity_sequences_by_key = load_entity_sequences(data_dir)
    apo_lookup: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    apo_tasks: list[ApoTask] = []
    prior_samples_by_type = defaultdict(lambda: defaultdict(list))
    ranked_records: list[RankedRecord] = []
    stats = defaultdict(int)

    for source_dir in iter_source_dirs(raw_root, "protein"):
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
    print(f"  apo_lookup_entries: {len(apo_lookup)}")
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
        data_dir, apo_tasks, args.num_workers, args.map_size_gb, args.overwrite
    )
    prior_result = write_prior_lmdbs(
        data_dir, prior_tasks, args.num_workers, args.map_size_gb, args.overwrite
    )
    print(f"apo_lmdb: {apo_result}")
    print(f"prior_lmdb: {prior_result}")


if __name__ == "__main__":
    main()
