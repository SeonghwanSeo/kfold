"""Materialize disordered PDB apo structures from RCSB train apo archives.

This reuses already generated RCSB train apo structures for disordered PDB
entries that also appear in RCSB train.  The script first matches by entry id,
then by exact polymer entity sequence.  Non-identical partial or longest-block
matches are intentionally ignored.
"""

import argparse
import json
import pathlib
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass

import lmdb
import msgpack

from kfold.data.utils.io.fasta import read_fasta

DEFAULT_SOURCES = (
    "prot_sampler_seed1_step20",
    "prot_sampler_seed2_step50",
    "esmfold",
    "rna_sampler_seed1_step100",
    "dna_helix",
)

POLYMER_CHAIN_TYPES = ("protein", "dna", "rna")


@dataclass(frozen=True)
class SequenceMatch:
    rcsb_entity_id: int
    match_type: str
    coverage: float
    residue_map: str | None


@dataclass(frozen=True)
class FetchRecord:
    entry_id: str
    entity_id: int
    rcsb_entity_id: int
    chain_type: str
    source: str
    source_name: str
    output_name: str
    source_path: str
    output_path: str
    match_type: str
    coverage: float
    residue_map: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory containing disordered_pdb and rcsb-train.",
    )
    parser.add_argument(
        "--rcsb_dir",
        type=pathlib.Path,
        default=None,
        help="Path to rcsb-train directory. Defaults to {data_dir}/rcsb-train.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated RCSB apo sources to reuse. afdb is intentionally excluded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied apo files and mapping outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Collect matches and print summary without copying files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of disordered polymer entities to inspect.",
    )
    return parser.parse_args()


def normalize_sources(sources: str) -> tuple[str, ...]:
    return tuple(source.strip() for source in sources.split(",") if source.strip())


def load_polymer_sequences(path: pathlib.Path) -> dict[str, dict[str, dict[int, str]]]:
    sequences: dict[str, dict[str, dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for header, sequence in read_fasta(path):
        entry_id, entity_id_str, chain_type = header.split("|")
        chain_type = chain_type.lower()
        if chain_type not in POLYMER_CHAIN_TYPES:
            continue
        sequences[entry_id.lower()][chain_type][int(entity_id_str)] = sequence
    return {
        entry_id: dict(chains_by_type) for entry_id, chains_by_type in sequences.items()
    }


def load_apo_lookup(path: pathlib.Path) -> dict[str, dict[int, list[dict]]]:
    with path.open("rb") as f:
        raw_lookup = msgpack.unpack(f, raw=False, strict_map_key=False)

    lookup: dict[str, dict[int, list[dict]]] = {}
    for entry_id, entry_lookup in raw_lookup.items():
        chains = entry_lookup.get("chains", entry_lookup)
        lookup[entry_id.lower()] = {
            int(entity_id): list(records) for entity_id, records in chains.items()
        }
    return lookup


def match_sequence(
    target_seq: str,
    apo_seq: str,
    *,
    rcsb_entity_id: int,
) -> SequenceMatch | None:
    """Return an exact sequence match, or None."""
    if target_seq != apo_seq:
        return None
    return SequenceMatch(
        rcsb_entity_id=rcsb_entity_id,
        match_type="exact",
        coverage=1.0,
        residue_map=None,
    )


def choose_sequence_match(
    target_seq: str,
    rcsb_candidates: dict[int, str],
    *,
    apo_records_by_entity: dict[int, list[dict]],
    selected_sources: set[str],
) -> SequenceMatch | None:
    scored: list[tuple[int, SequenceMatch]] = []
    for rcsb_entity_id, apo_seq in rcsb_candidates.items():
        match = match_sequence(
            target_seq,
            apo_seq,
            rcsb_entity_id=rcsb_entity_id,
        )
        if match is None:
            continue
        source_count = sum(
            record.get("source") in selected_sources
            for record in apo_records_by_entity.get(rcsb_entity_id, [])
        )
        if source_count == 0:
            continue
        scored.append((source_count, match))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def first_existing(paths: list[pathlib.Path]) -> pathlib.Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def structure_suffix(path: pathlib.Path) -> str:
    for suffix in (".pdb.zst", ".cif.zst", ".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        if path.name.endswith(suffix):
            return suffix
    return path.suffix


def find_source_structure(
    rcsb_dir: pathlib.Path,
    chain_type: str,
    source: str,
    source_name: str,
) -> pathlib.Path | None:
    source_root = rcsb_dir / "apo" / chain_type / source

    if chain_type == "dna":
        return first_existing(
            [
                source_root / f"{source_name}.pdb.zst",
                source_root / f"{source_name}.pdb",
                source_root / f"{source_name}.cif.zst",
                source_root / f"{source_name}.cif",
            ]
        )

    source_dir = source_root / source_name
    if not source_dir.exists():
        return None

    if chain_type == "protein" and source == "esmfold":
        candidates = [
            source_dir / "esmfold2.pdb",
            source_dir / "esmfold.pdb",
            source_dir / f"{source_name}.pdb",
            source_dir / f"{source_name}.pdb.zst",
        ]
    else:
        candidates = [
            source_dir / f"{source_name}_ranked_model.pdb.zst",
            source_dir / f"{source_name}_ranked.pdb.zst",
            source_dir / f"{source_name}_ranked_model.cif.zst",
            source_dir / f"{source_name}_ranked.cif.zst",
            source_dir / f"{source_name}_ranked_model.pdb",
            source_dir / f"{source_name}_ranked.pdb",
            source_dir / f"{source_name}_ranked_model.cif",
            source_dir / f"{source_name}_ranked.cif",
        ]
    found = first_existing(candidates)
    if found is not None:
        return found

    for pattern in (
        "*ranked_model*.pdb*",
        "*ranked*.pdb*",
        "*ranked_model*.cif*",
        "*ranked*.cif*",
    ):
        matches = sorted(source_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def collect_fetch_records(
    *,
    disordered_root: pathlib.Path,
    rcsb_dir: pathlib.Path,
    sources: tuple[str, ...],
    limit: int | None,
) -> tuple[list[FetchRecord], dict[str, int], dict[str, dict[str, list[dict]]]]:
    selected_sources = set(sources)
    disordered_sequences = load_polymer_sequences(
        disordered_root / "sequences" / "all_sequences.fasta"
    )
    rcsb_sequences = load_polymer_sequences(
        rcsb_dir / "sequences" / "all_sequences.fasta"
    )
    rcsb_apo_lookup = load_apo_lookup(rcsb_dir / "apo_lookup.msgpack")

    records: list[FetchRecord] = []
    mapping: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    stats: dict[str, int] = defaultdict(int)
    inspected = 0

    for entry_id, sequences_by_type in sorted(disordered_sequences.items()):
        rcsb_entry_sequences_by_type = rcsb_sequences.get(entry_id)
        rcsb_entry_apo_lookup = rcsb_apo_lookup.get(entry_id, {})
        if rcsb_entry_sequences_by_type is None:
            stats["missing_entry_id"] += sum(
                len(entity_sequences) for entity_sequences in sequences_by_type.values()
            )
            continue

        for chain_type in POLYMER_CHAIN_TYPES:
            entity_sequences = sequences_by_type.get(chain_type, {})
            rcsb_entry_sequences = rcsb_entry_sequences_by_type.get(chain_type, {})
            for entity_id, target_seq in sorted(entity_sequences.items()):
                if limit is not None and inspected >= limit:
                    break
                inspected += 1
                stats["polymer_entities"] += 1
                stats[f"{chain_type}_entities"] += 1

                match = choose_sequence_match(
                    target_seq,
                    rcsb_entry_sequences,
                    apo_records_by_entity=rcsb_entry_apo_lookup,
                    selected_sources=selected_sources,
                )
                if match is None:
                    stats[f"no_sequence_match_with_apo:{chain_type}"] += 1
                    continue

                output_name = f"{entry_id}_{entity_id}"
                lookup_records: list[dict] = []
                seen_sources: set[str] = set()
                for apo_record in rcsb_entry_apo_lookup.get(match.rcsb_entity_id, []):
                    source = apo_record.get("source")
                    record_chain_type = apo_record.get("chain_type", "protein")
                    if record_chain_type != chain_type:
                        continue
                    if source not in selected_sources or source in seen_sources:
                        continue
                    seen_sources.add(source)
                    source_name = apo_record["name"]
                    source_path = find_source_structure(
                        rcsb_dir, chain_type, source, source_name
                    )
                    if source_path is None:
                        stats[f"missing_file:{chain_type}:{source}"] += 1
                        continue
                    output_path = (
                        disordered_root
                        / "apo"
                        / chain_type
                        / source
                        / f"{output_name}{structure_suffix(source_path)}"
                    )
                    records.append(
                        FetchRecord(
                            entry_id=entry_id,
                            entity_id=entity_id,
                            rcsb_entity_id=match.rcsb_entity_id,
                            chain_type=chain_type,
                            source=source,
                            source_name=source_name,
                            output_name=output_name,
                            source_path=str(source_path),
                            output_path=str(output_path),
                            match_type=match.match_type,
                            coverage=match.coverage,
                            residue_map=match.residue_map,
                        )
                    )
                    lookup_records.append(
                        {
                            "source": source,
                            "name": output_name,
                            "chain_type": chain_type,
                            "copied_from": source_name,
                            "match_type": match.match_type,
                            "match_coverage": match.coverage,
                        }
                    )

                if lookup_records:
                    mapping[entry_id][str(entity_id)] = lookup_records
                    stats[f"matched:{chain_type}:{match.match_type}"] += 1
                    stats["entities_with_apo"] += 1
                else:
                    stats[f"matched_but_no_files:{chain_type}"] += 1

            if limit is not None and inspected >= limit:
                break

        if limit is not None and inspected >= limit:
            break

    stats["records_to_copy"] = len(records)
    return records, dict(stats), dict(mapping)


def copy_protein_tokens(
    records: list[FetchRecord],
    *,
    rcsb_dir: pathlib.Path,
    disordered_root: pathlib.Path,
    overwrite: bool,
) -> dict[str, int]:
    """Copy precomputed protein apo tokens when RCSB token LMDBs are available."""
    token_records: dict[str, list[FetchRecord]] = defaultdict(list)
    for record in records:
        if record.chain_type == "protein":
            token_records[record.source].append(record)

    if not token_records:
        return {}

    stats: dict[str, int] = defaultdict(int)
    out_root = disordered_root / "apo_tok_lmdb" / "protein"
    out_root.mkdir(parents=True, exist_ok=True)

    for source, source_records in sorted(token_records.items()):
        in_lmdb = rcsb_dir / "apo_tok_lmdb" / "protein" / f"{source}.lmdb"
        if not in_lmdb.exists():
            stats[f"token_source_missing:{source}"] += len(source_records)
            continue

        out_lmdb = out_root / f"{source}.lmdb"
        if out_lmdb.exists():
            if not overwrite:
                raise FileExistsError(f"{out_lmdb} already exists. Use --overwrite.")
            shutil.rmtree(out_lmdb)

        env_in = lmdb.open(str(in_lmdb), readonly=True, lock=False, readahead=False)
        env_out = lmdb.open(
            str(out_lmdb),
            map_size=64 * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        copied = 0
        missing = 0
        with env_in.begin(write=False) as txn_in:
            txn_out = env_out.begin(write=True)
            try:
                for record in source_records:
                    value = txn_in.get(record.source_name.encode("utf-8"))
                    if value is None:
                        missing += 1
                        continue
                    txn_out.put(record.output_name.encode("utf-8"), value)
                    copied += 1
                    if copied % 10000 == 0:
                        txn_out.commit()
                        txn_out = env_out.begin(write=True)
                txn_out.commit()
            except Exception:
                txn_out.abort()
                raise
        env_in.close()
        env_out.sync()
        env_out.close()
        stats[f"tokens_copied:{source}"] = copied
        stats[f"tokens_missing:{source}"] = missing

    return dict(stats)


def copy_prior_stacks(
    records: list[FetchRecord],
    *,
    rcsb_dir: pathlib.Path,
    disordered_root: pathlib.Path,
    overwrite: bool,
) -> dict[str, int]:
    """Copy matched RCSB prior stack records under disordered entity keys."""
    prior_records: dict[str, dict[str, str]] = {}
    for record in records:
        key = f"{record.chain_type}:{record.output_name}"
        prior_records.setdefault(
            key,
            {
                "chain_type": record.chain_type,
                "input_key": f"{record.entry_id}_{record.rcsb_entity_id}",
                "output_key": record.output_name,
            },
        )

    if not prior_records:
        return {}

    stats: dict[str, int] = defaultdict(int)
    records_by_chain_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in prior_records.values():
        records_by_chain_type[record["chain_type"]].append(record)

    out_root = disordered_root / "prior_lmdb"
    out_root.mkdir(parents=True, exist_ok=True)

    for chain_type, chain_records in sorted(records_by_chain_type.items()):
        in_lmdb = rcsb_dir / "prior_lmdb" / f"{chain_type}.lmdb"
        if not in_lmdb.exists():
            stats[f"prior_source_missing:{chain_type}"] += len(chain_records)
            continue

        out_lmdb = out_root / f"{chain_type}.lmdb"
        if out_lmdb.exists():
            if not overwrite:
                raise FileExistsError(f"{out_lmdb} already exists. Use --overwrite.")
            shutil.rmtree(out_lmdb)

        env_in = lmdb.open(str(in_lmdb), readonly=True, lock=False, readahead=False)
        env_out = lmdb.open(
            str(out_lmdb),
            map_size=128 * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        copied = 0
        missing = 0
        with env_in.begin(write=False) as txn_in:
            txn_out = env_out.begin(write=True)
            try:
                for record in sorted(chain_records, key=lambda item: item["output_key"]):
                    value = txn_in.get(record["input_key"].encode("utf-8"))
                    if value is None:
                        missing += 1
                        continue
                    txn_out.put(record["output_key"].encode("utf-8"), value)
                    copied += 1
                    if copied % 10000 == 0:
                        txn_out.commit()
                        txn_out = env_out.begin(write=True)
                txn_out.commit()
            except Exception:
                txn_out.abort()
                raise
        env_in.close()
        env_out.sync()
        env_out.close()
        stats[f"prior_copied:{chain_type}"] = copied
        stats[f"prior_missing:{chain_type}"] = missing

    return dict(stats)


def write_outputs(
    records: list[FetchRecord],
    mapping: dict[str, dict[str, list[dict]]],
    stats: dict[str, int],
    *,
    rcsb_dir: pathlib.Path,
    disordered_root: pathlib.Path,
    overwrite: bool,
) -> None:
    mapping_msgpack = disordered_root / "sequences" / "rcsb_apo_mapping.msgpack"
    mapping_json = disordered_root / "sequences" / "rcsb_apo_mapping.json"
    report_json = disordered_root / "sequences" / "rcsb_apo_fetch_report.json"
    for path in (mapping_msgpack, mapping_json, report_json):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite.")

    copied = 0
    skipped_existing = 0
    for record in records:
        src = pathlib.Path(record.source_path)
        dst = pathlib.Path(record.output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    token_stats = copy_protein_tokens(
        records,
        rcsb_dir=rcsb_dir,
        disordered_root=disordered_root,
        overwrite=overwrite,
    )
    prior_stats = copy_prior_stacks(
        records,
        rcsb_dir=rcsb_dir,
        disordered_root=disordered_root,
        overwrite=overwrite,
    )

    mapping_msgpack.parent.mkdir(parents=True, exist_ok=True)
    with mapping_msgpack.open("wb") as f:
        msgpack.pack(mapping, f)
    with mapping_json.open("w") as f:
        json.dump(mapping, f, indent=2)
    with report_json.open("w") as f:
        json.dump(
            {
                "stats": stats
                | {
                    "copied": copied,
                    "skipped_existing": skipped_existing,
                    **token_stats,
                    **prior_stats,
                },
                "records": [asdict(record) for record in records[:1000]],
            },
            f,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    disordered_root = args.data_dir / "disordered_pdb"
    rcsb_dir = args.rcsb_dir or args.data_dir / "rcsb-train"
    sources = normalize_sources(args.sources)

    records, stats, mapping = collect_fetch_records(
        disordered_root=disordered_root,
        rcsb_dir=rcsb_dir,
        sources=sources,
        limit=args.limit,
    )

    print("Fetch summary:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print(f"  mapping entries: {sum(len(v) for v in mapping.values())}")

    if args.dry_run:
        print("Dry run: no files copied.")
        return
    if not records and not mapping:
        raise RuntimeError(
            "No apo records were found. Refusing to overwrite existing mapping files. "
            "Check that the RCSB raw apo source files are available."
        )

    write_outputs(
        records,
        mapping,
        stats,
        rcsb_dir=rcsb_dir,
        disordered_root=disordered_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote apo files under {disordered_root / 'apo'}")
    print(f"Wrote mapping under {disordered_root / 'sequences'}")


if __name__ == "__main__":
    main()
