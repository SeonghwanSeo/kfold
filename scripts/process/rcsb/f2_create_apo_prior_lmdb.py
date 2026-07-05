"""Create source-specific apo LMDBs and chain-type prior stack LMDBs."""

import argparse
import json
import multiprocessing as mp
import pathlib
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from kfold.data.utils.io.fasta import read_fasta
from kfold.data.utils.io.structure import (
    read_dna_structure,
    read_protein_structure,
    read_rna_structure,
)
from kfold.training.dataset.utils.apo_io import pack_apo_record, pack_prior_stack_record


@dataclass(frozen=True)
class ApoTask:
    path: str
    key: str
    chain_type: str
    source: str


@dataclass(frozen=True)
class PriorStackTask:
    key: str
    chain_type: str
    paths: tuple[str, ...]
    sample_names: tuple[str, ...]
    confidence_paths: tuple[str | None, ...]


@dataclass(frozen=True)
class RankedRecord:
    entity_key: str
    chain_type: str
    source: str
    length: int
    ptm: float | None
    avg_plddt: float | None
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to dataset working directory.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Data split to process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
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
        "--sampler_dir",
        default="apo",
        help="Sampler archive directory under rcsb-{split}.",
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


def load_entity_sequences(seq_dir: pathlib.Path) -> dict[tuple[str, int], dict[str, str]]:
    entity_sequences: dict[tuple[str, int], dict[str, str]] = {}
    for header, sequence in read_fasta(seq_dir / "all_sequences.fasta"):
        pdb_id, entity_id_str, chain_type = header.split("|")
        entity_sequences[(pdb_id.lower(), int(entity_id_str))] = {
            "entity_key": f"{pdb_id.lower()}_{entity_id_str}",
            "chain_type": chain_type.lower(),
            "sequence": sequence,
        }
    return entity_sequences


def index_entity_sequences(
    entity_sequences: dict[tuple[str, int], dict[str, str]],
) -> dict[str, dict[str, str]]:
    return {entity["entity_key"]: entity for entity in entity_sequences.values()}


def load_confidence(path: pathlib.Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {"ptm": None, "avg_plddt": None}
    with path.open() as f:
        data = json.load(f)
    return {
        "ptm": data.get("ptm"),
        "avg_plddt": data.get("avg_plddt"),
    }


def remove_structure_suffix(path: pathlib.Path) -> str:
    for suffix in (".pdb.zst", ".cif.zst", ".pdb", ".cif"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def first_existing(paths: list[pathlib.Path]) -> pathlib.Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_existing_name(
    files_by_name: dict[str, pathlib.Path], names: list[str]
) -> pathlib.Path | None:
    for name in names:
        path = files_by_name.get(name)
        if path is not None:
            return path
    return None


def confidence_path_for_sample_name(
    files_by_name: dict[str, pathlib.Path],
    sample_name: str,
) -> pathlib.Path | None:
    name = remove_structure_suffix(pathlib.Path(sample_name))
    candidates = [f"{name}_confidence.json"]
    if name.endswith("_model"):
        candidates.append(f"{name.removesuffix('_model')}_confidence.json")
    return first_existing_name(files_by_name, candidates)


def get_entity_dir(
    source_dir: pathlib.Path, entry_id: str, entity_id: int
) -> pathlib.Path:
    entity_key = f"{entry_id}_{entity_id}"
    candidates = [
        source_dir / entity_key,
        source_dir / entry_id[1:3] / entry_id / str(entity_id),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_source_dirs(raw_root: pathlib.Path, chain_type: str) -> list[pathlib.Path]:
    root = raw_root / chain_type
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_entity_dirs(source_dir: pathlib.Path) -> list[pathlib.Path]:
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.iterdir() if path.is_dir())


def add_lookup_record(
    lookup: dict[str, dict[str, list[dict]]],
    entry_id: str,
    entity_id: int,
    record: dict,
) -> None:
    lookup[entry_id][str(entity_id)].append(record)


def make_apo_record(
    source: str,
    entity_key: str,
    chain_type: str,
    confidence: dict[str, float | None] | None = None,
    **extra,
) -> dict:
    record = {
        "source": source,
        "name": entity_key,
        "chain_type": chain_type,
    }
    if confidence is not None:
        for key in ("ptm", "avg_plddt"):
            if confidence.get(key) is not None:
                record[key] = float(confidence[key])
    record.update({k: v for k, v in extra.items() if v is not None})
    return record


def collect_protein_source(
    source_dir: pathlib.Path,
    source: str,
    entity_sequences_by_key: dict[str, dict[str, str]],
    apo_lookup: dict[str, dict[str, list[dict]]],
    apo_tasks: list[ApoTask],
    prior_samples: dict[str, list[tuple[pathlib.Path, str, pathlib.Path | None]]],
    ranked_records: list[RankedRecord],
) -> dict[str, int]:
    stats = defaultdict(int)
    for entity_dir in iter_entity_dirs(source_dir):
        entity_key = entity_dir.name
        entity = entity_sequences_by_key.get(entity_key)
        if entity is None or entity["chain_type"] != "protein":
            stats["skipped_protein_entity_dir"] += 1
            continue
        entry_id, entity_id_str = entity_key.rsplit("_", 1)
        entity_id = int(entity_id_str)
        files_by_name = {
            path.name: path for path in entity_dir.iterdir() if path.is_file()
        }
        ranked_path = first_existing_name(
            files_by_name,
            [
                f"{entity_key}_ranked_model.pdb.zst",
                f"{entity_key}_ranked.pdb.zst",
                f"{entity_key}_ranked_model.cif.zst",
                f"{entity_key}_ranked.cif.zst",
            ],
        )
        ranked_conf = first_existing_name(
            files_by_name,
            [
                f"{entity_key}_ranked_confidence.json",
                f"{entity_key}_ranked_confidences.json",
            ],
        )
        if ranked_path is not None:
            confidence = load_confidence(ranked_conf)
            record = make_apo_record(source, entity_key, "protein", confidence)
            add_lookup_record(apo_lookup, entry_id, entity_id, record)
            apo_tasks.append(ApoTask(str(ranked_path), entity_key, "protein", source))
            ranked_records.append(
                RankedRecord(
                    entity_key=entity_key,
                    chain_type="protein",
                    source=source,
                    length=len(entity["sequence"]),
                    ptm=confidence["ptm"],
                    avg_plddt=confidence["avg_plddt"],
                    path=str(ranked_path),
                )
            )
            stats["apo_protein"] += 1
        else:
            stats["missing_protein_ranked"] += 1

        sample_paths = sorted(
            path
            for name, path in files_by_name.items()
            if name.startswith(f"{entity_key}_seed-")
            and "_sample-" in name
            and (name.endswith(".pdb.zst") or name.endswith(".cif.zst"))
        )
        for sample_path in sample_paths:
            sample_name = f"{source}/{remove_structure_suffix(sample_path)}"
            conf_path = confidence_path_for_sample_name(files_by_name, sample_path.name)
            prior_samples[entity_key].append((sample_path, sample_name, conf_path))
            stats["prior_protein_samples"] += 1
    return dict(stats)


def collect_rna_source(
    source_dir: pathlib.Path,
    source: str,
    entity_sequences_by_key: dict[str, dict[str, str]],
    apo_lookup: dict[str, dict[str, list[dict]]],
    apo_tasks: list[ApoTask],
    prior_samples: dict[str, list[tuple[pathlib.Path, str, pathlib.Path | None]]],
    ranked_records: list[RankedRecord],
) -> dict[str, int]:
    stats = defaultdict(int)
    for entity_dir in iter_entity_dirs(source_dir):
        entity_key = entity_dir.name
        entity = entity_sequences_by_key.get(entity_key)
        if entity is None or entity["chain_type"] != "rna":
            stats["skipped_rna_entity_dir"] += 1
            continue
        entry_id, entity_id_str = entity_key.rsplit("_", 1)
        entity_id = int(entity_id_str)
        files_by_name = {
            path.name: path for path in entity_dir.iterdir() if path.is_file()
        }
        ranked_path = first_existing_name(
            files_by_name,
            [
                f"{entity_key}_ranked_model.pdb.zst",
                f"{entity_key}_ranked.pdb.zst",
                f"{entity_key}_ranked_model.cif.zst",
                f"{entity_key}_ranked.cif.zst",
            ],
        )
        ranked_conf = first_existing_name(
            files_by_name,
            [
                f"{entity_key}_ranked_confidences.json",
                f"{entity_key}_ranked_confidence.json",
            ],
        )
        if ranked_path is not None:
            confidence = load_confidence(ranked_conf)
            record = make_apo_record(source, entity_key, "rna", confidence)
            add_lookup_record(apo_lookup, entry_id, entity_id, record)
            apo_tasks.append(ApoTask(str(ranked_path), entity_key, "rna", source))
            ranked_records.append(
                RankedRecord(
                    entity_key=entity_key,
                    chain_type="rna",
                    source=source,
                    length=len(entity["sequence"]),
                    ptm=confidence["ptm"],
                    avg_plddt=confidence["avg_plddt"],
                    path=str(ranked_path),
                )
            )
            stats["apo_rna"] += 1
        else:
            stats["missing_rna_ranked"] += 1

        sample_paths = sorted(
            path
            for name, path in files_by_name.items()
            if name.startswith(f"{entity_key}_sample_")
            and (name.endswith("_model.pdb.zst") or name.endswith("_model.cif.zst"))
        )
        for sample_path in sample_paths:
            name = remove_structure_suffix(sample_path).removesuffix("_model")
            sample_name = f"{source}/{name}"
            conf_path = entity_dir / f"{name}_confidences.json"
            prior_samples[entity_key].append((sample_path, sample_name, conf_path))
            stats["prior_rna_samples"] += 1
    return dict(stats)


def collect_dna_source(
    source_dir: pathlib.Path,
    source: str,
    entity_sequences: dict[tuple[str, int], dict[str, str]],
    apo_lookup: dict[str, dict[str, list[dict]]],
    apo_tasks: list[ApoTask],
    prior_samples: dict[str, list[tuple[pathlib.Path, str, pathlib.Path | None]]],
) -> dict[str, int]:
    stats = defaultdict(int)
    for (entry_id, entity_id), entity in sorted(entity_sequences.items()):
        if entity["chain_type"] != "dna":
            continue
        entity_key = entity["entity_key"]
        path = first_existing(
            [
                source_dir / f"{entity_key}.pdb.zst",
                source_dir / f"{entity_key}.cif.zst",
                get_entity_dir(source_dir, entry_id, entity_id) / "helix.pdb.zst",
            ]
        )
        if path is None:
            stats["missing_dna_helix"] += 1
            continue
        record = make_apo_record(source, entity_key, "dna", model="single_helix")
        add_lookup_record(apo_lookup, entry_id, entity_id, record)
        apo_tasks.append(ApoTask(str(path), entity_key, "dna", source))
        prior_samples[entity_key].append((path, f"{source}/{entity_key}", None))
        stats["apo_dna"] += 1
        stats["prior_dna_samples"] += 1
    return dict(stats)


def make_prior_tasks(
    prior_samples_by_type: dict[
        str, dict[str, list[tuple[pathlib.Path, str, pathlib.Path | None]]]
    ],
) -> list[PriorStackTask]:
    tasks: list[PriorStackTask] = []
    for chain_type, samples_by_entity in sorted(prior_samples_by_type.items()):
        for entity_key, samples in sorted(samples_by_entity.items()):
            samples = sorted(samples, key=lambda item: item[1])
            tasks.append(
                PriorStackTask(
                    key=entity_key,
                    chain_type=chain_type,
                    paths=tuple(str(item[0]) for item in samples),
                    sample_names=tuple(item[1] for item in samples),
                    confidence_paths=tuple(
                        None if item[2] is None else str(item[2]) for item in samples
                    ),
                )
            )
    return tasks


def read_structure(path: pathlib.Path, chain_type: str) -> tuple[str, np.ndarray]:
    if chain_type == "protein":
        return read_protein_structure(path)
    if chain_type == "rna":
        return read_rna_structure(path)
    if chain_type == "dna":
        return read_dna_structure(path)
    raise ValueError(f"Unsupported chain type: {chain_type}")


def apo_worker(task: ApoTask) -> tuple[str, str, str, bytes | None, str | None]:
    try:
        sequence, coords = read_structure(pathlib.Path(task.path), task.chain_type)
        return (
            task.chain_type,
            task.source,
            task.key,
            pack_apo_record(sequence, coords, task.chain_type),
            None,
        )
    except Exception as e:
        return task.chain_type, task.source, task.key, None, f"{task.path}: {e}"


def prior_stack_worker(task: PriorStackTask) -> tuple[str, str, bytes | None, str | None]:
    try:
        sequences: list[str] = []
        coords_list: list[np.ndarray] = []
        ptms: list[float] = []
        avg_plddts: list[float] = []
        for path_str, confidence_path in zip(
            task.paths, task.confidence_paths, strict=True
        ):
            sequence, coords = read_structure(pathlib.Path(path_str), task.chain_type)
            sequences.append(sequence)
            coords_list.append(coords)
            confidence = load_confidence(
                None if confidence_path is None else pathlib.Path(confidence_path)
            )
            ptms.append(np.nan if confidence["ptm"] is None else float(confidence["ptm"]))
            avg_plddts.append(
                np.nan
                if confidence["avg_plddt"] is None
                else float(confidence["avg_plddt"])
            )

        if len(set(sequences)) != 1:
            lengths = [len(seq) for seq in sequences]
            raise ValueError(f"Sequence mismatch across prior samples: lengths={lengths}")
        shapes = {coords.shape for coords in coords_list}
        if len(shapes) != 1:
            raise ValueError(f"Coordinate shape mismatch across prior samples: {shapes}")

        value = pack_prior_stack_record(
            sequences[0],
            np.stack(coords_list, axis=0),
            task.chain_type,
            list(task.sample_names),
            ptm=np.asarray(ptms, dtype=np.float32),
            avg_plddt=np.asarray(avg_plddts, dtype=np.float32),
        )
        return task.chain_type, task.key, value, None
    except Exception as e:
        return task.chain_type, task.key, None, f"{task.key}: {e}"


def ensure_overwrite(path: pathlib.Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_lookup(path: pathlib.Path, lookup: dict[str, dict[str, list[dict]]]) -> None:
    with path.with_suffix(".json").open("w") as f:
        json.dump(lookup, f, indent=2)
    with path.with_suffix(".msgpack").open("wb") as f:
        msgpack.pack(lookup, f)


def write_apo_lmdbs(
    data_dir: pathlib.Path,
    tasks: list[ApoTask],
    num_workers: int,
    map_size_gb: int,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[str, str], list[ApoTask]] = defaultdict(list)
    for task in tasks:
        grouped[(task.chain_type, task.source)].append(task)

    root = data_dir / "apo_lmdb"
    ensure_overwrite(root, overwrite)
    root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, int]] = {}
    for (chain_type, source), group_tasks in sorted(grouped.items()):
        path = root / chain_type / f"{source}.lmdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(path),
            map_size=map_size_gb * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        written = 0
        failed = 0
        errors: list[str] = []
        with mp.Pool(processes=num_workers) as pool:
            txn = env.begin(write=True)
            try:
                for _, _, key, value, error in tqdm(
                    pool.imap_unordered(apo_worker, group_tasks, chunksize=16),
                    total=len(group_tasks),
                    desc=f"Writing apo {chain_type}/{source}",
                ):
                    if value is None:
                        failed += 1
                        if error is not None and len(errors) < 20:
                            errors.append(error)
                        continue
                    txn.put(key.encode("utf-8"), value)
                    written += 1
                    if written % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                txn.commit()
            except Exception:
                txn.abort()
                raise
        env.sync()
        env.close()
        if errors:
            print(f"First {len(errors)} errors while writing {path}:")
            for error in errors:
                print(f"  {error}")
        results[f"{chain_type}/{source}"] = {"written": written, "failed": failed}
    return results


def write_prior_lmdbs(
    data_dir: pathlib.Path,
    tasks: list[PriorStackTask],
    num_workers: int,
    map_size_gb: int,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[PriorStackTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.chain_type].append(task)

    root = data_dir / "prior_lmdb"
    ensure_overwrite(root, overwrite)
    root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, int]] = {}
    for chain_type, group_tasks in sorted(grouped.items()):
        path = root / f"{chain_type}.lmdb"
        env = lmdb.open(
            str(path),
            map_size=map_size_gb * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        written = 0
        failed = 0
        errors: list[str] = []
        with mp.Pool(processes=num_workers) as pool:
            txn = env.begin(write=True)
            try:
                for _, key, value, error in tqdm(
                    pool.imap_unordered(prior_stack_worker, group_tasks, chunksize=4),
                    total=len(group_tasks),
                    desc=f"Writing prior {chain_type}",
                ):
                    if value is None:
                        failed += 1
                        if error is not None and len(errors) < 20:
                            errors.append(error)
                        continue
                    txn.put(key.encode("utf-8"), value)
                    written += 1
                    if written % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                txn.commit()
            except Exception:
                txn.abort()
                raise
        env.sync()
        env.close()
        if errors:
            print(f"First {len(errors)} errors while writing {path}:")
            for error in errors:
                print(f"  {error}")
        results[chain_type] = {"written": written, "failed": failed}
    return results


def summarize_ranked(
    ranked_records: list[RankedRecord],
    top_n: int,
) -> dict[str, list[dict]]:
    low_ptm_candidates = [
        record
        for record in ranked_records
        if record.ptm is not None and record.length >= 100
    ]
    low_ptm = sorted(low_ptm_candidates, key=lambda record: record.ptm)[:top_n]
    long = sorted(ranked_records, key=lambda record: record.length, reverse=True)[:top_n]
    return {
        "low_ptm_len_ge_100": [asdict(record) for record in low_ptm],
        "longest": [asdict(record) for record in long],
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    raw_root = data_dir / args.sampler_dir
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    entity_sequences = load_entity_sequences(data_dir / "sequences")
    entity_sequences_by_key = index_entity_sequences(entity_sequences)
    apo_lookup: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    apo_tasks: list[ApoTask] = []
    prior_samples_by_type: dict[
        str, dict[str, list[tuple[pathlib.Path, str, pathlib.Path | None]]]
    ] = defaultdict(lambda: defaultdict(list))
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

    for source_dir in iter_source_dirs(raw_root, "rna"):
        part = collect_rna_source(
            source_dir,
            source_dir.name,
            entity_sequences_by_key,
            apo_lookup,
            apo_tasks,
            prior_samples_by_type["rna"],
            ranked_records,
        )
        for key, value in part.items():
            stats[f"{source_dir.name}:{key}"] += value

    for source_dir in iter_source_dirs(raw_root, "dna"):
        part = collect_dna_source(
            source_dir,
            source_dir.name,
            entity_sequences,
            apo_lookup,
            apo_tasks,
            prior_samples_by_type["dna"],
        )
        for key, value in part.items():
            stats[f"{source_dir.name}:{key}"] += value

    prior_tasks = make_prior_tasks(prior_samples_by_type)
    apo_lookup = {entry: dict(chains) for entry, chains in apo_lookup.items()}
    ranked_summary = summarize_ranked(ranked_records, args.top_n)

    print("Collection summary:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
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
