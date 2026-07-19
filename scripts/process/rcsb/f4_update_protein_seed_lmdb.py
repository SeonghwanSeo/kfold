"""Append newly added protein sampler seeds to the RCSB apo/prior LMDBs.

The sampler archive can use either of these layouts::

    rcsb-{split}/apo/prot_sampler_seed3/{entity_key}/...
    rcsb-{split}/apo/protein/prot_sampler_seed3/{entity_key}/...

Only sampler sources without an existing ``apo_lmdb/protein/{source}.lmdb``
are processed.  Existing prior samples are retained and the new samples are
appended in numerical seed order.

All outputs are first written below a resumable staging directory.  The live
LMDBs are installed only after the staged files have been validated.  The apo
lookup remains unchanged by default because newly added apo sources also need
matching ``apo_tok_lmdb`` files before training can select them.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import pathlib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime

import lmdb
import msgpack
import numpy as np
import zstandard
from tqdm import tqdm

from kfold.data.utils.io.structure import read_protein_structure
from kfold.training.dataset.utils.apo_io import (
    pack_apo_record,
    pack_prior_stack_record,
    unpack_prior_stack_record,
)

SOURCE_PATTERN = re.compile(r"^prot_sampler_seed(?P<seed>\d+)(?:_.+)?$")
STRUCTURE_SUFFIXES = (".pdb.zst", ".cif.zst", ".pdb", ".cif")


@dataclass(frozen=True)
class EntityTask:
    key: str
    sources: tuple[tuple[str, str], ...]


_PRIOR_ENV: lmdb.Environment | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Dataset root containing rcsb-{split}.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument(
        "--sampler_dir",
        default="apo",
        help="Sampler directory below rcsb-{split}.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optional explicit sampler source names to add.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
    )
    parser.add_argument(
        "--apo_map_size_gb",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--prior_map_size_gb",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--commit_interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--stage_dir",
        type=pathlib.Path,
        help="Resumable staging directory (default: below rcsb-{split}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N entities. Requires --no_install.",
    )
    parser.add_argument(
        "--no_install",
        action="store_true",
        help="Build and validate staging outputs without replacing live files.",
    )
    parser.add_argument(
        "--install_lookup",
        action="store_true",
        help=(
            "Also activate the new apo sources in apo_lookup. Use only after "
            "matching apo_tok_lmdb source files have been generated."
        ),
    )
    return parser.parse_args()


def source_sort_key(name: str) -> tuple[int, str]:
    match = SOURCE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a protein sampler source: {name}")
    return int(match.group("seed")), name


def discover_sources(raw_root: pathlib.Path) -> dict[str, pathlib.Path]:
    protein_root = raw_root / "protein"
    if not protein_root.is_dir():
        protein_root = raw_root
    return {
        path.name: path
        for path in protein_root.iterdir()
        if path.is_dir() and SOURCE_PATTERN.fullmatch(path.name)
    }


def remove_structure_suffix(name: str) -> str:
    for suffix in STRUCTURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return pathlib.Path(name).stem


def load_confidence(path: pathlib.Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {"ptm": None, "avg_plddt": None}
    with path.open() as f:
        data = json.load(f)
    return {"ptm": data.get("ptm"), "avg_plddt": data.get("avg_plddt")}


def ranked_paths(
    entity_dir: pathlib.Path, entity_key: str
) -> tuple[pathlib.Path, pathlib.Path | None]:
    files = {path.name: path for path in entity_dir.iterdir() if path.is_file()}
    structure_names = (
        f"{entity_key}_ranked_model.pdb.zst",
        f"{entity_key}_ranked.pdb.zst",
        f"{entity_key}_ranked_model.cif.zst",
        f"{entity_key}_ranked.cif.zst",
        f"{entity_key}_ranked_model.pdb",
        f"{entity_key}_ranked.pdb",
        f"{entity_key}_ranked_model.cif",
        f"{entity_key}_ranked.cif",
    )
    ranked = next((files[name] for name in structure_names if name in files), None)
    if ranked is None:
        raise FileNotFoundError(f"Ranked structure not found in {entity_dir}")
    confidence_names = (
        f"{entity_key}_ranked_confidence.json",
        f"{entity_key}_ranked_confidences.json",
    )
    confidence = next((files[name] for name in confidence_names if name in files), None)
    return ranked, confidence


def sample_paths(
    entity_dir: pathlib.Path, entity_key: str
) -> list[tuple[pathlib.Path, pathlib.Path | None]]:
    files = {path.name: path for path in entity_dir.iterdir() if path.is_file()}
    structures = sorted(
        path
        for name, path in files.items()
        if name.startswith(f"{entity_key}_seed-")
        and "_sample-" in name
        and name.endswith(STRUCTURE_SUFFIXES)
    )
    samples: list[tuple[pathlib.Path, pathlib.Path | None]] = []
    for structure in structures:
        name = remove_structure_suffix(structure.name)
        confidence_names = [f"{name}_confidence.json"]
        if name.endswith("_model"):
            confidence_names.append(f"{name.removesuffix('_model')}_confidence.json")
        confidence = next(
            (files[candidate] for candidate in confidence_names if candidate in files),
            None,
        )
        samples.append((structure, confidence))
    return samples


def init_worker(prior_path: str) -> None:
    global _PRIOR_ENV
    _PRIOR_ENV = lmdb.open(
        prior_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1024,
    )


def process_entity(
    task: EntityTask,
) -> tuple[str, dict, bytes, bytes] | tuple[str, None, None, str]:
    try:
        assert _PRIOR_ENV is not None
        with _PRIOR_ENV.begin(write=False) as txn:
            old_value = txn.get(task.key.encode())

        if old_value is None:
            sequence: str | None = None
            coords_list: list[np.ndarray] = []
            sample_names: list[str] = []
            ptms: list[float] = []
            avg_plddts: list[float] = []
        else:
            old = unpack_prior_stack_record(old_value)
            sequence = old["seq"]
            coords_list = list(old["coords"])
            sample_names = list(old["sample_names"])
            ptms = old.get(
                "ptm", np.full(len(sample_names), np.nan, dtype=np.float32)
            ).tolist()
            avg_plddts = old.get(
                "avg_plddt", np.full(len(sample_names), np.nan, dtype=np.float32)
            ).tolist()

        apo_values: dict[str, bytes] = {}
        lookup_records: list[dict] = []
        for source, source_dir_str in task.sources:
            entity_dir = pathlib.Path(source_dir_str) / task.key
            if not entity_dir.is_dir():
                continue

            ranked_path, ranked_confidence_path = ranked_paths(entity_dir, task.key)
            ranked_sequence, ranked_coords = read_protein_structure(ranked_path)
            if sequence is not None and ranked_sequence != sequence:
                raise ValueError(f"{source}: ranked sequence differs from prior")
            apo_values[source] = pack_apo_record(
                ranked_sequence, ranked_coords, "protein"
            )
            ranked_confidence = load_confidence(ranked_confidence_path)
            lookup_record: dict = {
                "source": source,
                "name": task.key,
                "chain_type": "protein",
            }
            for name, value in ranked_confidence.items():
                if value is not None:
                    lookup_record[name] = float(value)
            lookup_records.append(lookup_record)

            samples = sample_paths(entity_dir, task.key)
            if not samples:
                raise FileNotFoundError(f"{source}: no prior samples in {entity_dir}")
            for structure_path, confidence_path in samples:
                sample_sequence, sample_coords = read_protein_structure(structure_path)
                if sequence is None:
                    sequence = sample_sequence
                elif sample_sequence != sequence:
                    raise ValueError(
                        f"{source}/{structure_path.name}: sequence differs from prior"
                    )
                coords_list.append(sample_coords)
                sample_names.append(
                    f"{source}/{remove_structure_suffix(structure_path.name)}"
                )
                confidence = load_confidence(confidence_path)
                ptms.append(
                    np.nan if confidence["ptm"] is None else float(confidence["ptm"])
                )
                avg_plddts.append(
                    np.nan
                    if confidence["avg_plddt"] is None
                    else float(confidence["avg_plddt"])
                )

        if sequence is None or not coords_list:
            raise ValueError("No old or new protein prior samples")
        shapes = {coords.shape for coords in coords_list}
        if len(shapes) != 1:
            raise ValueError(f"Coordinate shapes differ: {sorted(shapes)}")

        prior_value = pack_prior_stack_record(
            sequence,
            np.stack(coords_list),
            "protein",
            sample_names,
            ptm=np.asarray(ptms, dtype=np.float32),
            avg_plddt=np.asarray(avg_plddts, dtype=np.float32),
        )
        lookup_value = msgpack.packb(lookup_records, use_bin_type=True)
        return task.key, apo_values, prior_value, lookup_value
    except Exception as exc:
        return task.key, None, None, f"{task.key}: {exc}"


def open_write_env(path: pathlib.Path, map_size_gb: int) -> lmdb.Environment:
    path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path),
        map_size=map_size_gb * 1024**3,
        meminit=False,
        map_async=True,
        sync=False,
    )


def lmdb_keys(path: pathlib.Path) -> set[str]:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    with env.begin() as txn:
        keys = {key.decode() for key, _ in txn.cursor()}
    env.close()
    return keys


def source_entity_keys(source_paths: dict[str, pathlib.Path]) -> set[str]:
    keys: set[str] = set()
    for source, path in source_paths.items():
        print(f"Collecting entity names from {source}...")
        keys.update(entry.name for entry in os.scandir(path) if entry.is_dir())
    return keys


def write_stage_metadata(
    path: pathlib.Path,
    source_paths: dict[str, pathlib.Path],
    prior_path: pathlib.Path,
) -> None:
    metadata = {
        "sources": {name: str(path.resolve()) for name, path in source_paths.items()},
        "prior_path": str(prior_path.resolve()),
        "prior_mtime_ns": prior_path.stat().st_mtime_ns,
    }
    if path.exists():
        with path.open() as f:
            existing = json.load(f)
        if existing != metadata:
            raise ValueError(
                f"Staging metadata differs from this run: {path}. "
                "Use a different --stage_dir."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metadata, f, indent=2)


def build_staging_outputs(
    *,
    source_paths: dict[str, pathlib.Path],
    prior_path: pathlib.Path,
    stage_dir: pathlib.Path,
    entity_keys: list[str],
    num_workers: int,
    apo_map_size_gb: int,
    prior_map_size_gb: int,
    commit_interval: int,
) -> None:
    stage_prior_path = stage_dir / "prior_lmdb" / "protein.lmdb"
    stage_lookup_path = stage_dir / "lookup_additions.lmdb"
    apo_envs = {
        source: open_write_env(
            stage_dir / "apo_lmdb" / "protein" / f"{source}.lmdb",
            apo_map_size_gb,
        )
        for source in source_paths
    }
    prior_env = open_write_env(stage_prior_path, prior_map_size_gb)
    lookup_env = open_write_env(stage_lookup_path, 8)

    with prior_env.begin() as txn:
        completed = {key.decode() for key, _ in txn.cursor()}
    remaining = [key for key in entity_keys if key not in completed]
    print(f"Staged prior entries: {len(completed)}; remaining: {len(remaining)}")
    if not remaining:
        for env in [*apo_envs.values(), prior_env, lookup_env]:
            env.close()
        return

    task_sources = tuple((name, str(path)) for name, path in source_paths.items())
    tasks = (EntityTask(key, task_sources) for key in remaining)
    errors: list[str] = []
    written_since_commit = 0
    apo_txns = {source: env.begin(write=True) for source, env in apo_envs.items()}
    prior_txn = prior_env.begin(write=True)
    lookup_txn = lookup_env.begin(write=True)

    def commit() -> None:
        nonlocal apo_txns, prior_txn, lookup_txn, written_since_commit
        for txn in apo_txns.values():
            txn.commit()
        lookup_txn.commit()
        prior_txn.commit()
        apo_txns = {source: env.begin(write=True) for source, env in apo_envs.items()}
        prior_txn = prior_env.begin(write=True)
        lookup_txn = lookup_env.begin(write=True)
        written_since_commit = 0

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(str(prior_path),),
    ) as pool:
        try:
            for key, apo_values, prior_value, lookup_value in tqdm(
                pool.imap_unordered(process_entity, tasks, chunksize=1),
                total=len(remaining),
                desc="Appending protein seed priors",
            ):
                if apo_values is None:
                    errors.append(str(lookup_value))
                    continue
                for source, value in apo_values.items():
                    apo_txns[source].put(key.encode(), value)
                lookup_txn.put(key.encode(), lookup_value)
                prior_txn.put(key.encode(), prior_value)
                written_since_commit += 1
                if written_since_commit >= commit_interval:
                    commit()
            commit()
        except Exception:
            for txn in apo_txns.values():
                txn.abort()
            lookup_txn.abort()
            prior_txn.abort()
            raise

    for env in [*apo_envs.values(), lookup_env, prior_env]:
        # commit() opens the next empty transaction for the resumable loop.
        # Abort those final empty transactions before closing the environments.
        if env is prior_env:
            prior_txn.abort()
        elif env is lookup_env:
            lookup_txn.abort()
        else:
            source = next(name for name, apo_env in apo_envs.items() if apo_env is env)
            apo_txns[source].abort()
        env.sync()
        env.close()

    if errors:
        error_path = stage_dir / "errors.txt"
        with error_path.open("w") as f:
            f.write("\n".join(errors) + "\n")
        raise RuntimeError(f"{len(errors)} entities failed; see {error_path}")


def lmdb_entry_count(path: pathlib.Path) -> int:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    with env.begin() as txn:
        count = txn.stat()["entries"]
    env.close()
    return count


def validate_staging(
    stage_dir: pathlib.Path,
    source_paths: dict[str, pathlib.Path],
    entity_keys: list[str],
) -> None:
    expected_prior_entries = len(entity_keys)
    prior_count = lmdb_entry_count(stage_dir / "prior_lmdb" / "protein.lmdb")
    if prior_count != expected_prior_entries:
        raise ValueError(
            f"Staged prior has {prior_count} entries; expected {expected_prior_entries}"
        )
    lookup_count = lmdb_entry_count(stage_dir / "lookup_additions.lmdb")
    if lookup_count != expected_prior_entries:
        raise ValueError(
            f"Staged lookup additions have {lookup_count} entries; "
            f"expected {expected_prior_entries}"
        )
    for source, source_path in source_paths.items():
        expected = sum((source_path / key).is_dir() for key in entity_keys)
        actual = lmdb_entry_count(stage_dir / "apo_lmdb" / "protein" / f"{source}.lmdb")
        if actual != expected:
            raise ValueError(
                f"Staged apo {source} has {actual} entries; expected {expected}"
            )
        print(f"Validated apo {source}: {actual} entries")
    print(f"Validated protein prior: {prior_count} entries")


def build_lookup_files(
    data_dir: pathlib.Path,
    stage_dir: pathlib.Path,
    new_sources: set[str],
) -> None:
    with (data_dir / "apo_lookup.msgpack").open("rb") as f:
        lookup = msgpack.unpack(f, raw=False, strict_map_key=False)

    additions_env = lmdb.open(
        str(stage_dir / "lookup_additions.lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    added = 0
    with additions_env.begin() as txn:
        for key_bytes, value in tqdm(
            txn.cursor(), total=txn.stat()["entries"], desc="Updating apo lookup"
        ):
            records = msgpack.unpackb(value, raw=False)
            if not records:
                continue
            entity_key = key_bytes.decode()
            entry_id, entity_id = entity_key.rsplit("_", 1)
            entity_records = lookup.setdefault(entry_id, {}).setdefault(entity_id, [])
            entity_records[:] = [
                record
                for record in entity_records
                if record.get("source") not in new_sources
            ]
            entity_records.extend(records)
            added += len(records)
    additions_env.close()

    msgpack_path = stage_dir / "apo_lookup.msgpack"
    with msgpack_path.open("wb") as f:
        msgpack.pack(lookup, f, use_bin_type=True)

    json_zst_path = stage_dir / "apo_lookup.json.zst"
    compressor = zstandard.ZstdCompressor(level=3)
    with json_zst_path.open("wb") as raw, compressor.stream_writer(raw) as compressed:
        text = io.TextIOWrapper(compressed, encoding="utf-8")
        json.dump(lookup, text, separators=(",", ":"))
        text.flush()
        text.detach()
    print(f"Added {added} protein apo lookup records")


def install_outputs(
    data_dir: pathlib.Path,
    stage_dir: pathlib.Path,
    sources: list[str],
    install_lookup: bool,
) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = data_dir / f".protein_seed_lmdb_backup_{timestamp}"
    backup_dir.mkdir()

    live_prior = data_dir / "prior_lmdb" / "protein.lmdb"
    staged_prior = stage_dir / "prior_lmdb" / "protein.lmdb"
    shutil.move(str(live_prior), str(backup_dir / "protein.lmdb"))
    shutil.move(str(staged_prior), str(live_prior))

    live_apo_root = data_dir / "apo_lmdb" / "protein"
    for source in sources:
        staged = stage_dir / "apo_lmdb" / "protein" / f"{source}.lmdb"
        live = live_apo_root / f"{source}.lmdb"
        if live.exists():
            raise FileExistsError(live)
        shutil.move(str(staged), str(live))

    if install_lookup:
        for name in ("apo_lookup.msgpack", "apo_lookup.json.zst"):
            live = data_dir / name
            staged = stage_dir / name
            if live.exists():
                shutil.move(str(live), str(backup_dir / name))
            shutil.move(str(staged), str(live))

    shutil.move(str(stage_dir / "metadata.json"), str(backup_dir / "metadata.json"))
    shutil.move(
        str(stage_dir / "lookup_additions.lmdb"),
        str(backup_dir / "lookup_additions.lmdb"),
    )
    return backup_dir


def main() -> None:
    args = parse_args()
    if args.limit is not None and not args.no_install:
        raise ValueError("--limit requires --no_install")

    data_dir = args.data_dir / f"rcsb-{args.split}"
    raw_root = data_dir / args.sampler_dir
    prior_path = data_dir / "prior_lmdb" / "protein.lmdb"
    apo_root = data_dir / "apo_lmdb" / "protein"
    stage_dir = args.stage_dir or data_dir / ".protein_seed_lmdb_update"

    discovered = discover_sources(raw_root)
    if args.sources is not None:
        missing = set(args.sources) - set(discovered)
        if missing:
            raise FileNotFoundError(f"Sampler sources not found: {sorted(missing)}")
        discovered = {name: discovered[name] for name in args.sources}
    existing_apo_sources = {path.stem for path in apo_root.glob("*.lmdb")}
    new_source_names = sorted(set(discovered) - existing_apo_sources, key=source_sort_key)
    if not new_source_names:
        print("No protein sampler sources need to be added.")
        return
    source_paths = {name: discovered[name] for name in new_source_names}
    print(f"New protein sampler sources: {new_source_names}")

    write_stage_metadata(stage_dir / "metadata.json", source_paths, prior_path)
    entity_keys = lmdb_keys(prior_path)
    entity_keys.update(source_entity_keys(source_paths))
    sorted_keys = sorted(entity_keys)
    if args.limit is not None:
        sorted_keys = sorted_keys[: args.limit]
    print(f"Protein prior entities to write: {len(sorted_keys)}")

    build_staging_outputs(
        source_paths=source_paths,
        prior_path=prior_path,
        stage_dir=stage_dir,
        entity_keys=sorted_keys,
        num_workers=args.num_workers,
        apo_map_size_gb=args.apo_map_size_gb,
        prior_map_size_gb=args.prior_map_size_gb,
        commit_interval=args.commit_interval,
    )
    validate_staging(stage_dir, source_paths, sorted_keys)
    if args.no_install:
        print(f"Staged outputs left at {stage_dir}")
        return

    if args.install_lookup:
        build_lookup_files(data_dir, stage_dir, set(new_source_names))
    backup_dir = install_outputs(
        data_dir, stage_dir, new_source_names, args.install_lookup
    )
    print(f"Installed protein seed update. Previous outputs: {backup_dir}")


if __name__ == "__main__":
    main()
