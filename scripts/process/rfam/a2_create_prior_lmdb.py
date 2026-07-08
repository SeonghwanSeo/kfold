"""Create RFAM RNA apo/prior LMDBs and apo lookup from RNA sampler outputs.

Each record is stored under `{sample_id}_1` so the monomer dataset can reuse
the entity-level apo/prior loaders.
"""

import argparse
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from kfold.data.utils.io.structure import read_rna_structure
from kfold.training.dataset.utils.apo_io import pack_apo_record, pack_prior_stack_record


@dataclass(frozen=True)
class PriorTask:
    sample_id: str
    paths: tuple[str, ...]
    sample_names: tuple[str, ...]
    apo_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the dataset parent directory containing rfam/.",
    )
    parser.add_argument(
        "--source",
        default="rna_sampler_seed1_step50",
        help="RNA sampler source directory name under apo/rna/.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parser workers.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=256,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing apo_lmdb/prior_lmdb outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Collect inputs and print a summary without writing LMDB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick checks.",
    )
    return parser.parse_args()


def load_manifest_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open("rb") as f:
        manifest = msgpack.unpack(f, raw=False)
    return [record["id"] for record in manifest]


def remove_structure_suffix(path: Path) -> str:
    return path.name.removesuffix(".pdb.zst")


def sample_dir_for(prior_dir: Path, sample_id: str) -> Path:
    return prior_dir / sample_id


def collect_structure_files(sample_dir: Path) -> list[Path]:
    if not sample_dir.exists():
        return []
    files = [
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.name.endswith(".pdb.zst")
    ]
    return sorted(
        path
        for path in files
        if "_sample_" in path.name and path.name.endswith("_model.pdb.zst")
    )


def collect_apo_file(sample_dir: Path) -> Path | None:
    """Find one representative RNA apo structure for trunk input."""
    preferred_names = (
        "ranked_model.pdb.zst",
        "ranked.pdb.zst",
        "best_model.pdb.zst",
        "best.pdb.zst",
    )
    for name in preferred_names:
        path = sample_dir / name
        if path.exists():
            return path

    ranked = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file()
        and path.name.endswith(".pdb.zst")
        and ("rank" in path.name or "best" in path.name)
    )
    if ranked:
        return ranked[0]
    return None


def collect_tasks(data_dir: Path, prior_dir: Path, limit: int | None) -> list[PriorTask]:
    sample_ids = load_manifest_ids(data_dir / "manifest.msgpack")
    if limit is not None:
        sample_ids = sample_ids[:limit]

    tasks: list[PriorTask] = []
    missing = 0
    missing_apo = 0
    for sample_id in sample_ids:
        sample_dir = sample_dir_for(prior_dir, sample_id)
        paths = collect_structure_files(sample_dir)
        if not paths:
            missing += 1
            continue
        apo_path = collect_apo_file(sample_dir)
        if apo_path is None:
            missing_apo += 1
        tasks.append(
            PriorTask(
                sample_id=sample_id,
                paths=tuple(str(path) for path in paths),
                sample_names=tuple(remove_structure_suffix(path) for path in paths),
                apo_path=None if apo_path is None else str(apo_path),
            )
        )

    print(f"manifest samples: {len(sample_ids)}")
    print(f"samples with prior structures: {len(tasks)}")
    print(f"samples missing prior structures: {missing}")
    print(f"samples missing apo representative: {missing_apo}")
    return tasks


def build_prior_record(task: PriorTask) -> tuple[str, bytes] | tuple[str, str, str]:
    try:
        seq: str | None = None
        coords_list: list[np.ndarray] = []
        for path in task.paths:
            sample_seq, coords = read_rna_structure(path)
            if seq is None:
                seq = sample_seq
            elif sample_seq != seq:
                raise ValueError(
                    f"sequence mismatch in {task.sample_id}: "
                    f"{len(sample_seq)} != {len(seq)} or bases differ"
                )
            coords_list.append(coords)

        assert seq is not None
        stack = np.stack(coords_list, axis=0)
        key = f"{task.sample_id}_1"
        value = pack_prior_stack_record(
            sequence=seq,
            coords=stack,
            chain_type="rna",
            sample_names=list(task.sample_names),
        )
        return key, value
    except Exception as e:
        return task.sample_id, type(e).__name__, str(e)


def build_apo_record(task: PriorTask) -> tuple[str, bytes] | tuple[str, str, str]:
    if task.apo_path is None:
        return task.sample_id, "MissingApo", "no representative apo structure found"
    try:
        seq, coords = read_rna_structure(task.apo_path)
        key = f"{task.sample_id}_1"
        value = pack_apo_record(sequence=seq, coords=coords, chain_type="rna")
        return key, value
    except Exception as e:
        return task.sample_id, type(e).__name__, str(e)


def write_lmdb(
    tasks: list[PriorTask],
    out_path: Path,
    map_size_gb: int,
    num_workers: int,
    worker,
) -> tuple[int, list[tuple[str, str, str]]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_path), map_size=map_size_gb * 1024**3)
    failures: list[tuple[str, str, str]] = []
    n_written = 0
    try:
        with env.begin(write=True) as txn:
            with mp.Pool(processes=num_workers) as pool:
                for result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
                    if len(result) == 3:
                        failures.append(result)  # type: ignore[arg-type]
                        continue
                    key, value = result
                    txn.put(key.encode("utf-8"), value)
                    n_written += 1
    finally:
        env.sync()
        env.close()
    return n_written, failures


def build_apo_lookup(
    tasks: list[PriorTask],
    failures: list[tuple[str, str, str]],
    source: str,
) -> dict[str, dict[str, list[dict]]]:
    failed_sample_ids = {sample_id for sample_id, _, _ in failures}
    lookup: dict[str, dict[str, list[dict]]] = {}
    for task in tasks:
        if task.apo_path is None or task.sample_id in failed_sample_ids:
            continue
        lookup[task.sample_id] = {
            "1": [
                {
                    "source": source,
                    "name": f"{task.sample_id}_1",
                    "chain_type": "rna",
                }
            ]
        }
    return lookup


def remove_existing_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / "rfam"
    prior_dir = data_dir / "apo" / "rna" / args.source
    if not prior_dir.exists():
        raise FileNotFoundError(
            f"RNA sampler source directory not found: {prior_dir}. "
            "Extract the sampler archive before running this script."
        )
    tasks = collect_tasks(data_dir, prior_dir, args.limit)
    if args.dry_run:
        return

    prior_out_path = data_dir / "prior_lmdb" / "rna.lmdb"
    apo_out_path = data_dir / "apo_lmdb" / "rna" / f"{args.source}.lmdb"
    apo_lookup_path = data_dir / "apo_lookup.msgpack"
    for out_path in (prior_out_path, apo_out_path, apo_lookup_path):
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path}")
    for out_path in (prior_out_path, apo_out_path, apo_lookup_path):
        if not args.overwrite:
            continue
        if out_path.exists():
            remove_existing_output(out_path)

    n_written, failures = write_lmdb(
        tasks,
        out_path=prior_out_path,
        map_size_gb=args.map_size_gb,
        num_workers=args.num_workers,
        worker=build_prior_record,
    )
    print(f"wrote {n_written} prior records -> {prior_out_path}")
    print(f"prior failures: {len(failures)}")
    for sample_id, err_type, message in failures[:20]:
        print(f"  {sample_id}: {err_type}: {message}")

    n_written, failures = write_lmdb(
        tasks,
        out_path=apo_out_path,
        map_size_gb=args.map_size_gb,
        num_workers=args.num_workers,
        worker=build_apo_record,
    )
    print(f"wrote {n_written} apo records -> {apo_out_path}")
    print(f"apo failures: {len(failures)}")
    for sample_id, err_type, message in failures[:20]:
        print(f"  {sample_id}: {err_type}: {message}")

    apo_lookup = build_apo_lookup(tasks, failures, args.source)
    with apo_lookup_path.open("wb") as f:
        msgpack.pack(apo_lookup, f)
    print(f"wrote {len(apo_lookup)} apo lookup entries -> {apo_lookup_path}")


if __name__ == "__main__":
    main()
