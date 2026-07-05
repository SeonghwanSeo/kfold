"""Create RFAM RNA prior stack LMDB from RNA sampler outputs.

The RFAM training set should not use apo structures for trunk input.  This
script only prepares diffusion-bridge priors in the shared
`prior_lmdb/rna.lmdb` layout.  Each record is stored under `{sample_id}_1` so
the existing monomer dataset prior lookup can reuse the entity-level loader.
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

from kfold.data.utils.io.apo import pack_prior_stack_record
from kfold.data.utils.io.structure import read_rna_structure


@dataclass(frozen=True)
class PriorTask:
    sample_id: str
    paths: tuple[str, ...]
    sample_names: tuple[str, ...]


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
        help="Overwrite existing prior_lmdb/rna.lmdb.",
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


def collect_tasks(data_dir: Path, prior_dir: Path, limit: int | None) -> list[PriorTask]:
    sample_ids = load_manifest_ids(data_dir / "manifest.msgpack")
    if limit is not None:
        sample_ids = sample_ids[:limit]

    tasks: list[PriorTask] = []
    missing = 0
    for sample_id in sample_ids:
        sample_dir = sample_dir_for(prior_dir, sample_id)
        paths = collect_structure_files(sample_dir)
        if not paths:
            missing += 1
            continue
        tasks.append(
            PriorTask(
                sample_id=sample_id,
                paths=tuple(str(path) for path in paths),
                sample_names=tuple(remove_structure_suffix(path) for path in paths),
            )
        )

    print(f"manifest samples: {len(sample_ids)}")
    print(f"samples with prior structures: {len(tasks)}")
    print(f"samples missing prior structures: {missing}")
    return tasks


def build_record(task: PriorTask) -> tuple[str, bytes] | tuple[str, str, str]:
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


def write_lmdb(
    tasks: list[PriorTask],
    out_path: Path,
    map_size_gb: int,
    num_workers: int,
) -> tuple[int, list[tuple[str, str, str]]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_path), map_size=map_size_gb * 1024**3)
    failures: list[tuple[str, str, str]] = []
    n_written = 0
    try:
        with env.begin(write=True) as txn:
            with mp.Pool(processes=num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(build_record, tasks), total=len(tasks)
                ):
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

    out_path = data_dir / "prior_lmdb" / "rna.lmdb"
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path}")
        shutil.rmtree(out_path)

    n_written, failures = write_lmdb(
        tasks,
        out_path=out_path,
        map_size_gb=args.map_size_gb,
        num_workers=args.num_workers,
    )
    print(f"wrote {n_written} records -> {out_path}")
    print(f"failures: {len(failures)}")
    for sample_id, err_type, message in failures[:20]:
        print(f"  {sample_id}: {err_type}: {message}")


if __name__ == "__main__":
    main()
