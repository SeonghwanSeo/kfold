"""Create source-specific apo LMDBs for disordered PDB apo structures."""

import argparse
import multiprocessing as mp
import pathlib
import shutil
from collections import defaultdict
from dataclasses import dataclass

import lmdb
from tqdm import tqdm

from kfold.data.utils.io.structure import (
    read_dna_structure,
    read_protein_structure,
    read_rna_structure,
)
from kfold.training.dataset.utils.apo_io import pack_apo_record


@dataclass(frozen=True)
class ApoTask:
    path: str
    chain_type: str
    source: str
    key: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
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
        default=128,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing apo_lmdb output.",
    )
    args = parser.parse_args()
    return args


def remove_structure_suffix(path: pathlib.Path) -> str:
    for suffix in (".pdb.zst", ".cif.zst", ".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def collect_tasks(apo_dir: pathlib.Path) -> list[ApoTask]:
    tasks: list[ApoTask] = []
    source_dirs: list[tuple[str, str, pathlib.Path]] = []
    for chain_type in ("protein", "dna", "rna"):
        chain_root = apo_dir / chain_type
        if not chain_root.exists():
            continue
        source_dirs.extend(
            (chain_type, source_dir.name, source_dir)
            for source_dir in sorted(chain_root.iterdir())
            if source_dir.is_dir()
        )

    # Legacy fallback: disordered_pdb/apo/{source}/... is treated as protein.
    source_dirs.extend(
        ("protein", source_dir.name, source_dir)
        for source_dir in sorted(apo_dir.iterdir())
        if source_dir.is_dir() and source_dir.name not in {"protein", "dna", "rna"}
    )

    suffixes = ("*.pdb", "*.pdb.zst", "*.pdb.gz", "*.cif", "*.cif.zst", "*.cif.gz")
    for chain_type, source, source_dir in source_dirs:
        print(f"Collecting files for apo source: {chain_type}/{source}")
        for suffix in suffixes:
            for path in sorted(source_dir.rglob(suffix)):
                tasks.append(
                    ApoTask(
                        path=str(path),
                        chain_type=chain_type,
                        source=source,
                        key=remove_structure_suffix(path),
                    )
                )
    return tasks


def worker(task: ApoTask) -> tuple[str, str, str, bytes | None, str | None]:
    try:
        if task.chain_type == "protein":
            sequence, coords = read_protein_structure(task.path)
        elif task.chain_type == "dna":
            sequence, coords = read_dna_structure(task.path)
        elif task.chain_type == "rna":
            sequence, coords = read_rna_structure(task.path)
        else:
            raise ValueError(f"Unsupported chain type: {task.chain_type}")
        return (
            task.chain_type,
            task.source,
            task.key,
            pack_apo_record(sequence, coords, task.chain_type),
            None,
        )
    except Exception as e:
        return task.chain_type, task.source, task.key, None, f"{task.path}: {e}"


def ensure_overwrite(path: pathlib.Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    apo_dir = data_dir / "apo"
    out_root = data_dir / "apo_lmdb"
    ensure_overwrite(out_root, args.overwrite)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(apo_dir)
    print(f"Total files to process: {len(tasks)}")

    tasks_by_source: dict[tuple[str, str], list[ApoTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_source[(task.chain_type, task.source)].append(task)

    for (chain_type, source), source_tasks in sorted(tasks_by_source.items()):
        out_dir = out_root / chain_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_lmdb = out_dir / f"{source}.lmdb"
        env = lmdb.open(
            str(out_lmdb),
            map_size=args.map_size_gb * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        written = 0
        failed = 0
        errors: list[str] = []
        with mp.Pool(processes=args.num_workers) as pool:
            txn = env.begin(write=True)
            try:
                for _, _, key, value, error in tqdm(
                    pool.imap_unordered(worker, source_tasks, chunksize=16),
                    total=len(source_tasks),
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
            print(f"First {len(errors)} errors while writing {out_lmdb}:")
            for error in errors:
                print(f"  {error}")
        print(
            f"{chain_type}/{source}: written={written}, "
            f"failed={failed}, output={out_lmdb}"
        )


if __name__ == "__main__":
    main()
