"""Create apo_complex.lmdb from archived multimer apo structures."""

import argparse
import multiprocessing as mp
import pathlib
import re

import lmdb
from tqdm import tqdm

from kfold.data.utils.io.apo import pack_apo_complex_record
from kfold.data.utils.io.structure import read_protein_multimer_structure

SAMPLE_RE = re.compile(r"^protein_multimer_sample_(?P<sample_id>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build apo_complex.lmdb from "
            "apo_complex/{kind}/{shard}/{pdb_id}/{group_id}/"
            "protein_multimer_sample_*.pdb[.zst] archives."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to processed dataset root.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="RCSB split to process.",
    )
    parser.add_argument(
        "--kind",
        default="ab",
        help="Subdirectory under apo_complex/ to scan, e.g. ab or protein.",
    )
    parser.add_argument(
        "--source",
        default="ab_complex",
        help="LMDB source prefix for keys.",
    )
    parser.add_argument(
        "--archive_dir",
        type=pathlib.Path,
        default=None,
        help="Optional apo_complex archive root. Defaults to rcsb-*/apo_complex.",
    )
    parser.add_argument(
        "--out_lmdb",
        type=pathlib.Path,
        default=None,
        help="Optional output LMDB path. Defaults to rcsb-*/apo_complex.lmdb.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=128,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parser workers.",
    )
    parser.add_argument(
        "--commit_interval",
        type=int,
        default=1000,
        help="Commit LMDB transaction every N records.",
    )
    return parser.parse_args()


def strip_structure_suffix(path: pathlib.Path) -> str:
    name = path.name
    for suffix in (".zst", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for suffix in (".pdb", ".cif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def parse_archive_path(path: pathlib.Path) -> tuple[str, str, str]:
    """Return pdb_id, group_id, and sample_id from an archive path."""
    group_id = path.parent.name
    pdb_id = path.parent.parent.name
    stem = strip_structure_suffix(path)
    match = SAMPLE_RE.match(stem)
    if match is None:
        raise ValueError(f"Unexpected apo complex file name: {path.name}")
    return pdb_id, group_id, match.group("sample_id")


def collect_archives(archive_root: pathlib.Path, kind: str) -> list[pathlib.Path]:
    kind_dir = archive_root / kind
    if not kind_dir.exists():
        return []

    paths: list[pathlib.Path] = []
    patterns = (
        "protein_multimer_sample_*.pdb",
        "protein_multimer_sample_*.pdb.gz",
        "protein_multimer_sample_*.pdb.zst",
        "protein_multimer_sample_*.cif",
        "protein_multimer_sample_*.cif.gz",
        "protein_multimer_sample_*.cif.zst",
    )
    for pattern in patterns:
        paths.extend(kind_dir.rglob(pattern))
    return sorted(paths)


def worker(task: tuple[pathlib.Path, str]) -> tuple[str, bytes | None, str | None]:
    path, source = task
    try:
        pdb_id, group_id, sample_id = parse_archive_path(path)
        chains = read_protein_multimer_structure(path)
        if not chains:
            raise ValueError(f"No protein chains found in {path}")
        key = f"{source}:{pdb_id}_{group_id}_sample{sample_id}"
        return key, pack_apo_complex_record(chains), None
    except Exception as e:
        return str(path), None, str(e)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    archive_root = args.archive_dir or data_dir / "apo_complex"
    out_lmdb = args.out_lmdb or data_dir / "apo_complex.lmdb"

    paths = collect_archives(archive_root, args.kind)
    print(f"Found {len(paths)} apo complex archives under {archive_root / args.kind}")
    if not paths:
        print("No apo complex archives found. Nothing to write.")
        return

    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(out_lmdb),
        map_size=args.map_size_gb * 1024**3,
        meminit=False,
        map_async=True,
        sync=False,
    )

    written = 0
    failed = 0
    seen_keys: set[str] = set()
    txn = env.begin(write=True)
    try:
        with mp.Pool(processes=args.num_workers) as pool:
            tasks = ((path, args.source) for path in paths)
            for key, value, error in tqdm(
                pool.imap_unordered(worker, tasks, chunksize=8), total=len(paths)
            ):
                if value is None:
                    failed += 1
                    print(f"Error processing {key}: {error}")
                    continue
                if key in seen_keys:
                    failed += 1
                    print(f"Duplicate apo complex key skipped: {key}")
                    continue
                seen_keys.add(key)
                txn.put(key.encode("utf-8"), value)
                written += 1
                if written % args.commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.close()

    print(f"LMDB creation finished: written={written}, failed={failed}")
    print(f"Output: {out_lmdb}")


if __name__ == "__main__":
    main()
