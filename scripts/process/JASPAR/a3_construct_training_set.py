"""Pack JASPAR NPZ files into structure.lmdb and manifest files."""

import argparse
import json
import multiprocessing
import pathlib

import lmdb
import msgpack
from tqdm import tqdm

from kfold.data.types.structure import RefStructure

DATASET_NAME = "JASPAR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The JASPAR/ folder is appended.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=256,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing structure.lmdb and manifest files.",
    )
    return parser.parse_args()


def process_npz(path: pathlib.Path):
    key = path.stem
    value = path.read_bytes()
    metadata = RefStructure.load_npz(path).metadata.to_dict()
    return key, value, metadata


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    npz_paths = sorted((data_dir / "npz").glob("*.npz"))
    print(f"Found NPZ files: {len(npz_paths)}")

    lmdb_path = data_dir / "structure.lmdb"
    if lmdb_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{lmdb_path} exists. Use --overwrite.")
        import shutil

        shutil.rmtree(lmdb_path)

    env = lmdb.open(str(lmdb_path), map_size=args.map_size_gb * 1024**3)
    txn = env.begin(write=True)
    metadata_dicts: list[dict] = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(process_npz, npz_paths, chunksize=100)
        for key, value, metadata in tqdm(
            results, total=len(npz_paths), desc="Writing JASPAR structure LMDB"
        ):
            txn.put(key.encode(), value)
            metadata_dicts.append(metadata)
            if len(metadata_dicts) % 10_000 == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()
    env.close()

    metadata_dicts.sort(key=lambda x: x["id"])
    for suffix, writer in (
        ("json", lambda obj, f: json.dump(obj, f, indent=2)),
        ("msgpack", lambda obj, f: msgpack.pack(obj, f)),
    ):
        path = data_dir / f"manifest.{suffix}"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite.")
        mode = "wb" if suffix == "msgpack" else "w"
        with path.open(mode) as f:
            writer(metadata_dicts, f)
        print(f"Wrote {path}")

    print(f"LMDB entries written: {len(metadata_dicts)}")


if __name__ == "__main__":
    main()
