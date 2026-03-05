"""concatenate multiple files into one file."""

import argparse
import json
import multiprocessing
import pathlib

import lmdb
import msgpack
from tqdm import tqdm

from kfold.data.types.structure import RefStructure


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=100,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
    )

    args = parser.parse_args()

    return args


def process_structure(file):
    key = file.stem
    # Read the npz file as bytes
    with open(file, "rb") as f:
        value_bytes = f.read()
    struct = RefStructure.load_npz(file)
    return key, value_bytes, struct.metadata.to_dict()


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    # Get npz files
    npz_dir: pathlib.Path = data_dir / "npz"
    npz_paths: list[pathlib.Path] = list(npz_dir.rglob("*.npz"))
    print(f"Found {len(npz_paths)} NPZ files in {npz_dir}")

    # Create lmdb environment (expected size of rcsb training set: ~100GB)
    print("Creating LMDB database...")
    metadata_dicts: list[dict] = []
    lmdb_path = data_dir / "structure.lmdb"
    if lmdb_path.exists():
        print(f"LMDB path {lmdb_path} already exists..")
        return

    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,  # size in GB
    )
    txn = env.begin(write=True)
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(process_structure, npz_paths, chunksize=100)
        for key, value_bytes, metadata_dict in tqdm(
            results, total=len(npz_paths), desc="Processing entries"
        ):
            txn.put(key.encode(), value_bytes)
            metadata_dicts.append(metadata_dict)
    txn.commit()
    env.close()

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(metadata_dicts)}")

    metadata_dicts.sort(key=lambda x: x["id"])

    # Save to json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest_all.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    # Save to msgpack file (efficient and fast)
    manifest_path: pathlib.Path = data_dir / "manifest_all.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadata_dicts, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    main()
