"""Concatenate multiple files into one file."""

import argparse
import json
import pathlib
import pickle

import lmdb
from tqdm import tqdm

from kfold.data.types.metadata import Metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Construct training set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    args = parser.parse_args()

    return args


def main():
    """Construct training set from preprocessed data."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get metadatas
    print("Loading metadata files...")
    metadata_dir: pathlib.Path = data_dir / "metadata"
    metadata_paths = list(metadata_dir.rglob("*.json"))
    metadatas = [Metadata.load_json(p) for p in metadata_paths]
    metadatas.sort(key=lambda x: x.id)
    print(f"Total entries found: {len(metadatas)}")

    # Save metadatas to a single manifest file.
    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]

    # Save to pickle file (efficient)
    manifest_path: pathlib.Path = data_dir / "manifest.pkl"
    with open(manifest_path, "wb") as f:
        pickle.dump(metadata_dicts, f)
    print(f"Saved manifest (pickle) to {manifest_path}")

    # Save to json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    # Get npz files
    npz_dir: pathlib.Path = args.data_dir / "npz"
    npz_path_dict: dict[str, pathlib.Path] = {p.stem: p for p in npz_dir.rglob("*.npz")}
    print(f"Total NPZ files found: {len(npz_path_dict)}")

    # Verify all metadatas have corresponding npz files
    # NOTE: We assume all metadatas is from previous clustering step, so
    # all entries are extracted from npz files.
    for m in metadatas:
        entry_id = m.id
        npz_path = npz_path_dict.get(entry_id)
        assert npz_path is not None, f"NPZ file not found for {entry_id}"

    # Create lmdb environment (expected size of rcsb training set: <25GB)
    print("Creating LMDB database...")
    lmdb_path = args.data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=25 * 1024 * 1024 * 1024,
        map_async=True,
    )
    with env.begin(write=True) as txn:
        for m in tqdm(metadatas):
            entry_id = m.id
            key = entry_id.encode()
            npz_path = npz_path_dict.get(entry_id)
            assert npz_path is not None, f"NPZ file not found for {entry_id}"
            # Read the npz file as bytes
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            # Put (key, value) pair into the transaction
            txn.put(key, value_bytes)
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")


if __name__ == "__main__":
    main()
