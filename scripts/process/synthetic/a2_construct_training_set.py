"""concatenate multiple files into one file."""

import argparse
import json
import pathlib
import pickle

import lmdb
from tqdm import tqdm

from kfold.data.types.metadata import Metadata
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
        "--split",
        choices=["short", "long"],
        default="long",
        help="Data split to process (short or long sequences).",
    )
    parser.add_argument(
        "--size_gb",
        type=int,
        default=200,
        help="LMDB map size in GB.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    # Get npz files
    npz_dir: pathlib.Path = data_dir / "npz"
    npz_path_dict: dict[str, pathlib.Path] = {p.stem: p for p in npz_dir.rglob("*.npz")}

    # Create lmdb environment (expected size of rcsb training set: ~20GB)
    print("Creating LMDB database...")
    metadatas: list[Metadata] = []
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.size_gb * 1024 * 1024 * 1024,  # size in GB
    )
    with env.begin(write=True) as txn:
        for entry_id in tqdm(sorted(npz_path_dict.keys()), desc="Processing entries"):
            key = entry_id.encode()
            npz_path = npz_path_dict.get(entry_id)
            if npz_path is None:
                print(f"Warning: NPZ file not found for {entry_id}, skipping.")
                continue
            # Read the npz file as bytes
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            # Put (key, value) pair into the transaction
            txn.put(key, value_bytes)
            # Load structure to get metadata
            # WARN: this does not include the cluster ID info.
            struct = RefStructure.load_npz(npz_path)
            metadatas.append(struct.metadata)
    env.close()

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(metadatas)}")

    # Save metadatas to a single manifest file.
    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]

    # Save to a pickle file (efficient)
    manifest_path: pathlib.Path = data_dir / "manifest.pkl"
    with open(manifest_path, "wb") as f:
        pickle.dump(metadata_dicts, f)
    print(f"Saved manifest (pickle) to {manifest_path}")

    # Save to a json file (human-readable; not used in pipeline)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")


if __name__ == "__main__":
    main()
