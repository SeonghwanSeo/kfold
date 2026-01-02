"""concatenate multiple files into one file."""

import argparse
import json
import pathlib
import pickle

import lmdb

from kfold.data.schema import Metadata
from kfold.data.structure import RefStructure


def parse_args():
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "--npz_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing preprocessed .npz files.",
    )
    parser.add_argument(
        "--key_path",
        type=pathlib.Path,
        default="assets/splits/kfold_v251213/validation_ids.txt",
        help="Path to the file containing entry IDs to include.",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        required=True,
        help="Output path for extracted sequences.",
    )
    args = parser.parse_args()

    return args


def main():
    """Main function to extract sequences from npz files."""
    args = parse_args()
    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get entry IDs to include
    with open(args.key_path) as f:
        entry_ids: list[str] = sorted(set(line.strip().lower() for line in f.readlines()))

    # Get npz files
    npz_dir: pathlib.Path = args.npz_dir
    npz_path_dict: dict[str, pathlib.Path] = {p.stem: p for p in npz_dir.rglob("*.npz")}
    metadatas: list[Metadata] = []

    # Create lmdb environment (expected size of rcsb training set: ~20GB)
    lmdb_path = out_dir / "structures.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=25 * 1024 * 1024 * 1024,
    )
    with env.begin(write=True) as txn:
        for entry_id in entry_ids:
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
    with open(out_dir / "manifest.pkl", "wb") as f:
        pickle.dump(metadata_dicts, f)
    print(f"Saved manifest to {out_dir / 'manifest.pkl'}")

    # Save to a json file (human-readable; not used in pipeline)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
