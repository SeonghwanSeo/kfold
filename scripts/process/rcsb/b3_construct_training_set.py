"""concatenate multiple files into one file."""

import argparse
import json
import pathlib
import pickle

import lmdb
from tqdm import tqdm

from kfold.data.schema import Metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "--npz_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing preprocessed .npz files.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing training metadata json files, including cluster IDs.",
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

    # Get metadatas
    metadata_dir: pathlib.Path = args.metadata_dir
    metadata_paths = list(metadata_dir.rglob("*.json"))
    metadatas = [Metadata.load_json(p) for p in metadata_paths]
    metadatas.sort(key=lambda x: x.id)
    print(f"Total entries found: {len(metadatas)}")

    # Save metadatas to a single manifest file.
    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]

    # Save to pickle file (efficient)
    with open(out_dir / "manifest.pkl", "wb") as f:
        pickle.dump(metadata_dicts, f)
    print(f"Saved manifest to {out_dir / 'manifest.pkl'}")

    # Save to json file (human-readable; not used in pipeline)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest to {out_dir / 'manifest.json'}")

    # Get npz files
    npz_dir: pathlib.Path = args.npz_dir
    npz_path_dict: dict[str, pathlib.Path] = {p.stem: p for p in npz_dir.rglob("*.npz")}
    print(f"Total NPZ files found: {len(npz_path_dict)}")

    # Verify all metadatas have corresponding npz files
    # NOTE: We assume all metadatas is from previous clustering step, so
    # all entries are extracted from npz files.
    for m in metadatas:
        entry_id = m.id
        npz_path = npz_path_dict.get(entry_id)
        assert npz_path is not None, f"NPZ file not found for {entry_id}"

    # Create lmdb environment (expected size of rcsb training set: ~20GB)
    lmdb_path = out_dir / "structures.lmdb"
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
