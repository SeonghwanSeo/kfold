"""concatenate multiple files into one file."""

import argparse
import pathlib

import lmdb


def parse_args():
    parser = argparse.ArgumentParser(description="Construct validation set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    args = parser.parse_args()

    return args


def main():
    """Main function to extract sequences from npz files."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb-val"

    # Get entry IDs to include
    print("Loading entry IDs...")
    key_path: pathlib.Path = data_dir / "validation_ids.txt"
    print(key_path.absolute())
    with open(key_path) as f:
        entry_ids: list[str] = [line.strip().lower() for line in f.readlines()]
    assert len(entry_ids) == len(set(entry_ids)), "Duplicate entry IDs found!"
    print(f"Total entry IDs to include: {len(entry_ids)}")

    # Create lmdb environment (expected size of rcsb training set: ~20GB)
    print("Creating LMDB database...")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=1024 * 1024 * 1024,  # 1 GB
    )
    npz_dir: pathlib.Path = data_dir / "npz"
    with env.begin(write=True) as txn:
        for entry_id in entry_ids:
            key = entry_id.encode()
            npz_path = npz_dir / entry_id[1:3] / f"{entry_id}.npz"
            assert npz_path.exists(), f"NPZ file not found: {npz_path}"
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            txn.put(key, value_bytes)
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(entry_ids)}")


if __name__ == "__main__":
    main()
