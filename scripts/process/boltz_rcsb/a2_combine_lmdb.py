"""Combine multiple .npz files into a single LMDB database."""

import argparse
import glob
import os
import pathlib

import lmdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple .npz files into a single LMDB database."
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="Directory containing .npz files to combine.",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        required=True,
        help="Output path for the LMDB database.",
    )
    return parser.parse_args()


def create_lmdb_from_npz(npz_dir: str, lmdb_path: str):
    """
    Reads all .npz files from npz_dir and writes them into an LMDB database
    at lmdb_path.

    The key for each entry will be the basename of the file (without .npz).
    The value will be the raw byte content of the .npz file.
    """

    # Find all .npz files in the directory
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))

    if not npz_files:
        print(f"No .npz files found in {npz_dir}")
        return

    print(f"Found {len(npz_files)} .npz files. Starting conversion...")

    map_size = 100 * 1024**3  # 100 GB

    # Open the LMDB environment
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        readonly=False,
        create=True,
        writemap=True,
        map_async=True,
    )

    try:
        # We will do all writes in a single transaction for efficiency
        with env.begin(write=True) as txn:
            # Use tqdm for a progress bar
            for npz_file_path in tqdm(npz_files, desc="Processing files"):
                # Create the key: filename without extension
                # e.g., './my_npz_files/data_001.npz' -> 'data_001'
                key = pathlib.Path(npz_file_path).stem
                key_bytes = key.encode("utf-8")

                # Read the value: raw bytes of the .npz file
                with open(npz_file_path, "rb") as f:
                    value_bytes = f.read()

                # Put (key, value) pair into the transaction
                txn.put(key_bytes, value_bytes)

        print(f"\nSuccessfully created LMDB at {lmdb_path}")
        print(f"Total entries written: {len(npz_files)}")

    except lmdb.MapFullError:
        print(f"Error: LMDB map_size (currently {map_size} bytes) is too small.")
        print("Please increase map_size to a larger value.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    create_lmdb_from_npz(args.npz_dir, args.lmdb_path)
