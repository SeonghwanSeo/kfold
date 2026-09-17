# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Concatenate multiple files into one file."""

import argparse
import pathlib

import lmdb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Construct training set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    args = parser.parse_args()

    return args


def main():
    """Construct training set from preprocessed data."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb-train"
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_dir: pathlib.Path = data_dir / "npz"
    npz_files = sorted(npz_dir.rglob("*.npz"))
    print(f"Total NPZ files found: {len(npz_files)}")
    # Create lmdb environment (expected size of rcsb training set: <30GB)
    print("Creating LMDB database...")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=50 * 1024 * 1024 * 1024,
    )
    with env.begin(write=True) as txn:
        for path in tqdm(npz_files, desc="Writing NPZ files to LMDB"):
            key = path.stem.encode("utf-8")
            with open(path, "rb") as f:
                value_bytes = f.read()
            txn.put(key, value_bytes)
    env.close()
    print(f"Successfully created LMDB at {lmdb_path}")


if __name__ == "__main__":
    main()
