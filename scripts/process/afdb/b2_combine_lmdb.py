"""Combine lmdb files"""

import argparse
import pathlib
import shutil

import lmdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        choices=["long", "short"],
        help="Data split to process.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Whether to remove original lmdb files after combining.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"

    combine_lmdb_path = data_dir / "apo_unitok.lmdb"
    env = lmdb.open(
        str(combine_lmdb_path),
        map_size=100 * 1024 * 1024 * 1024,  # 1 GB
        meminit=False,
        map_async=True,
        sync=False,
    )

    chunk_dir = data_dir / "apo_unitok_chunk/"
    t = 0
    lmdb_paths = sorted(chunk_dir.glob("*.lmdb"))
    with env.begin(write=True) as txn:
        for lmdb_path in lmdb_paths:
            print(f"Combining {lmdb_path}...")
            sub_env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
            with sub_env.begin() as sub_txn:
                cursor = sub_txn.cursor()
                for key, value in cursor:
                    new_key = f"{key.decode()}"
                    txn.put(new_key.encode(), value, overwrite=False)
                    t += 1
            sub_env.close()
            if args.clean:
                print(f"Removing {lmdb_path}...")
                shutil.rmtree(lmdb_path)
    env.close()
    print(f"Combined {t} entries into {combine_lmdb_path}.")


if __name__ == "__main__":
    main()
