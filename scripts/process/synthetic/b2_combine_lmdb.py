"""Combine lmdb files"""

import argparse
import pathlib

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
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    combine_lmdb_path = data_dir / "apo_unitok.lmdb"
    env = lmdb.open(
        str(combine_lmdb_path),
        map_size=1 * 1024 * 1024 * 1024,  # 1 GB
        meminit=False,
        map_async=True,
        sync=False,
    )

    chunk_dir = data_dir / "apo_unitok_chunk/"
    lmdb_paths = sorted(chunk_dir.glob("*.lmdb"))
    cnt = 0
    with env.begin(write=True) as txn:
        for lmdb_path in lmdb_paths:
            print(f"Combining {lmdb_path}...")
            sub_env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
            with sub_env.begin() as sub_txn:
                cursor = sub_txn.cursor()
                for key, value in cursor:
                    txn.put(key, value)
                    cnt += 1
            sub_env.close()
    env.close()
    print(
        f"Combined {len(lmdb_paths)} lmdb files with "
        f"total {cnt} entries into {combine_lmdb_path}"
    )


if __name__ == "__main__":
    main()
