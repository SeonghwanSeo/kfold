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
        choices=["train", "val", "test"],
        help="Data split to process (train/val/test).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Whether to remove original lmdb files after combining.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional apo source names to combine, e.g. esmfold afdb.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"

    for token_type in ("protein", "protein_multimer"):
        apo_tok_dir = data_dir / "apo_tok_chunk" / token_type
        if not apo_tok_dir.exists():
            continue
        out_root = data_dir / "apo_tok_lmdb" / token_type
        out_root.mkdir(parents=True, exist_ok=True)
        for apo_subdir in sorted(apo_tok_dir.iterdir()):
            if args.sources is not None and apo_subdir.name not in set(args.sources):
                continue
            source = apo_subdir.name
            print(f"Processing {apo_subdir} ({source})...")
            lmdb_paths = sorted(apo_subdir.glob("*.lmdb"))
            combine_lmdb_path = out_root / f"{source}.lmdb"
            if combine_lmdb_path.exists():
                shutil.rmtree(combine_lmdb_path)
            env = lmdb.open(
                str(combine_lmdb_path),
                map_size=10 * 1024 * 1024 * 1024,  # 10 GB
                meminit=False,
                map_async=True,
                sync=False,
            )
            with env.begin(write=True) as txn:
                for lmdb_path in lmdb_paths:
                    print(f"Combining {lmdb_path}...")
                    sub_env = lmdb.open(
                        str(lmdb_path), readonly=True, lock=False, readahead=True
                    )
                    with sub_env.begin() as sub_txn:
                        cursor = sub_txn.cursor()
                        for key, value in cursor:
                            txn.put(key, value, overwrite=False)
                    sub_env.close()
                    if args.clean:
                        print(f"Removing {lmdb_path}...")
                        shutil.rmtree(lmdb_path)
            env.close()


if __name__ == "__main__":
    main()
