"""Combine AFDB heterodimer apo token chunk LMDBs into source-specific LMDBs."""

import argparse
import pathlib
import shutil

import lmdb

DATASET_NAME = "AFDB-heterodimer"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help=f"Root preprocessed data directory. {DATASET_NAME}/ is appended.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove chunk LMDBs after combining.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional apo source names to combine.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    apo_tok_dir = data_dir / "apo_tok_chunk" / "protein"
    if not apo_tok_dir.exists():
        raise FileNotFoundError(apo_tok_dir)

    out_root = data_dir / "apo_tok_lmdb" / "protein"
    out_root.mkdir(parents=True, exist_ok=True)
    requested_sources = None if args.sources is None else set(args.sources)

    for source_dir in sorted(path for path in apo_tok_dir.iterdir() if path.is_dir()):
        if requested_sources is not None and source_dir.name not in requested_sources:
            continue
        source = source_dir.name
        print(f"Processing {source_dir} ({source})...")
        lmdb_paths = sorted(source_dir.glob("*.lmdb"))
        if not lmdb_paths:
            print(f"No chunk LMDBs found for {source}. Skipping.")
            continue

        out_lmdb_path = out_root / f"{source}.lmdb"
        if out_lmdb_path.exists():
            shutil.rmtree(out_lmdb_path)
        env = lmdb.open(
            str(out_lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,
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
                    for key, value in sub_txn.cursor():
                        txn.put(key, value, overwrite=False)
                sub_env.close()
                if args.clean:
                    print(f"Removing {lmdb_path}...")
                    shutil.rmtree(lmdb_path)
        env.close()


if __name__ == "__main__":
    main()
