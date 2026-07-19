"""Combine ENCORE apo token chunk LMDBs by sampler source."""

import argparse
import pathlib
import shutil

import lmdb

DATASET_NAME = "ENCORE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    chunk_root = data_dir / "apo_tok_chunk" / "protein"
    if not chunk_root.exists():
        raise FileNotFoundError(chunk_root)
    out_root = data_dir / "apo_tok_lmdb" / "protein"
    out_root.mkdir(parents=True, exist_ok=True)
    requested = None if args.sources is None else set(args.sources)

    for source_dir in sorted(path for path in chunk_root.iterdir() if path.is_dir()):
        if requested is not None and source_dir.name not in requested:
            continue
        chunks = sorted(source_dir.glob("*.lmdb"))
        if not chunks:
            continue
        out_path = out_root / f"{source_dir.name}.lmdb"
        if out_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out_path} exists. Use --overwrite.")
            shutil.rmtree(out_path)
        env_out = lmdb.open(
            str(out_path),
            map_size=10 * 1024**3,
            meminit=False,
            map_async=True,
            sync=False,
        )
        written = 0
        with env_out.begin(write=True) as txn:
            for chunk in chunks:
                print(f"Combining {chunk}")
                env_in = lmdb.open(str(chunk), readonly=True, lock=False)
                with env_in.begin() as in_txn:
                    for key, value in in_txn.cursor():
                        if not txn.put(key, value, overwrite=False):
                            raise ValueError(f"Duplicate token key: {key.decode()}")
                        written += 1
                env_in.close()
                if args.clean:
                    shutil.rmtree(chunk)
        env_out.sync()
        env_out.close()
        print(f"Wrote {written} entries: {out_path}")


if __name__ == "__main__":
    main()
