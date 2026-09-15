"""Combine complete token shards and verify coverage against their apo source."""

import argparse
import shutil
import tempfile
from itertools import zip_longest
from pathlib import Path

import lmdb


def combine_source(
    dataset_dir: Path,
    kind: str,
    source: str,
    num_chunk: int,
    overwrite: bool = False,
    clean: bool = False,
):
    if num_chunk < 1:
        raise ValueError("num_chunk must be positive")
    chunk_dir = dataset_dir / "apo_tok_chunk" / kind / source
    chunks = [chunk_dir / f"{index}_{num_chunk}.lmdb" for index in range(num_chunk)]
    missing = [str(path) for path in chunks if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing token shards: {missing}")
    output = dataset_dir / "apo_tok_lmdb" / kind / f"{source}.lmdb"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output}; use --overwrite to rebuild")
    apo_path = dataset_dir / "apo_lmdb" / kind / f"{source}.lmdb"
    if not apo_path.exists():
        raise FileNotFoundError(apo_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokens-", dir=output.parent) as temporary:
        staged = Path(temporary) / "tokens.lmdb"
        with lmdb.open(str(staged), map_size=10 * 1024**3) as env:
            for path in chunks:
                with lmdb.open(str(path), readonly=True, lock=False) as shard:
                    with shard.begin() as reader, env.begin(write=True) as writer:
                        for key, value in reader.cursor():
                            if not writer.put(key, value, overwrite=False):
                                raise ValueError(f"Duplicate token key: {key.decode()}")
            with lmdb.open(str(apo_path), readonly=True, lock=False) as apo:
                with apo.begin() as expected, env.begin() as actual:
                    expected_keys = expected.cursor().iternext(values=False)
                    actual_keys = actual.cursor().iternext(values=False)
                    if any(a != b for a, b in zip_longest(expected_keys, actual_keys)):
                        raise ValueError(
                            f"Token keys do not match apo source: {kind}/{source}"
                        )
        if output.exists():
            shutil.rmtree(output)
        staged.rename(output)
    if clean:
        for path in chunks:
            shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num_chunk", type=int, default=1)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.num_chunk < 1:
        parser.error("num_chunk must be positive")
    dataset_dir = args.data_dir / f"rcsb-{args.split}"
    inputs = [
        (kind, path.stem)
        for kind in ("protein", "protein-multimer")
        for path in sorted((dataset_dir / "apo_lmdb" / kind).glob("*.lmdb"))
    ]
    if args.sources:
        missing = set(args.sources) - {source for _, source in inputs}
        if missing:
            raise FileNotFoundError(f"Requested sources not found: {sorted(missing)}")
        inputs = [(kind, source) for kind, source in inputs if source in args.sources]
    for kind, source in inputs:
        combine_source(
            dataset_dir, kind, source, args.num_chunk, args.overwrite, args.clean
        )
        print(f"Combined {kind}/{source}")


if __name__ == "__main__":
    main()
