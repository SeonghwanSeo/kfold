"""Extract RFAM monomer sequences from structure.lmdb."""

import argparse
import io
from pathlib import Path

import lmdb
import numpy as np

from kfold.data.utils.io.fasta import write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the dataset parent directory containing rfam/.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help=(
            "Output FASTA path. Defaults to "
            "{data_dir}/rfam/sequences/rfam_sequences.fasta."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional FASTA line width.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick checks.",
    )
    return parser.parse_args()


def decode_sequence(value: np.ndarray) -> str:
    if value.ndim == 0:
        return value.item().decode("utf-8")
    if value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return "".join(value.astype(str).tolist())


def iter_lmdb_sequences(lmdb_path: Path, limit: int | None = None):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for i, (key_bytes, value_bytes) in enumerate(cursor):
                if limit is not None and i >= limit:
                    break
                with io.BytesIO(value_bytes) as byte_stream:
                    with np.load(byte_stream) as data:
                        yield key_bytes.decode("utf-8"), decode_sequence(data["sequence"])
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / "rfam"
    lmdb_path = data_dir / "structure.lmdb"
    if not lmdb_path.exists():
        raise FileNotFoundError(f"structure.lmdb not found: {lmdb_path}")

    out_path = args.out_path or data_dir / "sequences" / "rfam_sequences.fasta"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sequences = list(iter_lmdb_sequences(lmdb_path, limit=args.limit))
    write_fasta(sequences, out_path, width=args.width)
    print(f"wrote {len(sequences)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
