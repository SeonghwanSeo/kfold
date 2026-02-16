import argparse
import io
import multiprocessing
import pathlib

import lmdb
from tqdm import tqdm

from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import write_fasta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        choices=["long", "short"],
        help="Data split to process.",
    )
    args = parser.parse_args()
    return args


def extract_sequence(item):
    """Extract sequence from LMDB value bytes."""
    key, value = item
    with io.BytesIO(value) as f:
        struct = RefStructure.load_npz(f)
    seq = struct.chains[0].get_sequence()
    return key.decode("utf-8"), seq


def main():
    """Main function to extract sequences from AFDB LMDB and write to FASTA."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"

    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    sequences: list[tuple[str, str]] = []
    txn = env.begin()
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        cursor = txn.cursor()
        results = pool.imap_unordered(extract_sequence, cursor, chunksize=100)
        for key, seq in tqdm(
            results, desc="Extracting sequences", total=txn.stat()["entries"]
        ):
            sequences.append((key, seq))

    # Write sequences to FASTA file
    sequences.sort(key=lambda x: x[0])  # sort by key for consistency
    fasta_path = data_dir / "sequences.fasta"
    write_fasta(sequences, fasta_path)
    print(f"Successfully wrote {len(sequences)} sequences to {fasta_path}")


if __name__ == "__main__":
    main()
