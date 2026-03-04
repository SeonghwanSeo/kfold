"""Create apo LMDB for RCSB PDB entries using Multiprocessing."""

import argparse
import io
import multiprocessing as mp
import pathlib

import lmdb
import numpy as np
from tqdm import tqdm

from kfold.data.utils.io.structure import read_protein_structure


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
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of processes to use.",
    )
    args = parser.parse_args()
    return args


def worker(task):
    """Worker function to parse a single PDB file."""
    file_path, apo_type = task
    raw_id = file_path.name.split(".")[0]
    key = f"{apo_type}:{raw_id}"
    try:
        seq, coords = read_protein_structure(file_path)
        seq_arr = np.array(list(seq), dtype=np.dtype("S1"))
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, seq=seq_arr, coords=coords)
            value = buffer.getvalue()
        return key, value
    except Exception as e:
        # Return None to handle errors gracefully in the main loop
        print(f"Error processing {file_path}: {e}")
        return key, None


def main():
    """Main function using multiprocessing pool for heavy parsing tasks."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    apo_dir = data_dir / "apo/"
    out_lmdb = data_dir / "apo.lmdb"

    # Pre-collect all tasks (file paths and their types)
    tasks = []
    for apo_subdir in sorted(apo_dir.iterdir()):
        if not apo_subdir.is_dir():
            continue
        apo_type = apo_subdir.name
        print(f"Collecting files for apo type: {apo_type}")
        suffixes = ("*.pdb", "*.pdb.gz", "*.cif", "*.cif.gz")
        for suffix in suffixes:
            files = sorted(apo_subdir.rglob(suffix))
            for f in files:
                tasks.append((f, apo_type))

    print(f"Total files to process: {len(tasks)}")

    # Open LMDB Environment
    env = lmdb.open(
        str(out_lmdb),
        map_size=20 * 1024 * 1024 * 1024,  # 20 GB for safety
        meminit=False,
        map_async=True,
        sync=False,
    )

    # Use Multiprocessing Pool
    # chunksize controls how many tasks are sent to workers at once
    with mp.Pool(processes=args.num_workers) as pool:
        # Start transaction
        with env.begin(write=True) as txn:
            # imap_unordered yields results as soon as they are ready
            for k, v in tqdm(
                pool.imap_unordered(worker, tasks, chunksize=10), total=len(tasks)
            ):
                if v is not None:
                    txn.put(k.encode("utf-8"), v)

    env.close()
    print("LMDB creation finished successfully.")


if __name__ == "__main__":
    main()
