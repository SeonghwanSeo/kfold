import argparse
import io
import logging
import multiprocessing
import os
from pathlib import Path

import lmdb
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.tokenized import TokenizedStructure

logger = logging.getLogger(__name__)

# Global variable for the worker process to hold the LMDB environment
_env = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get polymer sequences from processed LMDB dataset (Multiprocessing)."
    )
    parser.add_argument(
        "--processed_lmdb_path",
        type=Path,
        help="Path to the processed LMDB dataset.",
        default="/cache/wykim_lab/kfold_data/structures/kfold_rcsb_processed_v251120.lmdb/",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of directory to save the sequences (fasta).",
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_polymer_sequences.fasta",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes. Defaults to CPU count.",
        default=len(os.sched_getaffinity(0)),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of keys to process per batch.",
        default=500,
    )
    return parser.parse_args()


def init_worker(lmdb_path: str):
    """
    Initialize the worker process by opening the LMDB environment.
    This runs once per process.
    """
    global _env
    _env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def process_batch(
    keys: list[bytes],
) -> list[tuple[str, list[tuple[int, C.ChainType, str]]]]:
    """
    Process a batch of keys. Returns a list of (pdb_id, polymer_sequences).
    """
    global _env
    results = []

    with _env.begin(write=False) as txn:
        for key in keys:
            pdb_id = key.decode()
            value_bytes = txn.get(key)

            if value_bytes is None:
                logger.warning(f"Key {pdb_id} not found in LMDB. Skipping.")
                continue

            try:
                # Use io.BytesIO to wrap the raw bytes
                with io.BytesIO(value_bytes) as byte_stream:
                    struct: TokenizedStructure = TokenizedStructure.load_npz(byte_stream)

                entity_sequences: dict[int, tuple[C.ChainType, str]] = {}
                num_chains = struct.num_chains

                for chain_i in range(num_chains):
                    entity_id: int = int(struct.chain.entity_id[chain_i])
                    # If we already processed this entity, skip
                    if entity_id in entity_sequences:
                        continue

                    ctype = C.ChainType(int(struct.chain.chain_type[chain_i]))

                    # NOTE: Currently, we skip ligands since it is impossible to
                    # distinguish small molecule ligands from metals.
                    if ctype is C.ChainType.LIGAND:
                        continue

                    residue_st = int(struct.chain.residue_start[chain_i])
                    residue_end = residue_st + int(struct.chain.num_residues[chain_i])

                    # Get sequence
                    sequence_res_names = struct.residue.name[residue_st:residue_end]
                    seq_chars = [
                        C.residue.get_one_letter(str(res_name).strip(), ctype)
                        for res_name in sequence_res_names
                    ]
                    sequence: str = "".join(seq_chars)
                    entity_sequences[entity_id] = (ctype, sequence)

                # Sort by entity_id for deterministic output
                polymer_sequences: list[tuple[int, C.ChainType, str]] = []
                for entity_id in sorted(entity_sequences.keys()):
                    ctype, sequence = entity_sequences[entity_id]
                    polymer_sequences.append((entity_id, ctype, sequence))

                results.append((pdb_id, polymer_sequences))

            except Exception as e:
                logger.error(f"Error processing {pdb_id}: {e}")
                continue

    return results


def get_all_keys(lmdb_path: str) -> list[bytes]:
    """Quickly retrieve all keys from the LMDB."""
    logger.info("Reading all keys from LMDB...")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        # keys=True, values=False is much faster just to list keys
        with txn.cursor() as cursor:
            keys = [key for key in cursor.iternext(keys=True, values=False)]
    env.close()
    logger.info(f"Found {len(keys)} keys.")
    return keys


def main(args):
    lmdb_path_str = str(args.processed_lmdb_path)

    # 1. Get all keys first (Main Process)
    all_keys = get_all_keys(lmdb_path_str)

    # 2. Chunk keys for batch processing
    # Processing in batches reduces pickling overhead
    chunk_size = args.batch_size
    key_chunks = [
        all_keys[i : i + chunk_size] for i in range(0, len(all_keys), chunk_size)
    ]

    # 3. Setup Multiprocessing Pool
    logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

    # Prepare output directory
    output_file = args.output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(
        processes=args.num_workers, initializer=init_worker, initargs=(lmdb_path_str,)
    ) as pool:
        with open(output_file, "w") as f:
            for batch_results in tqdm(
                pool.imap(process_batch, key_chunks),
                total=len(key_chunks),
                desc="Extracting sequences",
            ):
                for pdb_id, polymer_seqs in batch_results:
                    for entity_id, ctype, sequence in polymer_seqs:
                        chain_type_str = ctype.name.lower()
                        f.write(f">{pdb_id}_{entity_id}_{chain_type_str}\n")
                        f.write(f"{sequence}\n")

    logger.info(f"Done. Saved to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    main(args)
