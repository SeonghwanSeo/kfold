import argparse
import io
import logging
import multiprocessing
import os
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

import kfold.constants as C
from kfold.data.metadata import Metadata
from kfold.data.structure import TokenizedStructure
from kfold.training.folding.dataset.datamodule import load_manifest

logger = logging.getLogger(__name__)

# Global variable for the worker process to hold the LMDB environment
_env: lmdb.Environment | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get protein structures from processed LMDB dataset."
    )
    parser.add_argument(
        "--processed_lmdb_path",
        type=Path,
        help="Path to the processed LMDB dataset.",
        default="/cache/wykim_lab/kfold_data/kfold_rcsb_processed_v251120.lmdb/",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        help="Path of manifest file.",
        default="/cache/wykim_lab/kfold_data/manifests/af3_manifest.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path of directory to save the extracted protein structures.",
        default="/cache/wykim_lab/icl_shwan/kfold/rcsb_holo_chains/",
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


def process_batch(records: list[Metadata], save_dir: Path):
    """Process a batch of records to extract protein structures."""
    global _env
    assert _env is not None, "LMDB environment is not initialized."

    with _env.begin(write=False) as txn:
        for record in records:
            pdb_id = record.id
            value_bytes = txn.get(pdb_id.encode("utf-8"))

            pdb_save_dir = save_dir / pdb_id[1:3] / pdb_id
            pdb_save_dir.mkdir(parents=True, exist_ok=True)

            if value_bytes is None:
                logger.warning(f"Key {pdb_id} not found in LMDB. Skipping.")
                continue

            try:
                # Use io.BytesIO to wrap the raw bytes
                with io.BytesIO(value_bytes) as byte_stream:
                    struct: TokenizedStructure = TokenizedStructure.load_npz(byte_stream)
            except Exception as e:
                logger.error(f"Error processing {pdb_id}: {e}")
                continue

            visited_entities: set[int] = set()
            for chain_i in range(struct.num_chains):
                entity_id = int(struct.chain.entity_id[chain_i])
                if entity_id in visited_entities:
                    continue  # Already processed
                visited_entities.add(entity_id)

                chain_type = int(struct.chain.chain_type[chain_i])
                if chain_type != C.ChainType.PROTEIN:
                    continue  # Only process protein chains

                save_path = pdb_save_dir / f"{pdb_id}_{entity_id}_protein.pdb"
                token_st = int(struct.chain.token_start[chain_i])
                token_end = token_st + int(struct.chain.num_tokens[chain_i])
                # crop
                chain_struct = struct.crop(np.arange(token_st, token_end))
                chain_struct = chain_struct.reassign_token_indices()
                # Reset asym_id to 1 for single chain
                chain_struct.chain.asym_id[0] = 1
                chain_struct.token.asym_id[:] = 1
                try:
                    chain_struct.write(save_path, is_predicted=False)
                except Exception as e:
                    logger.error(f"Error saving {save_path}: {e}")
                    continue


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
    # 1. Load manifest
    manifest: list[Metadata] = load_manifest(args.manifest_path)
    logger.info(f"Loaded manifest with {len(manifest)} records.")

    # 2. Chunk keys for batch processing
    # Processing in batches reduces pickling overhead
    chunk_size = args.batch_size
    manifest_chunk = [
        manifest[i : i + chunk_size] for i in range(0, len(manifest), chunk_size)
    ]

    # 3. Setup Multiprocessing Pool
    logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

    process_batch_partial = partial(process_batch, save_dir=args.output_dir)

    lmdb_path_str = str(args.processed_lmdb_path)
    if args.num_workers > 1:
        with multiprocessing.Pool(
            processes=args.num_workers, initializer=init_worker, initargs=(lmdb_path_str,)
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_batch_partial, manifest_chunk),
                total=len(manifest_chunk),
                desc="Extracting protein structures",
            ):
                pass
    else:
        # Single process mode
        init_worker(lmdb_path_str)
        for records in tqdm(
            manifest_chunk,
            total=len(manifest_chunk),
            desc="Extracting protein structures",
        ):
            process_batch_partial(records)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    main(args)
