import argparse
import io
import logging
import multiprocessing
import os
from functools import partial
from pathlib import Path

import lmdb
import torch
from tqdm import tqdm

import kfold.constants as C
from kfold.data.schema import Metadata
from kfold.data.tokenized import TokenizedStructure
from kfold.training.folding.dataset.datamodule import load_manifest

logger = logging.getLogger(__name__)

# Global variable for the worker process to hold the LMDB environment
_env: lmdb.Environment | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match the length of structure embeddings to holo protein"
        " chain dimensions."
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
        "--embedding_path",
        type=Path,
        help="Directory containing the structure embeddings.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of directory to save the length-matched embeddings.",
        required=True,
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


def process_batch(metadatas: list[Metadata], root_dir: Path, save_dir: Path):
    """Process a batch of metadatas to extract protein structures."""
    global _env
    assert _env is not None, "LMDB environment is not initialized."

    with _env.begin(write=False) as txn:
        for metadata in metadatas:
            pdb_id = metadata.id

            src_dir = root_dir / pdb_id[1:3] / pdb_id
            dst_dir = save_dir / pdb_id[1:3] / pdb_id
            if not src_dir.exists():
                logger.warning(f"Source directory {src_dir} does not exist. Skipping.")
                continue

            if len(list(src_dir.glob("*_protein.pt"))) == 0:
                logger.info(f"No protein chains found for {pdb_id}. Skipping.")
                continue  # No protein chains found, skip

            dst_dir.mkdir(parents=True, exist_ok=True)

            value_bytes = txn.get(pdb_id.encode("utf-8"))

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

                residue_st = int(struct.chain.residue_start[chain_i])
                num_residues = int(struct.chain.num_residues[chain_i])
                if num_residues == 0:
                    continue  # Skip empty chains

                src_path = src_dir / f"{pdb_id}_{entity_id}_protein.pt"
                dst_path = dst_dir / f"{pdb_id}_{entity_id}_protein.pt"

                if not src_path.exists():
                    continue

                if dst_path.exists():
                    continue

                embedding = torch.load(src_path, "cpu", weights_only=True).to(
                    torch.bfloat16
                )
                length = embedding.shape[0]
                if length == num_residues:
                    torch.save(embedding, dst_path)
                else:
                    is_resolved = struct.residue.resolved_mask[
                        residue_st : residue_st + num_residues
                    ]
                    if is_resolved.sum().item() != length:
                        logger.error(
                            f"Resolved residue count {is_resolved.sum().item()} does not"
                            f" match embedding length {length} for {pdb_id}_{entity_id}."
                        )
                        continue
                    out_embedding = torch.zeros(
                        num_residues, embedding.shape[1], dtype=embedding.dtype
                    )
                    out_embedding[is_resolved] = embedding
                    torch.save(out_embedding, dst_path)


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
    logger.info(f"Loaded manifest with {len(manifest)} metadatas.")

    # 2. Chunk keys for batch processing
    # Processing in batches reduces pickling overhead
    chunk_size = args.batch_size
    manifest_chunk = [
        manifest[i : i + chunk_size] for i in range(0, len(manifest), chunk_size)
    ]

    # 3. Setup Multiprocessing Pool
    logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

    process_batch_partial = partial(
        process_batch, root_dir=args.embedding_path, save_dir=args.output_path
    )

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
        for metadatas in tqdm(
            manifest_chunk,
            total=len(manifest_chunk),
            desc="Extracting protein structures",
        ):
            process_batch_partial(metadatas)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    main(args)
