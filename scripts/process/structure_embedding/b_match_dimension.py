"""Match the length of structure embeddings to original sequence."""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match the length of structure embeddings to original sequence."
    )
    parser.add_argument(
        "--sequence_path",
        type=Path,
        help="Path to the sequence fasta file.",
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/raw_data/rcsb_polymer_sequences.fasta",
    )
    parser.add_argument(
        "--embedding_dir",
        type=Path,
        help="Directory containing the structure embeddings.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
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
        "--chunksize",
        type=int,
        help="Chunk size for each worker process.",
        default=500,
    )
    parser.add_argument(
        "--max_ensembles",
        type=int,
        help="Maximum number of ensembles to consider per structure.",
        default=5,
    )
    return parser.parse_args()


def fetch_subdir(name: str, root_dir: Path) -> Path | None:
    if len(name) < 4:
        raise ValueError("Name must be at least 4 characters long.")
    if (subdir := root_dir / name[:2] / name).exists():
        return subdir
    elif (subdir := root_dir / name[1:3] / name).exists():
        return subdir
    return None


def check_and_match(
    key: str,
    sequence: str,
    source_dir: Path,
    output_dir: Path,
    max_ensembles: int = 5,
):
    pdb_id, entity_id, chain_type = key.split("_")  # noqa

    output_file = output_dir / pdb_id[1:3] / pdb_id / f"{key}.pt"
    if output_file.exists():
        # logger.info(f"Output file for {key} already exists. Skipping.")
        return

    chain_type = chain_type.lower()
    if chain_type != "protein":
        # logger.info(f"Skipping non-protein chain: {key}.")
        return

    subdir = fetch_subdir(pdb_id, source_dir)
    if subdir is None:
        logger.warning(f"Embedding subdir for {key} not found. Skipping.")
        return
    filename = subdir / f"{key}.pt"
    if not filename.exists():
        logger.warning(f"Embedding file for {key} not found. Skipping.")
        return

    embedding: torch.Tensor = torch.load(filename, "cpu", weights_only=True)
    # Convert tensor to bfloat16 to save storage
    embedding = embedding.to(torch.bfloat16)

    # Match dimension
    if embedding.ndim == 4:
        embedding = embedding.squeeze(0)
    if embedding.ndim not in (2, 3):
        logger.error(f"Unexpected embedding shape {embedding.shape} for {filename}.")
        return
    if embedding.ndim == 2:
        # Add number of structure dimension
        # [seq_len, feature_dim] -> [1, seq_len, feature_dim]
        embedding = embedding.unsqueeze(0)

    # Maximum ensemble handling
    if embedding.shape[0] > max_ensembles:
        embedding = embedding[:max_ensembles, :, :]

    seq_length = len(sequence)
    emb_shape = embedding.shape
    num_struct, emb_length, dim = emb_shape
    if emb_length > seq_length:
        logger.error(
            f"Embedding length {emb_shape} is greater than sequence length"
            f" {seq_length} for {key}."
        )
        return
    elif emb_length < seq_length:
        # This happens when there is unk amino acids in the sequence
        # NOTE: AF3/ESMFold does not save unk amino acids in the pdb file
        match chain_type:
            case "protein":
                unk_token = "X"
            case "dna" | "rna":
                unk_token = "N"
            case _:
                raise ValueError(f"Unknown chain type: {chain_type}")
        mask = torch.tensor([res != unk_token for res in sequence], dtype=torch.bool)
        if mask.sum().item() != emb_length:
            logger.error(
                f"Mask sum {mask.sum().item()} does not match embedding length"
                f" {emb_length} for {key}."
            )
            return

        # Create a new embedding with the original sequence length
        logger.info(f"Resizing embedding for {key} from {emb_shape} to {seq_length}.")
        out_embedding = torch.zeros((num_struct, seq_length, dim), dtype=embedding.dtype)
        out_embedding[:, mask, :] = embedding
    else:
        out_embedding = embedding.clone()

    # Save the matched embedding
    output_file = output_dir / pdb_id[1:3] / pdb_id / f"{key}.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_embedding, output_file)


def check_and_match_wrapper(args):
    """Helper to unpack arguments for pool.imap"""
    return check_and_match(*args)


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. FASTA Parsing (Handle wrapped sequences)
    seq_dict: dict[str, str] = {}
    with open(args.sequence_path) as f:
        lines = f.readlines()
        assert len(lines) % 2 == 0, "Fasta file should have even number of lines."
        for i in range(0, len(lines), 2):
            header = lines[i].strip()
            seq_id = header[1:]  # Remove '>' character
            seq = lines[i + 1].strip()
            seq_dict[seq_id] = seq

    # 3. Prepare all tasks (No manual batching needed)
    keys = list(seq_dict.keys())
    # Create a list of tuples for arguments
    all_tasks = [
        (key, seq_dict[key], args.embedding_dir, args.output_dir) for key in keys
    ]
    # 4. Process with Smooth Progress Bar
    # If num_workers=1, standard loop is fine, but pool works too (just slower overhead)
    if args.num_workers <= 1:
        for task in tqdm(all_tasks, desc="Processing"):
            check_and_match_wrapper(task)
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            # imap_unordered is often faster if order doesn't matter
            # chunksize helps performance on large lists
            results = pool.imap_unordered(
                check_and_match_wrapper, all_tasks, chunksize=args.chunksize
            )

            # This loop drives the progress bar based on ACTUAL completions
            for _ in tqdm(results, total=len(all_tasks), desc="Processing"):
                pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(args)
