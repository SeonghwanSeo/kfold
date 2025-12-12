"""Match the length of structure embeddings to original sequence."""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
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
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_polymer_sequences.fasta",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        help="Directory containing the structure embeddings.",
        default="/cache/wykim_lab/kfold_data/pretrained/saprot_650m_raw/",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of directory to save the matched embeddings.",
        default="/cache/wykim_lab/kfold_data/pretrained/saprot_650m_matched/",
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


def load_array_or_tensor(prefix: str, dtype: torch.dtype) -> torch.Tensor | None:
    """Load a numpy array or torch tensor from a file."""
    npy_path = f"{prefix}.npy"
    pt_path = f"{prefix}.pt"
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        return torch.as_tensor(data, dtype=dtype)
    elif os.path.exists(pt_path):
        data = torch.load(pt_path, "cpu", weights_only=True)
        return data.to(dtype)
    else:
        return None


def check_and_match(key: str, sequence: str, source_path: Path, output_path: Path):
    pdb_id, entity_id, chain_type = key.split("_")  # noqa
    chain_type = chain_type.lower()

    embedding_filename = str(source_path / key[:2] / key[:4] / key)
    embedding = load_array_or_tensor(embedding_filename, dtype=torch.bfloat16)

    if embedding is None:
        # HACK: only log for protein
        if chain_type == "protein":
            logger.warning(f"Embedding for {key} not found. Skipping: {sequence}")
        return

    seq_length = len(sequence)
    if embedding.shape[0] != seq_length:
        # This happens when there is unk amino acids in the sequence
        # NOTE: ESMFold does not save unk amino acids in the pdb file
        if embedding.shape[0] > seq_length:
            logger.error(
                f"Embedding length {embedding.shape[0]} is greater than sequence length"
                f" {seq_length} for {key}."
            )
            return

        match chain_type:
            case "protein":
                unk_token = "X"
            case "dna" | "rna":
                unk_token = "N"
            case _:
                raise ValueError(f"Unknown chain type: {chain_type}")
        mask = torch.tensor([res != unk_token for res in sequence], dtype=torch.bool)
        if mask.sum().item() != embedding.shape[0]:
            logger.error(
                f"Mask sum {mask.sum().item()} does not match embedding length"
                f" {embedding.shape[0]} for {key}."
            )
            return

        # Create a new embedding with the original sequence length
        logger.info(
            f"Resizing embedding for {key} from {embedding.shape[0]} to {seq_length}."
        )
        out_embedding = torch.zeros(seq_length, embedding.shape[1], dtype=embedding.dtype)
        out_embedding[mask] = embedding
    else:
        out_embedding = embedding

    # Save the matched embedding
    output_file = output_path / key[:2] / key[:4] / f"{key}.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_embedding, output_file)


def main(args):
    # Ensure output directory exists
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Load sequences from fasta file
    seq_dict: dict[str, str] = {}
    with open(args.sequence_path) as f:
        lines = f.readlines()
        assert len(lines) % 2 == 0, "Fasta file should have even number of lines."
        for i in range(0, len(lines), 2):
            header = lines[i].strip()
            seq_id = header[1:]  # Remove '>' character
            seq = lines[i + 1].strip()
            seq_dict[seq_id] = seq

    # Process embeddings in batches
    keys = list(seq_dict.keys())
    with multiprocessing.Pool(args.num_workers) as pool:
        for i in tqdm(
            range(0, len(keys), args.batch_size),
            desc="Processing batches",
        ):
            batch_keys = keys[i : i + args.batch_size]
            tasks = [
                (
                    key,
                    seq_dict[key],
                    args.embedding_path,
                    args.output_path,
                )
                for key in batch_keys
            ]
            pool.starmap(check_and_match, tasks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(args)
