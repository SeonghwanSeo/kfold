import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from kfold.model.modules.sequence_encoder.esmc import ESMC, ESMCConfig
from kfold.utils.files import load_fasta

try:
    import esm  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ESM package is required to run this script. "
        "Please install it via 'pip install esm'."
    ) from e


logger = logging.getLogger(__name__)


# TODO: remove default path before publish
def parse_args():
    parser = argparse.ArgumentParser(
        description="Get ESM-C embedding for protein sequences in the dataset."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the fasta file containing protein sequences.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["esmc_300m", "esmc_600m"],
        default="esmc_600m",
        help="ESM model to use for embedding.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Path of directory to save the embeddings.",
    )
    parser.add_argument(
        "--budget_size",
        type=int,
        default=2048 * 32,
        help="Budget size for ESM model.",
    )
    return parser.parse_args()


def main(args):
    # No gradient computation
    torch.set_grad_enabled(False)

    # Initialize ESM model
    config = ESMCConfig.from_model_name(args.model)
    model = ESMC(config).eval().cuda()
    logger.info(f"Using model: {args.model}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read fasta file
    sequences: dict[str, str] = load_fasta(args.input)

    # Process each sequence and store embeddings
    all_keys = sorted(sequences.keys(), key=lambda x: (len(sequences[x]), x))
    logger.info(f"Total sequences to process: {len(all_keys)}")
    logger.info(f"Max sequence length: {max(len(seq) for seq in sequences.values())}")

    # Process in batches
    budget = 0
    batches: list[list[str]] = []
    last_batch: list[str] = []
    for key in all_keys:
        seq = sequences[key]
        seq_len = len(seq) + 2  # +2 for special tokens
        # Check if adding this sequence exceeds budget
        if (budget + seq_len) > args.budget_size:
            batches.append(last_batch)
            # Reset batch
            budget = 0
        # Add current sequence to batch
        last_batch.append(key)
        budget += seq_len
    # Add the last batch if not empty
    if last_batch:
        batches.append(last_batch)

    logger.info(f"Total batches to process: {len(batches)}")

    # Process batches
    for batch_keys in tqdm(batches, desc="Processing sequences"):
        batch_sequences = [sequences[key] for key in batch_keys]
        embeddings = model.encode(batch_sequences)
        for key, emb in zip(batch_keys, embeddings, strict=True):
            emb = emb.cpu()
            save_path = args.output_dir / f"{key}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, save_path)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
