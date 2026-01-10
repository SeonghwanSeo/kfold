import argparse
import logging
from collections.abc import Generator
from pathlib import Path

import torch
from tqdm import tqdm

from kfold.model.modules.sequence_encoder.esmc import ESMC, ESMCConfig

# Error handling for ESM package
try:
    import esm  # noqa: F401
except ImportError as e:
    raise ImportError("ESM package is required. Install it via 'pip install esm'.") from e

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ESM-C embeddings.")
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Path to the input fasta file."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["esmc_300m", "esmc_600m"],
        default="esmc_600m",
        help="ESM model variant.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save .pt files.",
    )
    parser.add_argument(
        "--budget_size", type=int, default=2048 * 64, help="Token budget per batch."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing embeddings."
    )
    return parser.parse_args()


def fasta_reader(fasta_path: Path) -> Generator[tuple[str, str], None, None]:
    """Memory-efficient fasta reader."""
    with open(fasta_path) as f:
        header, seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line)
        if header:
            yield header, "".join(seq)


def get_batches(
    all_keys: list[str], sequences: dict[str, str], budget_limit: int
) -> Generator[tuple[list[str], list[str]], None, None]:
    """Yields batches of sequences based on token budget."""
    batch_seqs, batch_keys = [], []
    current_budget = 0

    for key in all_keys:
        seq = sequences[key]
        # +2 for BOS/EOS tokens
        seq_len = len(seq) + 2

        if (current_budget + seq_len) > budget_limit and batch_seqs:
            yield batch_seqs, batch_keys
            batch_seqs, batch_keys, current_budget = [], [], 0

        batch_seqs.append(seq)
        batch_keys.append(key)
        current_budget += seq_len

    if batch_seqs:
        yield batch_seqs, batch_keys


@torch.inference_mode()
def main(args):
    # Output setup
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ESMCConfig.from_model_name(args.model)
    model = ESMC(config).eval().to(device)
    logger.info(f"Initialized {args.model} on {device}")

    # Read and sort sequences by length to minimize padding
    logger.info(f"Loading sequences from {args.input}...")
    sequences = {header: seq for header, seq in fasta_reader(args.input)}

    # Sorting by length is a common trick to improve batch efficiency
    sorted_keys = sorted(sequences.keys(), key=lambda k: len(sequences[k]))

    # Filter existing files if not overwriting
    if not args.overwrite:
        sorted_keys = [
            k for k in sorted_keys if not (args.output_dir / f"{k}.pt").exists()
        ]
        logger.info(f"Skipping existing files. Remaining: {len(sorted_keys)}")

    # Process batches
    pbar = tqdm(total=len(sorted_keys), desc="Encoding sequences")
    for batch_seqs, batch_keys in get_batches(sorted_keys, sequences, args.budget_size):
        # Forward pass
        embeddings = model.encode(batch_seqs)  # Assuming this returns a list of tensors

        # Save results
        for key, emb in zip(batch_keys, embeddings, strict=True):
            save_path = args.output_dir / f"{key}.pt"
            # Using non_blocking if moving to CPU for minor speedup
            torch.save(emb.to("cpu", non_blocking=True), save_path)

        pbar.update(len(batch_keys))
    pbar.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    main(args)
