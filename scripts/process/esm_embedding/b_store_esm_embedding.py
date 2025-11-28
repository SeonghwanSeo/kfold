import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from kfold.model.modules.sequence_encoder.esmc import ESMC, ESMCConfig

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
        description="Get ESM embedding for protein sequences in the dataset."
    )
    parser.add_argument(
        "--fasta_path",
        type=Path,
        help="Path to the fasta file containing protein sequences.",
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_protein_sequences.fasta",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["esmc_300m", "esmc_600m"],
        default="esmc_300m",
        help="ESM model to use for embedding.",
    )
    parser.add_argument(
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
    sequences: dict[tuple[str, int], str] = {}
    with open(args.fasta_path) as f:
        lines = f.readlines()
        assert len(lines) % 2 == 0, "Fasta file should have even number of lines."
        for i in range(0, len(lines), 2):
            header = lines[i].strip()
            seq_id = header[1:]  # Remove '>' character
            seq = lines[i + 1].strip()
            # Key format: {pdb_id}_{entity_id}_protein
            pdb_id, entity_id, suffix = seq_id.split("_")
            assert suffix == "protein", f"Unexpected suffix in header: {suffix}"
            sequences[(pdb_id, int(entity_id))] = seq

    # Process each sequence and store embeddings
    all_keys = sorted(sequences.keys(), key=lambda x: (len(sequences[x]), x))
    logger.info(f"Total sequences to process: {len(all_keys)}")
    logger.info(f"Max sequence length: {max(len(seq) for seq in sequences.values())}")

    # Process in batches
    budget = 0
    batch_sequences: list[str] = []
    batch_keys: list[tuple[str, int]] = []
    for i in tqdm(range(len(all_keys)), desc="Processing sequences"):
        key = all_keys[i]
        seq = sequences[key]
        seq_len = len(seq) + 2  # +2 for special tokens

        # Check if adding this sequence exceeds budget
        if (budget + seq_len) > args.budget_size:
            embeddings = model.encode(batch_sequences)
            for idx, key_i in enumerate(batch_keys):
                emb = embeddings[idx].cpu()
                pdb_id, entity_id = key_i
                save_path = (
                    args.output_dir
                    / pdb_id[:2]
                    / pdb_id
                    / f"{pdb_id}_{entity_id}_protein.pt"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb, save_path)
            del embeddings

            # Reset batch
            budget = 0
            batch_sequences = []
            batch_keys = []

        # Add current sequence to batch
        batch_sequences.append(seq)
        batch_keys.append(key)
        budget += seq_len

    # Process any remaining sequences in the last batch
    if len(batch_sequences) > 0:
        embeddings = model.encode(batch_sequences)
        for idx, key_i in enumerate(batch_keys):
            emb = embeddings[idx].cpu()
            pdb_id, entity_id = key_i
            save_path = (
                args.output_dir / pdb_id[:2] / pdb_id / f"{pdb_id}_{entity_id}_protein.pt"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, save_path)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
