import argparse
import pathlib

from kfold.constants.residue import PROTEIN_AMINO_ACID_MAPPING
from kfold.data.utils.io.fasta import read_fasta, write_fasta


def parse_args():
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF files.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir
    seq_dir = data_dir / "sequences"

    # Prepare pretrained model inputs
    protein_seqs: list[tuple[str, str]] = read_fasta(
        seq_dir / "unique_protein_sequences.fasta"
    )
    # Map ambiguous amino acids to standard ones
    pretrain_emb_inputs: list[tuple[str, str]] = [
        (key, "".join(PROTEIN_AMINO_ACID_MAPPING.get(aa, aa) for aa in seq))
        for key, seq in protein_seqs
    ]
    # Map 'UNK' to 'ALA' for ESMFold compatibility
    pretrain_emb_inputs = [
        (key, seq.replace("X", "A")) for key, seq in pretrain_emb_inputs
    ]
    write_fasta(
        pretrain_emb_inputs,
        seq_dir / "esmfold_input.fasta",
    )


if __name__ == "__main__":
    main()
