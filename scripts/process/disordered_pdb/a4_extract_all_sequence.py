"""Construct RCSB training set with cluster IDs."""

import argparse
import multiprocessing
import os
import pathlib

from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import write_fasta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster training set sequences using MMseqs2.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    return args


def parse_npz(
    npz_path: pathlib.Path,
) -> list[tuple[str, int, int, str]]:
    """Parse a NPZ file and return sequences and metadata."""
    struct: RefStructure = RefStructure.load_npz(npz_path)
    name = struct.id
    sequences: dict[int, tuple[int, str]] = {}
    for c in struct.chains:
        if c.entity_id in sequences:
            continue  # Skip duplicate chains
        if c.ctype.is_polymer:
            seq = c.get_sequence(map_to_standard=True)
        else:
            seq = "-".join(c.get_ccd_sequence())
        sequences[c.entity_id] = (c.ctype.value, seq)
    return [
        (name, entity_id, ctype, seq) for entity_id, (ctype, seq) in sequences.items()
    ]


def main():
    """Main function to construct RCSB training set."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb"
    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(exist_ok=True, parents=True)

    # Extract sequences from npz files
    npz_dir = data_dir / "npz"
    npz_files = sorted(npz_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} preprocessed files to process.")
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_npz, npz_files),
                total=len(npz_files),
                desc="Extracting sequences",
            )
        )
    print("Finished extracting sequences from NPZ files.")

    prot_v = C.ChainType.PROTEIN.value
    dna_v = C.ChainType.DNA.value
    rna_v = C.ChainType.RNA.value
    lig_v = C.ChainType.LIGAND.value
    ctype_dict = {prot_v: "protein", dna_v: "dna", rna_v: "rna", lig_v: "ligand"}
    # Collect all chain sequences
    all_sequences = []
    for seqs in results:
        all_sequences.extend(seqs)
    print(f"Total sequences extracted: {len(all_sequences)}")
    all_rcsb_sequences: list[tuple[str, str]] = [
        (f"{pdb_id}|{entity_id}|{ctype_dict[ctype]}", sequence)
        for pdb_id, entity_id, ctype, sequence in sorted(all_sequences)
    ]
    write_fasta(
        all_rcsb_sequences,
        seq_dir / "all_sequences.fasta",
    )

    # save unique sequences only
    uniq_protein_fasta_path = seq_dir / "unique_protein_sequences.fasta"
    uniq_proteins = set(seq for _, _, ctype, seq in all_sequences if ctype == prot_v)
    uniq_proteins: list[tuple[str, str]] = [
        (f"uniq_protein_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_proteins, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_proteins, uniq_protein_fasta_path)

    uniq_dna_fasta_path = seq_dir / "unique_dna_sequences.fasta"
    uniq_dnas = set(seq for _, _, ctype, seq in all_sequences if ctype == dna_v)
    uniq_dnas: list[tuple[str, str]] = [
        (f"uniq_dna_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_dnas, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_dnas, uniq_dna_fasta_path)

    uniq_rna_fasta_path = seq_dir / "unique_rna_sequences.fasta"
    uniq_rnas = set(seq for _, _, ctype, seq in all_sequences if ctype == rna_v)
    uniq_rnas: list[tuple[str, str]] = [
        (f"uniq_rna_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_rnas, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_rnas, uniq_rna_fasta_path)


if __name__ == "__main__":
    main()
