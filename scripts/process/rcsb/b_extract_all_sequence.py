import argparse
import dataclasses
import functools
import multiprocessing
import os
import pathlib
from datetime import datetime

import gemmi
from tqdm import tqdm

import kfold.constants as C
from kfold.data.pipelines import cif_factory
from kfold.data.types.metadata import Metadata
from kfold.data.utils.io.fasta import write_fasta

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
METHOD_FILTERED = 4
CHAIN_COUNT_FILTERED = 5
RESIDUE_COUNT_FILTERED = 6


polymer_type_to_ctype = {
    gemmi.PolymerType.PeptideL: C.ChainType.PROTEIN,
    gemmi.PolymerType.Dna: C.ChainType.DNA,
    gemmi.PolymerType.Rna: C.ChainType.RNA,
}
ctype_to_unk = {
    C.ChainType.PROTEIN: "X",
    C.ChainType.DNA: "N",
    C.ChainType.RNA: "N",
}
PROTEIN_AMINO_ACID_MAPPING = C.residue.PROTEIN_AMINO_ACID_MAPPING
PROTEIN_AMINO_ACIDS_SET = C.residue.PROTEIN_AMINO_ACIDS_SET
DNA_BASES_SET = C.residue.DNA_BASES_SET
RNA_BASES_SET = C.residue.RNA_BASES_SET


@dataclasses.dataclass
class DataFilter:
    date_start: datetime = datetime.min
    date_end: datetime = datetime.max
    max_resolution: float | None = None
    min_chains: int = 1
    max_chains: int = 100_000
    min_residues: int = 1
    max_residues: int = 1_000_000_000
    filter_nmr: bool = False

    def __repr__(self):
        return (
            f"DataFilter(\n"
            f"  date_start={self.date_start},\n"
            f"  date_end={self.date_end},\n"
            f"  max_resolution={self.max_resolution},\n"
            f"  min_chains={self.min_chains},\n"
            f"  max_chains={self.max_chains},\n"
            f"  min_residues={self.min_residues},\n"
            f"  max_residues={self.max_residues},\n"
            f"  filter_nmr={self.filter_nmr}\n"
            f")"
        )


AF3_SPLITS = {
    "train": DataFilter(
        date_start=datetime.min,
        date_end=datetime.fromisoformat("2021-09-30 23:59:59"),
        max_resolution=9.0,
        max_chains=300,
    ),
    "val": DataFilter(
        date_start=datetime.fromisoformat("2021-10-01 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
        max_resolution=4.5,
        max_chains=1000,
        max_residues=2560,
    ),
    "test": DataFilter(
        date_start=datetime.fromisoformat("2022-05-02 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
        max_resolution=4.5,
        max_chains=1000,
        max_residues=5120,
        filter_nmr=True,
    ),
}

# NOTE(SeonghwanSeo): mmCIF files were downloaded on 2024-01-09.
# The training/validation cutoff is set to 2023-12-31, aligning with
# the Boltz2 cutoff (2024-01-01). Since no PDB releases occurred on
# 2024-01-01, using 2023-12-31 as the inclusive end date is functionally
# equivalent and ensures a clean separation between val and test sets.
KFOLD_SPLITS = {
    "train": DataFilter(
        date_start=datetime.min,
        date_end=datetime.fromisoformat("2022-12-31 23:59:59"),
        max_resolution=9.0,
        max_chains=300,
    ),
    "val": DataFilter(
        date_start=datetime.fromisoformat("2023-01-01 00:00:00"),
        date_end=datetime.fromisoformat("2023-12-31 23:59:59"),
        max_resolution=4.5,
        max_chains=1000,
        max_residues=2048,
    ),
    "test": DataFilter(
        date_start=datetime.fromisoformat("2024-01-01 00:00:00"),
        date_end=datetime.fromisoformat("2026-01-09 23:59:59"),
        max_resolution=4.5,
        min_chains=2,
        max_chains=1000,
        min_residues=64,
        max_residues=5120,
        filter_nmr=True,
    ),
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract sequences from RCSB mmCIF files."
    )
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `mmCIF/` directory from RCSB.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )

    # Predefined splits for date and resolution cutoffs
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="If given, cutoffs are set according to "
        "the specified split (train/val/test) used in AlphaFold3.\n"
        "- train: up to 2021-09-30, max resolution 9.0A, max chains 300\n"
        "- val: 2021-10-01 to 2023-01-12, max resolution 4.5A, max chains 1000, "
        "max residues 2560\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def check_chain_count_cutoff(
    raw_struct: gemmi.Structure,
    min_chains: int,
    max_chains: int,
) -> bool:
    """Returns True if the structure passes the chain count filter."""
    polymer_asym_ids: set[str] = set()
    for entity in raw_struct.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            polymer_asym_ids.update(entity.subchains)

    # Count the number of polymer chains
    model: gemmi.Model = raw_struct[0]
    num_polymer_chains = 0
    for res_span in model.subchains():
        asym_id = res_span.subchain_id()
        if asym_id in polymer_asym_ids:
            num_polymer_chains += 1

    return min_chains <= num_polymer_chains <= max_chains


def check_residue_count_cutoff(
    raw_struct: gemmi.Structure,
    min_residues: int,
    max_residues: int,
) -> bool:
    """Returns True if the structure passes the residue count filter."""
    # Count total residues
    n_residues = sum(len(chain) for chain in raw_struct[0].subchains())
    return min_residues <= n_residues <= max_residues


def parse_cif(
    cif_path: pathlib.Path,
    data_filter: DataFilter,
) -> tuple[int, list[tuple[str, str, C.ChainType, str]]]:
    """Parse a CIF file and return a gemmi.cif.Document object."""

    pdb_id = cif_path.name.split(".")[0].lower()  # both .cif and .cif.gz

    # Read CIF file
    if cif_path.suffix == ".gz":
        doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    else:
        doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]
    del doc

    # Get metadata without chain information
    # Handle cases like "1abc.cif.gz"
    pdb_id = cif_path.name.split(".")[0].lower()
    metadata: Metadata = cif_factory.prepare_metadata_from_rcsb(pdb_id, block)
    assert metadata.exp is not None, "Experimental metadata should not be None."
    if metadata.exp.release_date == "2024-01-01":
        print(f"Found release date 2024-01-01 for {pdb_id}.")

    # Filter by date
    if not cif_factory.check_date_cutoff(
        metadata, data_filter.date_start, data_filter.date_end
    ):
        return DATE_FILTERED, []

    # Filter by experimental method (NMR)
    if data_filter.filter_nmr:
        exclude_methods = C.training.NMR_METHODS
        if not cif_factory.check_method(metadata, exclude_methods):
            return METHOD_FILTERED, []

    # Filter by resolution
    if data_filter.max_resolution is not None:
        if not cif_factory.check_resolution_cutoff(
            metadata, data_filter.max_resolution, skip_nmr=True
        ):
            return RESOLUTION_FILTERED, []

    # Prepare gemmi Structure
    raw_struct: gemmi.Structure = cif_factory.prepare_gemmi_structure(
        block, clean_up=True, expand_assembly=True
    )

    # Filter by chain count
    if not check_chain_count_cutoff(
        raw_struct, data_filter.min_chains, data_filter.max_chains
    ):
        return CHAIN_COUNT_FILTERED, []

    # Filter by residue count
    if not check_residue_count_cutoff(
        raw_struct, data_filter.min_residues, data_filter.max_residues
    ):
        return RESIDUE_COUNT_FILTERED, []

    # Extract polymer sequences
    polymer_sequences: list[tuple[str, str, C.ChainType, str]] = []
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        entity_id = entity.name
        assert entity_id.isdigit(), f"Entity ID is not digit: {entity_id}"
        assert int(entity_id) >= 1
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        if entity.polymer_type not in polymer_type_to_ctype:
            continue
        if len(entity.subchains) == 0:
            continue
        ccd_sequence: list[str] = entity.full_sequence
        if len(ccd_sequence) < 4:
            # Skip very short polymers
            continue

        ctype = polymer_type_to_ctype[entity.polymer_type]
        unk = ctype_to_unk[ctype]

        # Convert CCD names to one-letter codes
        seq = "".join(
            C.residue.convert_ccd_name_to_one_letter(residue_name, unk=unk)
            for residue_name in ccd_sequence
        )
        # Replace ambiguous residues with standard ones
        if ctype.is_protein:
            tokens = [PROTEIN_AMINO_ACID_MAPPING.get(aa, aa) for aa in seq]
            tokens = [aa if aa in PROTEIN_AMINO_ACIDS_SET else unk for aa in tokens]
        elif ctype.is_dna:
            tokens = [base if base in DNA_BASES_SET else unk for base in seq]
        elif ctype.is_rna:
            tokens = [base if base in RNA_BASES_SET else unk for base in seq]
        seq = "".join(tokens)
        polymer_sequences.append((pdb_id, entity_id, ctype, seq))

    # Extract ligand sequences
    nonpolymer_sequences: list[tuple[str, str, C.ChainType, str]] = []
    for entity in raw_struct.entities:
        entity: gemmi.Entity
        entity_id = entity.name
        assert entity_id.isdigit(), f"Entity ID is not digit: {entity_id}"
        if entity.entity_type not in {
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        }:
            continue
        if len(entity.subchains) == 0:
            continue
        ref_subchain = entity.subchains[0]
        raw_chain: gemmi.ResidueSpan = raw_struct[0].get_subchain(ref_subchain)
        ccd_sequence: list[str] = [res.name for res in raw_chain]
        if len(ccd_sequence) == 0:
            continue
        ctype = C.ChainType.LIGAND
        seq = "-".join(ccd_sequence)
        nonpolymer_sequences.append((pdb_id, entity_id, ctype, seq))

    return SUCCESS, polymer_sequences + nonpolymer_sequences


def worker_fn(cif_path: pathlib.Path, data_filter: DataFilter):
    try:
        return parse_cif(cif_path, data_filter)
    except Exception as e:
        pdb_id = cif_path.name.split(".")[0].lower()
        print(f"Failed to process ({pdb_id}): {e}")
        return FAILED


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"

    # Apply split defaults if specified
    print(f"Applying {args.split} split parameters...")
    data_filter = KFOLD_SPLITS[args.split]
    print(data_filter)

    # Prepare partial function for multiprocessing
    parse_cif_partial = functools.partial(
        worker_fn,
        data_filter=data_filter,
    )

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")
    with multiprocessing.Pool(args.num_workers) as pool:
        results: list[tuple[int, list[tuple[str, str, C.ChainType, str]]]] = list(
            tqdm(
                pool.imap_unordered(parse_cif_partial, cif_paths, chunksize=20),
                total=len(cif_paths),
                desc="Processing RCSB mmCIF files",
            )
        )
    print("Processing completed.")

    flags = [result[0] for result in results]

    # Print stats
    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Failed to process: {flags.count(FAILED)}")
    print(f"  Date filtered: {flags.count(DATE_FILTERED)}")
    print(f"  Resolution filtered: {flags.count(RESOLUTION_FILTERED)}")
    print(f"  Method filtered: {flags.count(METHOD_FILTERED)}")
    print(f"  Chain count filtered: {flags.count(CHAIN_COUNT_FILTERED)}")
    print(f"  Residue count filtered: {flags.count(RESIDUE_COUNT_FILTERED)}")

    # Collect all sequences
    all_sequences: list[tuple[str, str, C.ChainType, str]] = []
    for flag, sequences in results:
        if flag == SUCCESS:
            all_sequences.extend(sequences)

    # Print stats
    print("Sequence statistics:")
    print(f"  Total sequences extracted: {len(all_sequences)}")
    print(f"  Proteins: {sum(ctype.is_protein for _, _, ctype, _ in all_sequences)}")
    print(f"  DNAs: {sum(ctype.is_dna for _, _, ctype, _ in all_sequences)}")
    print(f"  RNAs: {sum(ctype.is_rna for _, _, ctype, _ in all_sequences)}")
    print(f"  Ligands: {sum(ctype.is_ligand for _, _, ctype, _ in all_sequences)}")

    # save to fasta
    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    output_fasta_path = seq_dir / "all_sequences.fasta"
    all_rcsb_sequences: list[tuple[str, str]] = [
        (f"{pdb_id}|{entity_id}|{ctype.name.lower()}", sequence)
        for pdb_id, entity_id, ctype, sequence in sorted(all_sequences)
    ]
    write_fasta(all_rcsb_sequences, output_fasta_path)

    # save unique sequences only
    uniq_protein_fasta_path = seq_dir / "unique_protein_sequences.fasta"
    uniq_proteins = set(seq for _, _, ctype, seq in all_sequences if ctype.is_protein)
    uniq_proteins: list[tuple[str, str]] = [
        (f"uniq_protein_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_proteins, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_proteins, uniq_protein_fasta_path)

    uniq_dna_fasta_path = seq_dir / "unique_dna_sequences.fasta"
    uniq_dnas = set(seq for _, _, ctype, seq in all_sequences if ctype.is_dna)
    uniq_dnas: list[tuple[str, str]] = [
        (f"uniq_dna_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_dnas, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_dnas, uniq_dna_fasta_path)

    uniq_rna_fasta_path = seq_dir / "unique_rna_sequences.fasta"
    uniq_rnas = set(seq for _, _, ctype, seq in all_sequences if ctype.is_rna)
    uniq_rnas: list[tuple[str, str]] = [
        (f"uniq_rna_{i + 1}", seq)
        for i, seq in enumerate(sorted(uniq_rnas, key=lambda x: (len(x), x)))
    ]
    write_fasta(uniq_rnas, uniq_rna_fasta_path)


if __name__ == "__main__":
    main()
