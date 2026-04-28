"""Extract sequences from RCSB mmCIF files to construct RecentPDB benchmark."""

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
DATE_FILTERED = 1


polymer_type_to_ctype: dict[gemmi.PolymerType, C.ChainType] = {
    gemmi.PolymerType.PeptideL: C.ChainType.PROTEIN,
    gemmi.PolymerType.Dna: C.ChainType.DNA,
    gemmi.PolymerType.Rna: C.ChainType.RNA,
}
ctype_to_unk: dict[C.ChainType, str] = {
    C.ChainType.PROTEIN: "X",
    C.ChainType.DNA: "N",
    C.ChainType.RNA: "N",
}
PROTEIN_AMINO_ACID_MAPPING: dict[str, str] = C.residue.PROTEIN_AMINO_ACID_MAPPING
PROTEIN_AMINO_ACIDS_SET: set[str] = C.residue.PROTEIN_AMINO_ACIDS_SET
DNA_BASES_SET: set[str] = C.residue.DNA_BASES_SET
RNA_BASES_SET: set[str] = C.residue.RNA_BASES_SET


@dataclasses.dataclass
class DataFilter:
    date_start: datetime = datetime.min
    date_end: datetime = datetime.max

    def __repr__(self):
        return (
            f"DataFilter(\n"
            f"  date_start={self.date_start},\n"
            f"  date_end={self.date_end},\n"
            f")"
        )


# NOTE(SeonghwanSeo): mmCIF files were downloaded on 2024-01-09.
# The training/validation cutoff is set to 2023-12-31, aligning with
# the Boltz2 cutoff (2024-01-01). Since no PDB releases occurred on
# 2024-01-01, using 2023-12-31 as the inclusive end date is functionally
# equivalent and ensures a clean separation between val and test sets.
KFOLD_SPLITS = {
    "train": DataFilter(  # train/val split
        date_start=datetime.min,
        date_end=datetime.fromisoformat("2023-12-31 23:59:59"),
    ),
    "test": DataFilter(  # test split
        date_start=datetime.fromisoformat("2024-01-01 00:00:00"),
        date_end=datetime.fromisoformat("2026-01-09 23:59:59"),
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
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="If given, cutoffs are set according to the specified split",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def parse_cif(
    cif_path: pathlib.Path,
    data_filter: DataFilter,
) -> tuple[int, list[tuple[str, str, C.ChainType, str]]]:
    """Parse a CIF file and return a gemmi.cif.Document object."""

    # Read CIF file
    if cif_path.suffix == ".gz":
        block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
    else:
        block: gemmi.cif.Block = gemmi.cif.read_file(str(cif_path))[0]

    # Get metadata without chain information
    pdb_id = cif_path.name.split(".")[0].lower()  # both .cif and .cif.gz
    metadata: Metadata = cif_factory.prepare_metadata_from_rcsb(pdb_id, block)
    assert metadata.exp is not None, "Experimental metadata should not be None."

    # Filter by date
    if not cif_factory.check_date_cutoff(
        metadata, data_filter.date_start, data_filter.date_end
    ):
        return DATE_FILTERED, []

    # Prepare gemmi Structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    cif_factory.clean_up_gemmi_structure(raw_struct)

    # Extract polymer sequences
    polymer_sequences: list[tuple[str, str, C.ChainType, str]] = []
    for entity in raw_struct.entities:
        entity_id = entity.name
        assert entity_id.isdigit(), f"Entity ID is not digit: {entity_id}"
        assert int(entity_id) >= 1
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        if entity.polymer_type not in polymer_type_to_ctype:
            continue
        ccd_sequence: list[str] = entity.full_sequence
        if len(ccd_sequence) < 4:
            # Skip very short polymers
            continue

        ctype: C.ChainType = polymer_type_to_ctype[entity.polymer_type]
        unk: str = ctype_to_unk[ctype]

        # Convert CCD names to one-letter codes
        tokens: list[str] = [
            C.residue.convert_ccd_name_to_one_letter(residue_name, unk=unk)
            for residue_name in ccd_sequence
        ]
        # Replace ambiguous residues with standard ones
        if ctype.is_protein:
            tokens = [PROTEIN_AMINO_ACID_MAPPING.get(aa, aa) for aa in tokens]
            tokens = [aa if aa in PROTEIN_AMINO_ACIDS_SET else unk for aa in tokens]
        elif ctype.is_dna:
            tokens = [base if base in DNA_BASES_SET else unk for base in tokens]
        else:
            tokens = [base if base in RNA_BASES_SET else unk for base in tokens]

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


def main():
    """Main function to extract all sequences from RCSB mmCIF files."""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Apply split defaults if specified
    print(f"Applying {args.split} split parameters...")
    data_filter = KFOLD_SPLITS[args.split]
    print(data_filter)

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    # Process CIF files in parallel
    parse_cif_partial = functools.partial(
        parse_cif,
        data_filter=data_filter,
    )
    with multiprocessing.Pool(args.num_workers) as pool:
        results: list[tuple[int, list[tuple[str, str, C.ChainType, str]]]] = list(
            tqdm(
                pool.imap_unordered(parse_cif_partial, cif_paths, chunksize=20),
                total=len(cif_paths),
                desc="Processing RCSB mmCIF files",
            )
        )
    print("Processing completed.")

    # Print stats
    flags = [result[0] for result in results]
    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Date filtered: {flags.count(DATE_FILTERED)}")

    # Collect all sequences
    all_sequences: list[tuple[str, str, C.ChainType, str]] = []
    for flag, sequences in results:
        if flag == SUCCESS:
            all_sequences.extend(sequences)

    # save to fasta
    output_fasta_path = data_dir / f"rcsb-{args.split}-sequences.fasta"
    all_rcsb_sequences: list[tuple[str, str]] = [
        (f"{pdb_id}|{entity_id}|{ctype.name.lower()}", sequence)
        for pdb_id, entity_id, ctype, sequence in sorted(all_sequences)
    ]
    write_fasta(all_rcsb_sequences, output_fasta_path)


if __name__ == "__main__":
    main()
