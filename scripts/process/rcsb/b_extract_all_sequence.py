import argparse
import dataclasses
import functools
import multiprocessing
import os
import pathlib
from collections import Counter
from datetime import datetime

import gemmi
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.utils.io.fasta import write_fasta
from kfold.training.preprocess import cif_factory

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


SPLITS = {
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
        min_residues=4,
        max_residues=2560,
    ),
    "test": DataFilter(
        date_start=datetime.fromisoformat("2022-05-02 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
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
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Process only this split. By default, generate both train and val.",
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

    # Read CIF file
    if cif_path.suffix == ".gz":
        block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
    else:
        block: gemmi.cif.Block = gemmi.cif.read_file(str(cif_path))[0]

    pdb_id = cif_path.name.split(".")[0].lower()
    metadata = cif_factory.prepare_metadata_from_rcsb(pdb_id, block)
    return extract_sequences(block, metadata, data_filter)


def extract_sequences(
    block: gemmi.cif.Block,
    metadata: Metadata,
    data_filter: DataFilter,
) -> tuple[int, list[tuple[str, str, C.ChainType, str]]]:
    """Apply one split's filters to an already-read CIF block."""
    pdb_id = metadata.id
    assert metadata.exp is not None, "Experimental metadata should not be None."

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
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    cif_factory.expand_first_assembly(raw_struct)
    cif_factory.clean_up_gemmi_structure(raw_struct)

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


def worker_fn(cif_path: pathlib.Path, splits: tuple[str, ...]):
    """Read each CIF once and return filtered sequences for the requested splits."""
    try:
        if cif_path.suffix == ".gz":
            block = gemmi.cif.read(str(cif_path))[0]
        else:
            block = gemmi.cif.read_file(str(cif_path))[0]
        pdb_id = cif_path.name.split(".")[0].lower()
        metadata = cif_factory.prepare_metadata_from_rcsb(pdb_id, block)
    except Exception as error:
        print(f"Failed to read {cif_path.name}: {error}")
        return {split: (FAILED, []) for split in splits}

    results = {}
    for split in splits:
        try:
            results[split] = extract_sequences(block, metadata, SPLITS[split])
        except Exception as error:
            print(f"Failed to process {cif_path.name} ({split}): {error}")
            results[split] = (FAILED, [])
    return results


def main():
    """Generate train and validation sequence files in one CIF pass."""
    args = parse_args()
    splits = (args.split,) if args.split else ("train", "val")
    for split in splits:
        print(f"Applying {split} split parameters: {SPLITS[split]}")
    cif_paths = sorted(
        path
        for path in args.cif_dir.rglob("*")
        if path.is_file()
        and path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
    )
    print(f"Found {len(cif_paths)} mmCIF files to process once for {splits}.")
    sequences_by_split = {split: [] for split in splits}
    counts = {split: Counter() for split in splits}
    worker = functools.partial(worker_fn, splits=splits)
    with multiprocessing.Pool(args.num_workers) as pool:
        for results in tqdm(
            pool.imap_unordered(worker, cif_paths, chunksize=20),
            total=len(cif_paths),
            desc="Extracting RCSB sequences",
        ):
            for split, (flag, sequences) in results.items():
                counts[split][flag] += 1
                if flag == SUCCESS:
                    sequences_by_split[split].extend(sequences)

    statuses = {
        SUCCESS: "Successful",
        FAILED: "Failed",
        DATE_FILTERED: "Date filtered",
        RESOLUTION_FILTERED: "Resolution filtered",
        METHOD_FILTERED: "Method filtered",
        CHAIN_COUNT_FILTERED: "Chain count filtered",
        RESIDUE_COUNT_FILTERED: "Residue count filtered",
    }
    for split in splits:
        print(f"Processing statistics ({split}):")
        for flag, label in statuses.items():
            print(f"  {label}: {counts[split][flag]}")
        write_sequences(args.data_dir / f"rcsb-{split}", sequences_by_split[split])


def write_sequences(data_dir: pathlib.Path, all_sequences: list):
    """Write one split's entity records and unique protein FASTA."""
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


if __name__ == "__main__":
    main()
