"""Extract sequences from RCSB data

This script processes mmCIF files from the RCSB PDB database to extract
sequences of polymer entities (proteins, DNA, RNA). It applies filters based
on resolution, release date, number of chains, and number of residues. The
extracted sequences are saved in FASTA format.

## Usage:
```
python b_extract_sequences.py \
    --cif_dir /path/to/mmCIF/ \  # Path to RCSB mmCIF files
    --out_path /path/to/output_sequences.fasta \  # Output FASTA file
    --split train \               # Use predefined AlphaFold3 train split
    --num_workers 8               # Number of parallel workers
```

## Train/valid splits:
- train: up to 2021-09-30, max resolution 9.0A, max chains 300
- val: 2021-10-01 to 2023-01-12, max resolution 4.5A, max chains 1000, max residues 2560
"""

import argparse
import functools
import multiprocessing
import os
import pathlib
from datetime import datetime
from typing import Any

import gemmi
from tqdm import tqdm

import kfold.constants.residue as C

AF3_SPLITS = {
    "train": {
        "date_start": "1000-01-01",
        "date_end": "2021-09-30",
        "max_resolution": 9.0,
        "max_chains": 300,
        "max_residues": None,
    },
    "val": {
        "date_start": "2021-10-01",
        "date_end": "2023-01-12",
        "max_resolution": 4.5,
        "max_chains": 1000,
        "max_residues": 2560,
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `mmCIF/` directory from RCSB.",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        required=True,
        help="Output path for extracted sequences.",
    )

    # Predefined splits for date and resolution cutoffs
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        help="If given, cutoffs are set according to "
        "the specified split (train/val/test) used in AlphaFold3.\n"
        "- train: up to 2021-09-30, max resolution 9.0A, max chains 300\n"
        "- val: 2021-10-01 to 2023-01-12, max resolution 4.5A, max chains 1000, "
        "max residues 2560\n",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        help="Maximum resolution to consider.",
    )
    parser.add_argument(
        "--date_start",
        type=str,
        help="Start date for processing entries (YYYY-MM-DD). ",
    )
    parser.add_argument(
        "--date_end",
        type=str,
        help="Date cutoff for processing entries (YYYY-MM-DD). "
        "Default is 2021-09-30, the date of AlphaFold3 train cutoff.",
    )
    parser.add_argument(
        "--max_chains",
        type=int,
        help="Maximum number of chains to process.",
    )
    parser.add_argument(
        "--max_residues",
        type=int,
        help="Maximum number of residues to process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    # Apply split defaults if specified
    if args.split in AF3_SPLITS:
        print(f"Applying AlphaFold3 {args.split} split parameters...")
        split_params = AF3_SPLITS[args.split]
        if args.date_start is None:
            args.date_start = split_params["date_start"]
        if args.date_end is None:
            args.date_end = split_params["date_end"]
        if args.max_resolution is None:
            args.max_resolution = split_params["max_resolution"]
        if args.max_chains is None:
            args.max_chains = split_params["max_chains"]
        if args.max_residues is None:
            args.max_residues = split_params["max_residues"]

    return args


def get_first_value(block: gemmi.cif.Block, tag: str) -> Any | None:
    """
    Helper to get the first value of a tag, or None if missing.
    Gemmi's find_values() works for both single items and loops.
    """
    values = block.find_values(tag)
    if len(values) > 0:
        return values[0]
    return None


def extract_metadata(block: gemmi.cif.Block) -> dict[str, Any | None]:
    # 1. Dates (Deposit, Release, Revision)
    deposit = get_first_value(
        block, "_pdbx_database_status.recvd_initial_deposition_date"
    )
    # Revisions are stored in a loop.
    # Usually, the first item is the initial release, the last is the latest revision.
    rev_dates = block.find_values("_database_PDB_rev.date")
    release = rev_dates[0] if rev_dates else None
    latest_revision = rev_dates[-1] if rev_dates else None

    # 2. Method (e.g., X-RAY DIFFRACTION)
    method = get_first_value(block, "_exptl.method")

    # 3. Resolution (Handle X-ray vs EM vs NMR)
    resolution = get_first_value(block, "_refine.ls_d_res_high")  # X-ray standard
    if not resolution:
        resolution = get_first_value(block, "_em_3d_reconstruction.resolution")  # EM
    if not resolution:
        resolution = get_first_value(block, "_reflns.d_resolution_high")  # Fallback

    # 4. Temperature (Kelvin)
    temp = get_first_value(block, "_diffrn.ambient_temp")

    # 5. pH (Crystallization condition)
    ph = get_first_value(block, "_exptl_crystal_grow.pH")

    return {
        "deposit_date": deposit,
        "release_date": release,
        "revision_date": latest_revision,
        "method": method,
        "resolution": resolution,
        "temperature": temp,
        "ph": ph,
    }


def get_release_date(
    metadata: dict[str, Any | None], ref: str = "released"
) -> datetime | None:
    """Extracts the release date from metadata dictionary."""
    release_str = metadata.get("release_date")
    if release_str is not None:
        return datetime.fromisoformat(release_str)
    # Fallback to deposit date if no release date
    deposit_str = metadata.get("deposit_date")
    if deposit_str is not None:
        return datetime.fromisoformat(deposit_str)
    return None


def to_one_letter_sequence(
    ccd_codes: list[str],
    ctype: str,
) -> str:
    """
    Converts a list of 3-letter CCD codes to a 1-letter string.
    e.g., ['MET', 'ALA', 'MSE'] -> 'MAM'

    if remap_ambiguous_residues is True, maps ambiguous codes to standard ones:
        B -> D (Aspartic Acid or Asparagine)
        U -> C (Selenocysteine to Cysteine)
        Z -> E (Glutamic Acid or Glutamine)

    if remap_nonstandard_residues is True, maps non-standard codes to standard ones.
    one-letter codes not in the standard set are replaced with 'X' for proteins,
    'N' for DNA/RNA.

    """
    one_letter_seq = []

    match ctype:
        case "Protein":
            unk_letter = "X"
            standard_set = set(C.PROTEIN_AMINO_ACIDS)
        case "DNA":
            unk_letter = "N"
            standard_set = set(C.DNA_BASES)
        case "RNA":
            unk_letter = "N"
            standard_set = set(C.RNA_BASES)
        case _:
            raise ValueError(f"Unknown chain type: {ctype}")

    for code in ccd_codes:
        letter = C.convert_ccd_name_to_one_letter(code, unk_letter)

        if ctype == "Protein":
            # Remap ambiguous residues
            if letter == "B":
                letter = "D"  # Aspartic Acid or Asparagine
            elif letter == "U":
                letter = "C"  # Selenocysteine to Cysteine
            elif letter == "Z":
                letter = "E"  # Glutamic Acid or Glutamine
        # Map non-standard residues to Alanine
        if letter not in standard_set:
            letter = unk_letter

        one_letter_seq.append(letter)

    return "".join(one_letter_seq)


def filter_by_resolution(
    metadata: dict[str, Any | None],
    max_resolution: float | None,
) -> bool:
    """Returns True if the entry passes the resolution filter."""
    if max_resolution is None:
        return True

    res_raw = metadata.get("resolution")
    # Policy: Discard if resolution is missing (NMR) or too low (high value)
    if res_raw is None:
        return False
    try:
        resolution = float(res_raw)
        if resolution > max_resolution:
            return False
    except ValueError:
        # Handle cases where resolution might be a string like "NULL" or "N/A"
        return False
    return True


def filter_by_date(
    metadata: dict[str, Any | None],
    date_range: tuple[datetime | None, datetime | None],
) -> bool:
    """Returns True if the entry passes the date filter."""
    release_date = get_release_date(metadata)
    if release_date is None:
        return False
    start_date, end_date = date_range
    if start_date is None:
        start_date = datetime.min
    if end_date is None:
        end_date = datetime.max
    if not (start_date <= release_date <= end_date):
        return False
    return True


def process_cif_file(
    cif_path: pathlib.Path,
    date_range: tuple[datetime | None, datetime | None] = (None, None),
    max_resolution: float | None = None,
    max_chains: int | None = None,
    max_residues: int | None = None,
) -> tuple[str, dict[tuple[str, str], str] | None]:
    """
    Parses an mmCIF file using Gemmi and extracts 3-letter CCD codes,
    filtering by resolution and release date.
    """
    pdb_id = cif_path.name.split(".")[0].lower()
    try:
        # Load the CIF document
        # str(cif_path) handles pathlib objects safely for gemmi bindings
        doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
        if not doc:
            return pdb_id, None
        block: gemmi.cif.Block = doc[0]
    except Exception as e:
        print(f"Error reading CIF file {cif_path}: {e}")
        return pdb_id, None

    # --- 1. Metadata & Filtering ---
    # Extract metadata
    metadata = extract_metadata(block)

    # Check date filter
    if not filter_by_date(metadata, date_range):
        return pdb_id, None

    # Check resolution filter
    if not filter_by_resolution(metadata, max_resolution):
        return pdb_id, None

    # --- 2. Chain Counting for Filtering ---

    # Get polymer entities
    polymer_entity_ids = set()
    ent_ids = block.find_loop("_entity.id")
    ent_types = block.find_loop("_entity.type")
    for eid, etype in zip(ent_ids, ent_types, strict=True):
        if etype == "polymer":
            polymer_entity_ids.add(eid)

    # Map subchains to entities
    asym_id_to_entity_id: dict[str, str] = {}
    asym_ids = block.find_loop("_struct_asym.id")
    asym_ent_ids = block.find_loop("_struct_asym.entity_id")
    for asymid, eid in zip(asym_ids, asym_ent_ids, strict=True):
        asym_id_to_entity_id[asymid] = eid

    # Count the number of polymer chains while considering assemblies
    # Prepare Structure
    struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    struct.merge_chain_parts()
    struct.remove_alternative_conformations()
    struct.remove_empty_chains()

    # Map entity IDs to entities
    entity_id_to_entity: dict[str, gemmi.Entity] = {}
    for entity in struct.entities:
        entity_id_to_entity[entity.name] = entity

    num_polymer_chains = 0
    num_polymer_residues = 0
    if struct.assemblies:
        assembly = struct.assemblies[0]
        for gen in assembly.generators:
            # Number of operators * Number of chains in this generator
            for asym_id in gen.subchains:
                eid = asym_id_to_entity_id.get(asym_id)
                if eid in polymer_entity_ids:
                    entity = entity_id_to_entity[eid]
                    num_entity_chains = len(gen.operators)
                    num_polymer_chains += num_entity_chains
                    num_polymer_residues += num_entity_chains * len(entity.full_sequence)
    else:
        # Fallback: Count chains in asymmetric unit if no assembly defined
        for entity in struct.entities:
            if entity.entity_type == gemmi.EntityType.Polymer:
                num_entity_chains = len(entity.subchains)
                num_polymer_chains += num_entity_chains
                num_polymer_residues += num_entity_chains * len(entity.full_sequence)

    # Compute the number of polymer chains
    if max_chains is not None and num_polymer_chains > max_chains:
        return pdb_id, None

    # Compute the number of polymer residues
    if max_residues is not None and num_polymer_residues > max_residues:
        return pdb_id, None

    # --- 3. Sequence Extraction ---
    # Parse entities
    entities: list[gemmi.Entity] = []
    for entity in struct.entities:
        entity: gemmi.Entity
        if entity.entity_type == gemmi.EntityType.Water:
            continue
        entities.append(entity)

    entity_sequences: dict[tuple[str, str], str] = {}
    for entity in entities:
        entity: gemmi.Entity
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue

        polymer_type: gemmi.PolymerType = entity.polymer_type
        if polymer_type not in {
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.Dna,
            gemmi.PolymerType.Rna,
        }:
            continue

        match polymer_type:
            case gemmi.PolymerType.PeptideL:
                chaintype = "Protein"
            case gemmi.PolymerType.Dna:
                chaintype = "DNA"
            case gemmi.PolymerType.Rna:
                chaintype = "RNA"
            case _:
                raise ValueError(f"Unknown polymer type: {polymer_type}")

        # Extract sequences from all chains corresponding to this entity
        entity_id = entity.name
        sequence = to_one_letter_sequence(entity.full_sequence, chaintype)
        entity_sequences[(entity_id, chaintype)] = sequence

    return pdb_id, entity_sequences


if __name__ == "__main__":
    args = parse_args()

    date_start = datetime.fromisoformat(args.date_start)
    date_end = datetime.fromisoformat(args.date_end)
    date_range = (date_start, date_end)

    cif_files = list(sorted(args.cif_dir.glob("*/*.cif.gz")))
    print(f"Found {len(cif_files)} mmCIF files.")

    process_file = functools.partial(
        process_cif_file,
        date_range=date_range,
        max_resolution=args.max_resolution,
        max_chains=args.max_chains,
        max_residues=args.max_residues,
    )

    all_sequences: dict[tuple[str, str, str], str] = {}
    parsed_count = 0

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # imap_unordered yields results as soon as they are ready
        results = pool.imap_unordered(process_file, cif_files, chunksize=100)

        for pdb_id, sequences in tqdm(results, total=len(cif_files), desc="Processing"):
            if sequences is None:
                continue
            for (entity_id, chain_type), seq in sequences.items():
                all_sequences[(pdb_id, entity_id, chain_type)] = seq
            parsed_count += 1

    print(
        "Extraction complete. "
        f"Writing {len(all_sequences)} sequences from {parsed_count} entries to "
        f"{args.out_path}...",
    )

    # Write output
    pathlib.Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as out_file:
        for (pdb_id, entity_id, chaintype), sequence in all_sequences.items():
            out_file.write(f">{pdb_id}_{entity_id}_{chaintype.lower()}\n")
            out_file.write(f"{sequence}\n")

    print("Done.")
