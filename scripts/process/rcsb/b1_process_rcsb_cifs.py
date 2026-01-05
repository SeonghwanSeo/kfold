"""Preprocess RCSB mmCIF files.

This script processes mmCIF files from the RCSB PDB database.

## Usage:
```
python c_process_rcsb.py \
    --cif_dir /path/to/mmCIF/ \         # Path to RCSB mmCIF files
    --ccd_path /path/to/ccd.pkl \       # Path to CCD pickled file
    --out_dir /path/to/output_npz/ \    # Output Directory
    --split train \                     # Use predefined AlphaFold3 train split
    --num_workers 8                     # Number of parallel workers
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

import gemmi
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure

AF3_SPLITS = {
    "train": {
        "date_start": "1000-01-01",
        "date_end": "2021-09-30",
        "max_resolution": 9.0,
        "max_chains": 300,
        "max_residues": None,
        "handle_invalid_chains": "allow",
    },
    "val": {
        "date_start": "2021-10-01",
        "date_end": "2023-01-12",
        "max_resolution": 4.5,
        "max_chains": 1000,
        "max_residues": 2560,
        "handle_invalid_chains": "allow",
    },
}

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
CHAIN_COUNT_FILTERED = 4
RESIDUE_COUNT_FILTERED = 5
EMPTY_STRUCTURE_FILTERED = 6
INVALID_CHAIN_FILTERED = 7


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF files.")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `mmCIF/` directory from RCSB.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        required=True,
        help="Path to CCD pickled file.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
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
        "--handle_invalid_chains",
        type=str,
        choices=["allow", "disallow"],
        help="Whether to allow structures with invalid chains.",
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
        if args.handle_invalid_chains is None:
            args.handle_invalid_chains = split_params["handle_invalid_chains"]

    return args


_CCD_CACHE = None


def init_worker(ccd_path):
    """Initialize worker process with global CCD data."""
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def check_resolution_cutoff(
    metadata: Metadata,
    max_resolution: float | None,
) -> bool:
    """Returns True if the entry passes the resolution filter."""
    if max_resolution is None:
        return True

    assert metadata.exp is not None
    resolution = metadata.exp.resolution
    if resolution is None:
        return False
    return resolution <= max_resolution


def check_date_cutoff(
    metadata: Metadata,
    date_range: tuple[datetime | None, datetime | None],
) -> bool:
    """Returns True if the entry passes the date filter."""
    assert metadata.exp is not None
    release_date = metadata.exp.release_date
    if release_date is None:
        return False
    release_date: datetime = datetime.fromisoformat(release_date)
    start_date, end_date = date_range
    start_date: datetime = start_date or datetime.min
    end_date: datetime = end_date or datetime.max
    if not (start_date <= release_date <= end_date):
        return False
    return True


def check_chain_count_cutoff(
    raw_struct: gemmi.Structure,
    max_chains: int | None,
) -> bool:
    """Returns True if the structure passes the chain count filter."""
    if max_chains is None:
        return True
    return len(raw_struct[0].subchains()) <= max_chains


def check_residue_count_cutoff(
    raw_struct: gemmi.Structure,
    max_residues: int | None,
) -> bool:
    """Returns True if the structure passes the residue count filter."""
    if max_residues is None:
        return True
    n_residues = sum(len(chain) for chain in raw_struct[0].subchains())
    return n_residues <= max_residues


def parse_cif(
    cif_path: pathlib.Path,
    ccd: CCD,
    out_path: pathlib.Path,
    date_start: datetime | None = None,
    date_end: datetime | None = None,
    max_resolution: float | None = None,
    max_chains: int | None = None,
    max_residues: int | None = None,
    allow_invalid_chains: bool = True,
) -> int:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    if out_path.exists():
        return SUCCESS

    # Read CIF file
    if cif_path.suffix == ".gz":
        doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    else:
        doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # Get metadata
    # Handle cases like "1abc.cif.gz"
    pdb_id = cif_path.name.split(".")[0].lower()
    metadata = cif_factory.prepare_metadata_from_experimental_data(
        pdb_id,
        block,
        source="rcsb",
    )

    # Filter by date
    if not check_date_cutoff(metadata, (date_start, date_end)):
        return DATE_FILTERED

    # Filter by resolution
    if not check_resolution_cutoff(metadata, max_resolution):
        return RESOLUTION_FILTERED

    # Prepare raw structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    # Clean up raw structure
    cif_factory.clean_up_raw_structure(raw_struct)
    # Expand the first assembly
    cif_factory.expand_first_assembly(raw_struct)

    # Filter by chain count
    if not check_chain_count_cutoff(raw_struct, max_chains):
        return CHAIN_COUNT_FILTERED
    # Filter by residue count
    if not check_residue_count_cutoff(raw_struct, max_residues):
        return RESIDUE_COUNT_FILTERED

    # Prepare reference structure
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)
    # Clean valid chains
    cif_factory.validate_chain_geometry(ref_struct)
    # Get interfaces
    cif_factory.detect_interfaces_and_prune_clashes(ref_struct)

    if not allow_invalid_chains:
        if not all(c_m.is_valid for c_m in ref_struct.metadata.chains):
            return INVALID_CHAIN_FILTERED

    # Drop invalid chains
    cif_factory.prune_invalid_chains(ref_struct)

    # Final checks
    if ref_struct.num_chains == 0:
        return EMPTY_STRUCTURE_FILTERED
    elif ref_struct.num_polymer_chains == 0:
        return EMPTY_STRUCTURE_FILTERED

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    date_start: datetime | None = None,
    date_end: datetime | None = None,
    max_resolution: float | None = None,
    max_chains: int | None = None,
    max_residues: int | None = None,
    allow_invalid_chains: bool = True,
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    pdb_id = cif_path.name.split(".")[0].lower()
    out_path = output_dir / pdb_id[1:3] / f"{pdb_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return parse_cif(
            cif_path,
            ccd,
            out_path,
            date_start,
            date_end,
            max_resolution,
            max_chains,
            max_residues,
            allow_invalid_chains,
        )
    except Exception as e:
        print(f"Failed to process ({pdb_id}): {e}")
        # raise e
        return FAILED


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    out_dir: pathlib.Path = args.data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for multiprocessing
    parse_cif_partial = functools.partial(
        worker_fn,
        output_dir=out_dir,
        date_start=datetime.fromisoformat(args.date_start) if args.date_start else None,
        date_end=datetime.fromisoformat(args.date_end) if args.date_end else None,
        max_resolution=args.max_resolution,
        max_chains=args.max_chains,
        max_residues=args.max_residues,
        allow_invalid_chains=args.handle_invalid_chains == "allow",
    )

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")
    with multiprocessing.Pool(
        args.num_workers,
        initializer=init_worker,
        initargs=(args.ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_cif_partial, cif_paths),
                total=len(cif_paths),
                desc="Processing RCSB mmCIF files",
            )
        )
    print("Processing completed.")

    # Print stats
    print("Processing statistics:")
    print(f"  Total files processed: {len(results)}")
    print(f"  Successfully processed: {results.count(SUCCESS)}")
    print(f"  Failed to process: {results.count(FAILED)}")
    print(f"  Date filtered: {results.count(DATE_FILTERED)}")
    print(f"  Resolution filtered: {results.count(RESOLUTION_FILTERED)}")
    print(f"  Chain count filtered: {results.count(CHAIN_COUNT_FILTERED)}")
    print(f"  Residue count filtered: {results.count(RESIDUE_COUNT_FILTERED)}")
    print(f"  Empty structure filtered: {results.count(EMPTY_STRUCTURE_FILTERED)}")
    print(f"  Invalid chain filtered: {results.count(INVALID_CHAIN_FILTERED)}")


if __name__ == "__main__":
    main()
