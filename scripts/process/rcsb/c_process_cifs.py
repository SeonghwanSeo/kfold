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
import dataclasses
import functools
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime

import gemmi
from tqdm import tqdm

import kfold.constants as C
from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
METHOD_FILTERED = 4
CHAIN_COUNT_FILTERED = 5
RESIDUE_COUNT_FILTERED = 6
EMPTY_STRUCTURE_FILTERED = 7
INVALID_CHAIN_FILTERED = 8


@dataclasses.dataclass
class DataFilter:
    date_start: datetime = datetime.min
    date_end: datetime = datetime.max
    max_resolution: float | None = None
    min_chains: int = 1
    max_chains: int = 100_000  # A large number
    min_tokens: int = 1
    max_tokens: int = 1_000_000_000  # A large number
    filter_nmr: bool = False
    handle_invalid_chains: str = "allow"  # "allow" or "disallow"

    def __post_init__(self):
        if self.handle_invalid_chains not in ["allow", "disallow"]:
            raise ValueError(
                "handle_invalid_chains must be either 'allow' or 'disallow'."
            )

    def __repr__(self):
        return (
            f"DataFilter(\n"
            f"  date_start={self.date_start},\n"
            f"  date_end={self.date_end},\n"
            f"  max_resolution={self.max_resolution},\n"
            f"  min_chains={self.min_chains},\n"
            f"  max_chains={self.max_chains},\n"
            f"  min_tokens={self.min_tokens},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  filter_nmr={self.filter_nmr},\n"
            f"  handle_invalid_chains='{self.handle_invalid_chains}'\n"
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
        max_tokens=2560,
    ),
    "test": DataFilter(
        date_start=datetime.fromisoformat("2022-05-02 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
        max_resolution=4.5,
        max_chains=1000,
        max_tokens=5120,
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
    # NOTE: 2023-12-31 same to the boltz2 validation end date (2024-01-01),
    # There is no entry released on 2024-01-01 within this cutoff range.
    "val": DataFilter(
        date_start=datetime.fromisoformat("2023-01-01 00:00:00"),
        date_end=datetime.fromisoformat("2023-12-31 23:59:59"),
        max_resolution=4.5,
        max_chains=1000,
        max_tokens=2560,
    ),
    "test": DataFilter(
        date_start=datetime.fromisoformat("2024-01-01 00:00:00"),
        date_end=datetime.fromisoformat("2026-01-09 23:59:59"),
        max_resolution=4.5,
        min_chains=1,
        max_chains=1000,
        max_tokens=5120,
        filter_nmr=True,
        handle_invalid_chains="disallow",
    ),
}


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
        required=True,
        choices=["train", "val", "test"],
        help="Predefined data split to use:\n"
        "- train: up to 2022-12-31, max resolution 9.0A, max chains 300\n"
        "- val: 2023-01-01 to 2023-12-31, max resolution 4.5A, max chains 1000, "
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


_CCD_CACHE = None


def init_worker(ccd_path):
    """Initialize worker process with global CCD data."""
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def check_chain_count_cutoff(
    raw_struct: gemmi.Structure,
    min_chains: int,
    max_chains: int,
) -> bool:
    """Returns True if the structure passes the chain count filter."""
    num_polymer_chains = 0
    for entity in raw_struct.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            num_polymer_chains += len(entity.subchains)
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
    ccd: CCD,
    out_path: pathlib.Path,
    data_filter: DataFilter,
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

    # Get metadata without chain information
    # Handle cases like "1abc.cif.gz"
    pdb_id = cif_path.name.split(".")[0].lower()
    metadata: Metadata = cif_factory.prepare_metadata_from_rcsb(pdb_id, block)

    # Filter by date
    if not cif_factory.check_date_cutoff(
        metadata, data_filter.date_start, data_filter.date_end
    ):
        return DATE_FILTERED

    # Filter by experimental method (NMR)
    if data_filter.filter_nmr:
        exclude_methods = C.training.NMR_METHODS
        if not cif_factory.check_method(metadata, exclude_methods):
            return METHOD_FILTERED

    # Filter by resolution
    if data_filter.max_resolution is not None:
        if not cif_factory.check_resolution_cutoff(
            metadata, data_filter.max_resolution, skip_nmr=True
        ):
            return RESOLUTION_FILTERED

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = cif_factory.prepare_gemmi_structure(
        block, clean_up=True, expand_assembly=True
    )

    # Filter by chain count
    if not check_chain_count_cutoff(
        raw_struct, data_filter.min_chains, data_filter.max_chains
    ):
        return CHAIN_COUNT_FILTERED

    # Filter by token count (naive filter with residue count)
    if not check_residue_count_cutoff(
        raw_struct, data_filter.min_tokens, data_filter.max_tokens
    ):
        return RESIDUE_COUNT_FILTERED

    # Prepare reference structure with chain metadata
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    # Identify invalid chains
    invalid_chains: set[int] = set()

    # Validate chain geometry
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)

    # Get interfaces and those metadata; Detect clashes
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)

    if data_filter.handle_invalid_chains != "allow":
        if len(invalid_chains) > 0:
            return INVALID_CHAIN_FILTERED

    # Drop invalid chains
    cif_factory.prune_invalid_chains(ref_struct, invalid_chains)

    # Final checks
    if ref_struct.num_chains == 0:
        return EMPTY_STRUCTURE_FILTERED
    if ref_struct.num_polymer_chains == 0:
        return EMPTY_STRUCTURE_FILTERED
    if not (data_filter.min_chains <= ref_struct.num_chains <= data_filter.max_chains):
        return CHAIN_COUNT_FILTERED
    if not (data_filter.min_tokens <= ref_struct.num_tokens <= data_filter.max_tokens):
        return RESIDUE_COUNT_FILTERED

    # Validate final structure
    ref_struct.validate()

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    data_filter: DataFilter,
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    pdb_id = cif_path.name.split(".")[0].lower()
    out_path = output_dir / pdb_id[1:3] / f"{pdb_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return parse_cif(cif_path, ccd, out_path, data_filter)
    except Exception as e:
        print(f"Failed to process ({pdb_id}): {e}")
        raise e
        return FAILED


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    out_dir: pathlib.Path = args.data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply split defaults if specified
    print(f"Applying KFold {args.split} split parameters...")
    data_filter = KFOLD_SPLITS[args.split]
    print(data_filter)

    # Prepare partial function for multiprocessing
    worker_wrapped = functools.partial(
        worker_fn,
        output_dir=out_dir,
        data_filter=data_filter,
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
                pool.imap_unordered(worker_wrapped, cif_paths, chunksize=10),
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
    print(f"  Method filtered: {results.count(METHOD_FILTERED)}")
    print(f"  Chain count filtered: {results.count(CHAIN_COUNT_FILTERED)}")
    print(f"  Residue count filtered: {results.count(RESIDUE_COUNT_FILTERED)}")
    print(f"  Empty structure filtered: {results.count(EMPTY_STRUCTURE_FILTERED)}")
    print(f"  Invalid chain filtered: {results.count(INVALID_CHAIN_FILTERED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
