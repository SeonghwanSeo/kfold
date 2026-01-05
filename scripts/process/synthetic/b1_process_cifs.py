"""Preprocess synthetic data mmCIF files.

This script processes mmCIF files.

## Usage:
```
python b1_process_cifs.py \
    --cif_dir /path/to/cif/ \           # Path to mmCIF files
    --ccd_path /path/to/ccd.pkl \       # Path to CCD pickled file
    --out_dir /path/to/output_npz/ \    # Output Directory
    --num_workers 8                     # Number of parallel workers
```
"""

import argparse
import functools
import multiprocessing
import os
import pathlib

import gemmi
from tqdm import tqdm

from kfold.data.ccd import CCD
from kfold.data.pipelines import parse_input
from kfold.data.structure import RefStructure

# Error handling
SUCCESS = 0
FILTERED = 1
FAILED = 2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process synthetic mmCIF files.")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `.cif` files directory.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        required=True,
        help="Path to CCD pickled file.",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument(
        "--handle_invalid_chains",
        type=str,
        choices=["allow", "disallow"],
        default="disallow",
        help="Whether to allow structures with invalid chains.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name for predicted structures.",
    )
    parser.add_argument(
        "--clash_distance_cutoff",
        type=float,
        default=1.7,
        help="Distance cutoff to consider atom clashes.",
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


def parse_cif(
    cif_path: pathlib.Path,
    ccd: CCD,
    out_path: pathlib.Path,
    model: str | None = None,
    clash_distance_cutoff: float = 1.7,
    allow_invalid_chains: bool = False,
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
    name = cif_path.name.split(".")[0]
    metadata = parse_input.prepare_metadata(name, block, source="prediction")
    # Add prediction record
    assert metadata.prediction is not None
    metadata.prediction.model = model

    # Prepare raw structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)

    # Prepare reference structure
    ref_struct: RefStructure = parse_input.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    # Insert coordinates
    parse_input.insert_coordinates(ref_struct, raw_struct, metadata)
    # Clean valid chains
    parse_input.validate_chain_geometry(ref_struct)
    # Get interfaces
    parse_input.detect_interfaces_and_prune_clashes(ref_struct, clash_distance_cutoff)

    if not allow_invalid_chains:
        if not all(c_m.is_valid for c_m in ref_struct.metadata.chains):
            return FILTERED

    # Drop invalid chains
    parse_input.prune_invalid_chains(ref_struct)

    # Final checks
    if ref_struct.num_chains == 0:
        return FILTERED
    elif ref_struct.num_polymer_chains == 0:
        return FILTERED

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    model: str | None = None,
    clash_distance_cutoff: float = 1.7,
    allow_invalid_chains: bool = False,
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    name = cif_path.name.split(".")[0]
    out_path = output_dir / f"{name}.npz"
    try:
        return parse_cif(
            cif_path,
            ccd,
            out_path,
            model=model,
            clash_distance_cutoff=clash_distance_cutoff,
            allow_invalid_chains=allow_invalid_chains,
        )
    except Exception as e:
        print(f"Failed to process ({name}): {e}")
        # raise e
        return FAILED


def main():
    """Main function to process mmCIF files in parallel."""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    out_dir: pathlib.Path = args.out_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for multiprocessing
    parse_cif_partial = functools.partial(
        worker_fn,
        output_dir=out_dir,
        model=args.model,
        clash_distance_cutoff=args.clash_distance_cutoff,
        allow_invalid_chains=args.handle_invalid_chains == "allow",
    )

    cif_paths = sorted(cif_dir.rglob("*.cif*"))
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
                desc="Processing synthetic data",
            )
        )
    print("Processing completed.")
    n_success = results.count(SUCCESS)
    n_filtered = results.count(FILTERED)
    n_failed = results.count(FAILED)
    print(f"Successful: {n_success}, Filtered: {n_filtered}, Failed: {n_failed}")


if __name__ == "__main__":
    main()
