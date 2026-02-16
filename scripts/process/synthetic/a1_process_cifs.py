"""Preprocess synthetic data mmCIF files."""

import argparse
import functools
import multiprocessing
import os
import pathlib

import gemmi
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure

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
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for predicted structures.",
    )
    parser.add_argument(
        "--use_dir_name",
        action="store_true",
        help=(
            "Whether to use the parent directory name as the entry ID "
            "(instead of the file name)"
        ),
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
    model: str,
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
    name = out_path.stem
    metadata = cif_factory.prepare_metadata_from_synthetic_data(name, block, model)

    # Prepare raw structure
    raw_struct: gemmi.Structure = cif_factory.prepare_gemmi_structure(
        block, clean_up=True
    )

    # Prepare reference structure
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    # Identify invalid chains
    invalid_chains: set[int] = set()

    # Validate chain geometry
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)

    # Get interfaces and detect clashes
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)

    if len(invalid_chains) > 0:
        return FILTERED

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    model: str,
    use_dir_name: bool = False,
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    if use_dir_name:
        filename = cif_path.name.split(".")[0]
        assert filename.startswith("structure_"), (
            f"Expected file name to start with 'structure_', got {cif_path}"
        )
        name = f"{cif_path.parent.name}_{filename[len('structure_') :]}"
    else:
        name = cif_path.name.split(".")[0]
    out_path = output_dir / f"{name}.npz"
    try:
        return parse_cif(cif_path, ccd, out_path, model)
    except Exception as e:
        print(f"Failed to process ({name}): {e}")
        # raise e
        return FAILED


def main():
    """Main function to process mmCIF files in parallel."""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.out_dir / args.name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for mmCIF files in {cif_dir}...")
    cif_paths = sorted(cif_dir.rglob("*.cif*"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    # Run cif processing in parallel
    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    parse_cif_partial = functools.partial(
        worker_fn,
        output_dir=out_dir,
        model=args.model,
        use_dir_name=args.use_dir_name,
    )
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
