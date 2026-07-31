import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

import gemmi
import msgpack
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure

# Error handling
SUCCESS = 0
FAILED = 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process disordered PDB mmCIF files")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the directory including disordered pdb files",
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


_CCD_CACHE: CCD | None = None


def init_worker(ccd_path):
    """Initialize worker process with global CCD data."""
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def parse_cif(
    cif_path: pathlib.Path,
    ccd: CCD,
    out_path: pathlib.Path,
) -> int:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    # Get metadata without chain information
    # Handle cases like "1abc.cif.gz"
    pdb_id = cif_path.name.split(".")[0].lower()
    metadata: Metadata = cif_factory.prepare_metadata_from_synthetic_data(pdb_id, "AF-M")

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.read_structure(str(cif_path))
    cif_factory.add_entity_info(raw_struct, format="cif")
    cif_factory.clean_up_gemmi_structure(raw_struct)

    # Prepare reference structure with chain metadata
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    if len(ref_struct.chains) == 0:
        print(f"Warning: No valid chains found in {cif_path}. Skipping.")

    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    # Identify invalid chains
    invalid_chains: set[int] = set()

    # Validate chain geometry
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)

    # Get interfaces and those metadata; Detect clashes
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)

    # Detect orphaned branched ligands
    cif_factory.propagate_invalidity_to_ligands(ref_struct, invalid_chains)

    # Drop invalid chains
    cif_factory.prune_invalid_chains(ref_struct, invalid_chains)

    if invalid_chains:
        return FAILED

    # Validate final structure
    ref_struct.validate()

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    pdb_id = cif_path.name.split(".")[0].lower()
    out_path = output_dir / pdb_id[1:3] / f"{pdb_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return parse_cif(cif_path, ccd, out_path)
    except Exception as e:
        print(f"Failed to process ({pdb_id}): {e}")
        return FAILED


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb"
    assert args.ccd_path.exists(), f"CCD file not found: {args.ccd_path}"

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the training keys
    rcsb_train_dir = args.data_dir / "rcsb-train"
    manifest_path = rcsb_train_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        manifest = msgpack.load(f, strict_map_key=False)
    pdb_ids = set(m["id"].lower() for m in manifest)
    print(f"Loaded {len(pdb_ids)} training keys from {manifest_path}")

    cif_paths = sorted(
        file for pattern in ("*.cif", "*.cif.gz") for file in cif_dir.rglob(pattern)
    )
    print(f"Found {len(cif_paths)} mmCIF files in {cif_dir} before filtering.")

    cif_paths = [file for file in cif_paths if file.stem.split(".")[0].lower() in pdb_ids]
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    # Prepare partial function for multiprocessing
    worker_wrapped = functools.partial(worker_fn, output_dir=out_dir)

    with multiprocessing.Pool(
        args.num_workers,
        initializer=init_worker,
        initargs=(args.ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_wrapped, cif_paths),
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
