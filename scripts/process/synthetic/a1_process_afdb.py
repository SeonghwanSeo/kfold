"""Preprocess synthetic protein monomers from mmCIF files."""

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
        "--id_path",
        type=pathlib.Path,
        required=True,
        help="Path to the synthetic data IDs file.",
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
        "--split",
        choices=["short", "long"],
        default="long",
        help="Data split to process (short or long sequences).",
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


def check_residue_count_cutoff(
    raw_struct: gemmi.Structure,
    split: str,
) -> bool:
    """Returns True if the structure passes the residue count filter."""
    # Count total residues
    n_residues = sum(len(chain) for chain in raw_struct[0].subchains())
    if split == "short":
        return n_residues <= 200
    else:
        return n_residues > 200


def parse_cif(
    cif_path: pathlib.Path,
    ccd: CCD,
    out_path: pathlib.Path,
    split: str = "long",
) -> int:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    if out_path.exists():
        return SUCCESS

    # Read CIF file
    if "cif" in cif_path.name:
        block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
    else:
        block: gemmi.cif.Block = gemmi.cif.read_file(str(cif_path))[0]

    # Get metadata
    name = cif_path.name.split(".")[0]
    model = "AlphaFold2"
    metadata = cif_factory.prepare_metadata_from_synthetic_data(name, block, model)

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = cif_factory.prepare_gemmi_structure(
        block, expand_assembly=False, clean_up=True
    )
    # Filter by residue count
    if not check_residue_count_cutoff(raw_struct, split):
        return FILTERED

    # Prepare reference structure
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd
    )
    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    # Save output if path is given
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    split: str = "long",
):
    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    # Output path
    filename: str = cif_path.name
    uniprot_id = filename.split("AF-")[1].split("-F1-")[0]
    out_path = output_dir / uniprot_id[:2] / uniprot_id[2:4] / f"{uniprot_id}.npz"
    try:
        return parse_cif(cif_path, ccd, out_path, split)
    except Exception as e:
        print(f"Failed to process ({cif_path}): {e}")
        # raise e
        return FAILED


def main():
    """Main function to process mmCIF files in parallel."""
    args = parse_args()
    id_path: pathlib.Path = args.id_path
    cif_dir: pathlib.Path = args.cif_dir

    data_dir: pathlib.Path = args.out_dir / f"afdb-{args.split}"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(id_path) as f:
        ids = set(line.strip() for line in f.readlines())

    def filter_fn(path: pathlib.Path) -> bool:
        name = path.name.split("AF-")[1].split("-F1-model")[0]
        return name in ids

    print(f"Loaded {len(ids)} synthetic data IDs.")
    cif_paths = sorted([f for f in tqdm(cif_dir.rglob("*.cif*")) if filter_fn(f)])
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    if len(cif_paths) == 0:
        print("No files to process. Exiting.")
        return

    # Run cif processing in parallel
    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    parse_cif_partial = functools.partial(
        worker_fn,
        output_dir=out_dir,
        split=args.split,
    )
    with multiprocessing.Pool(
        args.num_workers,
        initializer=init_worker,
        initargs=(args.ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_cif_partial, cif_paths, chunksize=100),
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
