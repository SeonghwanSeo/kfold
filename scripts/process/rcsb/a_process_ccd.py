"""Save ccd data from RCSB to the local database."""

import argparse
import datetime
import logging
import multiprocessing
import os
import pathlib
import sys

import gemmi
import numpy as np
from rdkit import Chem, RDLogger, rdBase
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component

try:
    # pdbeccdutils is required for reading RCSB CCD data
    from pdbeccdutils.core import ccd_reader
except ImportError as e:
    raise ImportError(
        "pdbeccdutils is required for this script. "
        "Please install it via 'pip install pdbeccdutils'."
    ) from e

# Set property saving
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")
rdBase.BlockLogs()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "-i",
        "--cif_path",
        type=pathlib.Path,
        required=True,
        help="Path to the components.cif file from RCSB.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=pathlib.Path,
        required=True,
        help="Path (.pkl) to save the processed CCD data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    # Conformer generation parameters
    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to indicate if the data is for training purposes.",
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        help="Number of conformers to generate for each component.",
    )
    parser.add_argument(
        "--num_conformers_standard_residues",
        type=int,
        help="Number of conformers for standard residues (if different).",
    )
    parser.add_argument(
        "--date_cutoff",
        type=str,
        default="2021-09-30",
        help="Date cutoff for processing components (YYYY-MM-DD).",
    )
    return parser.parse_args()


def shift_seed(seed: int, code: str) -> int:
    """Shift seed based on component code for variability."""
    return seed + sum(ord(c) for c in code) % 1000


def process_pdbe_ccd_component_and_save(
    code: str,
    mol: Chem.Mol,
    ccd_cif_string: str,
    num_confs: int,
    compute_symmetry: bool,
    date_cutoff: datetime.date,
    seed: int,
    safety_mode: bool = True,
) -> dict | None:
    """Convert PDBeCCDComponent to Component."""
    seed = shift_seed(seed, code)
    rng = np.random.default_rng(seed)

    ccd_cif_block = gemmi.cif.read_string(ccd_cif_string).sole_block()
    try:
        comp = Component.from_ccd_cif(
            code=code,
            mol=mol,
            cif_block=ccd_cif_block,
            num_confs=num_confs,
            compute_symmetry=compute_symmetry,
            date_cutoff=date_cutoff,
            timeout=120,
            rng=rng,
        )
    except Exception as e:
        if safety_mode:
            print(f"Error processing component {code}: {e}")
            return
        else:
            raise e
    return comp.to_dict()


def _process_wrapper(args_bundle) -> tuple[str, dict | None]:
    code, mol, ccd_cif_string = args_bundle["dynamic"]
    static = args_bundle["static"]

    if code in C.residue.STANDARD_RESIDUES_STR:
        nconfs = static["num_confs_standard_residues"] or static["num_confs"]
    else:
        nconfs = static["num_confs"]

    comp_state = process_pdbe_ccd_component_and_save(
        code,
        mol,
        ccd_cif_string=ccd_cif_string,
        num_confs=nconfs,
        compute_symmetry=static["compute_symmetry"],
        date_cutoff=static["date_cutoff"],
        seed=static["seed"],
        safety_mode=static["safety_mode"],
    )
    return (code, comp_state)


def construct_ccd(
    cif_path: pathlib.Path,
    num_confs: int,
    num_confs_standard_residues: int | None,
    compute_symmetry: bool,
    date_cutoff: datetime.date | None,
    seed: int,
    num_workers: int,
    logger: logging.Logger,
) -> CCD:
    # Load CCD components
    logger.info("Reading CCD components from CIF file...")
    pdbe_results = ccd_reader.read_pdb_components_file(str(cif_path))
    total_count = len(pdbe_results)
    logger.info(f"Total components in CIF: {total_count}")

    # Reset stdout and stderr, as pdbccdutils messes with them
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Prepare static arguments for processing
    static_args = {
        "num_confs": num_confs,
        "num_confs_standard_residues": num_confs_standard_residues,
        "compute_symmetry": compute_symmetry,
        "date_cutoff": date_cutoff,
        "seed": seed,
    }
    if num_workers > 1:
        static_args["safety_mode"] = True  # Avoid crashes in multiprocessing
    else:
        static_args["safety_mode"] = False  # Raise exceptions directly

    # Generator for tasks (to avoid overhead due to cif string conversion)
    def task_generator():
        for code, result in pdbe_results.items():
            # Extract RDKit molecule and CIF string (to avoid serialization issues)
            mol: Chem.Mol = result.component.mol
            cif_string: str = result.component.ccd_cif_block.as_string()

            # Yield dynamic and static arguments
            yield {"dynamic": (code, mol, cif_string), "static": static_args}

    # Process components in parallel
    logger.info("Processing components...")
    with multiprocessing.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_wrapper, task_generator()),
                total=total_count,
            )
        )

    # Collect processed components
    components = {}
    for code, comp_state in results:
        if comp_state is not None:
            components[code] = Component.from_dict(comp_state)
    logger.info(f"Successfully processed {len(components)} components.")

    return CCD(components)


def main():
    """Main function to process CCD data."""
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.cif_path.exists():
        logger.info("Downloading components.cif from RCSB...")
        os.makedirs(args.cif_path.parent, exist_ok=True)
        link = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif"
        os.system(f"wget {link} -O {args.cif_path}")
        logger.info("Download completed.")

    if args.train:
        logger.info("Processing CCD data for training.")
        if args.num_conformers is None:
            logger.warning(
                "Number of conformers not specified or non-positive. "
                "Defaulting to 10 conformers for training data."
            )
            args.num_conformers = 10
        if args.num_conformers_standard_residues is None:
            logger.warning(
                "Number of conformers for standard residues not specified. "
                "Defaulting to 100 conformers for standard residues."
            )
            args.num_conformers_standard_residues = 100
        compute_symmetry = True
    else:
        logger.info("Processing CCD data for evaluation.")
        if args.num_conformers is None:
            logger.warning(
                "Number of conformers not specified or non-positive. "
                "Defaulting to 0 conformers for evaluation data."
            )
            args.num_conformers = 0
        if args.num_conformers_standard_residues is None:
            logger.warning(
                "Number of conformers for standard residues not specified. "
                "Defaulting to 100 conformers for standard residues."
            )
            args.num_conformers_standard_residues = 100
        compute_symmetry = False

    if args.num_conformers > 0:
        nconfs = args.num_conformers
        logger.info(
            f"Generating {args.num_conformers} conformers per component."
            f" (Random seed set to: {args.seed})"
        )
    if args.num_conformers_standard_residues is not None:
        nconfs = args.num_conformers_standard_residues
        logger.info(
            f"Standard residues will have {nconfs} conformers."
            f" (Random seed set to: {args.seed})"
        )

    if args.date_cutoff == "today":
        logger.info("Using today's date as cutoff.")
        date_cutoff = datetime.date.today()
    else:
        logger.info(f"Using date cutoff: {args.date_cutoff}")
        date_cutoff = datetime.date.fromisoformat(args.date_cutoff)

    ccd = construct_ccd(
        cif_path=args.cif_path,
        num_confs=args.num_conformers,
        num_confs_standard_residues=args.num_conformers_standard_residues,
        compute_symmetry=compute_symmetry,
        date_cutoff=date_cutoff,
        seed=args.seed,
        num_workers=args.num_workers,
        logger=logger,
    )
    ccd.save(args.out_path)
    logger.info("CCD data processing completed.")


if __name__ == "__main__":
    main()
