"""Save ccd data from RCSB to the local database."""

import argparse
import logging
import os
import pathlib
import pickle

from rdkit import Chem, RDLogger, rdBase
from tqdm import tqdm

from kfold.data.processing.component import CCD, Component

# Set property saving
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")
rdBase.BlockLogs()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "--boltz_ccd_path",
        type=pathlib.Path,
        required=True,
        help="Path to the ccd.pkl file of Boltz1.",
    )
    parser.add_argument(
        "--output_path",
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
    return parser.parse_args()


def main():
    """Main function to process CCD data."""
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load Boltz1 CCD data
    logger.info("Loading Boltz1 CCD data...")
    with open(args.boltz_ccd_path, "rb") as f:
        boltz_ccd: dict[str, Chem.Mol] = pickle.load(f)

    # Convert Boltz1 CCD data to ours
    components: dict[str, Component] = {}
    for code, mol in tqdm(boltz_ccd.items(), desc="Converting Boltz1 CCD to Component"):
        component = Component.from_mol(
            code,
            mol,
            ideal_conf_id=0,
            model_conf_id=1,
            etkdg_conf_ids=[2],
        )
        components[code] = component

    ccd = CCD(components)
    ccd.save(args.output_path)
    logger.info("CCD data processing completed.")


if __name__ == "__main__":
    main()
