"""Get train/validation/test split used in Boltz1

This includes multiple pre-filtering steps:
    1. Max chain filtering (>300 chains)
    2. Single-chain filtering (optional)
    3. Nucleic acid filtering (optional)

TODO:
    - For single-chain filtering, consider adding a minimum length threshold
      to identify meaningful chains. (e.g., ignore a single metal ion)
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import kfold.constants as C
from kfold.data.metadata import Metadata
from kfold.utils.boltz.process import parse_record

logger = logging.getLogger(__name__)


# FIXME: remove default path before publish
def parse_args():
    parser = argparse.ArgumentParser(
        description="Save apo-holo pairs from a dataset of protein structures."
    )
    parser.add_argument(
        "--boltz_manifest_path",
        type=Path,
        help="Path to the input file containing boltz1 processed dataset.",
        default="/cache/wykim_lab/rcsb_processed_targets/manifest.json",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the save the output manifest file.",
        default="/cache/wykim_lab/kfold-data/manifest/af3_manifest.json",
    )
    parser.add_argument(
        "--exclude_large_complex",
        action="store_true",
        help="Whether to filter out structures with more than 300 chains.",
    )
    parser.add_argument(
        "--exclude_single_chain",
        action="store_true",
        help="Whether to filter out single-chain only structures.",
    )
    # FIXME: remove this argument after extending modality (DNA, RNA)
    parser.add_argument(
        "--exclude_nucleic_acids",
        action="store_true",
        help="Whether to filter out nucleic-acid containing structures.",
    )
    return parser.parse_args()


def main(args):
    # Load records
    st_time = time.time()
    with open(args.boltz_manifest_path) as f:
        record_dicts = json.load(f)
    all_records: list[Metadata] = [parse_record(r) for r in record_dicts]

    total_count = len(all_records)
    logger.info(
        f"Loaded {total_count} total structures in {time.time() - st_time:.2f} seconds."
    )

    # Apply filters
    if args.exclude_large_complex:
        st_time = time.time()
        logger.info("Filtering out structures with more than 300 chains.")
        prev_count = len(all_records)
        all_records = [r for r in all_records if r.num_chains <= 300]
        logger.info(
            f"Filtered {prev_count - len(all_records)} large-complex structures "
            f"in {time.time() - st_time:.2f} seconds."
        )

    if args.exclude_single_chain:
        st_time = time.time()
        logger.info("Filtering out single-chain only structures.")
        prev_count = len(all_records)
        all_records = [r for r in all_records if r.num_chains > 1]
        logger.info(
            f"Filtered {prev_count - len(all_records)} single-chain structures "
            f"in {time.time() - st_time:.2f} seconds."
        )

    if args.exclude_nucleic_acids:
        st_time = time.time()
        logger.info("Excluding NA-NA, NA-ligand complex structures.")

        def has_nucleic_acid(record: Metadata) -> bool:
            return any(
                chain.chain_type in {C.chain.ChainType.DNA, C.chain.ChainType.RNA}
                for chain in record.chains
            )

        prev_count = len(all_records)
        all_records = [r for r in all_records if not has_nucleic_acid(r)]
        logger.info(
            f"Filtered {prev_count - len(all_records)} nucleic-acid complexes "
            f"in {time.time() - st_time:.2f} seconds."
        )

    logger.info(
        f"Final dataset contains {len(all_records)} structures "
        f"out of {total_count} ({total_count - len(all_records)} filtered)."
    )

    start_time = time.time()
    # Save the filtered manifest
    format = Path(args.output_path).suffix.lower()
    dict_list = [r.to_dict() for r in all_records]
    if format == ".json":
        with open(args.output_path, "w") as f:
            json.dump(dict_list, f, indent=2)
    elif format in {".pkl", ".pickle"}:
        with open(args.output_path, "wb") as f:
            pickle.dump(dict_list, f)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    logger.info(
        f"Saved filtered manifest to {args.output_path} in "
        f"{time.time() - start_time:.2f} seconds."
    )

    # Check loading time
    start_time = time.time()
    if format in {".pkl", ".pickle"}:
        with open(args.output_path, "rb") as f:
            res = pickle.load(f)
    else:
        with open(args.output_path) as f:
            res = json.load(f)
    manifest: list[Metadata] = [Metadata.from_dict(r) for r in res]  # noqa: F841
    logger.info(
        f"Verified loading filtered manifest in {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
