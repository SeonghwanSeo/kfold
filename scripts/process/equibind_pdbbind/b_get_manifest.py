"""Get train/validation/test split used in EquiBind: Stärk et al, 2022.

This includes multiple pre-filtering steps:
    1. Max chain filtering (>300 chains)
    2. Nucleic acid filtering (optional)

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
        "--split_path",
        type=Path,
        help="Path to the directory containing equibind time-splits.",
        default="./assets/splits/equibind/",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the save the output manifest file.",
        default="/cache/wykim_lab/kfold-data/manifest/pdbbind_manifest.json",
    )
    parser.add_argument(
        "--exclude_large_complex",
        action="store_true",
        help="Whether to filter out structures with more than 300 chains.",
    )
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

    total_count = len(record_dicts)
    logger.info(
        f"Loaded {total_count} total records in {time.time() - st_time:.2f} seconds."
    )

    # Load splits
    with open(args.split_path / "train_ids.txt") as f:
        train_ids = set(line.strip() for line in f)
    with open(args.split_path / "validation_ids.txt") as f:
        val_ids = set(line.strip() for line in f)
    with open(args.split_path / "test_ids.txt") as f:
        test_ids = set(line.strip() for line in f)
    all_ids = sorted(list(train_ids)) + sorted(list(val_ids)) + sorted(list(test_ids))
    all_ids = [k.lower() for k in all_ids]
    logger.info(
        f"Loaded EquiBind splits: {len(train_ids)} train, "
        f"{len(val_ids)} val, {len(test_ids)} test."
    )
    logger.info(f"Total of {len(all_ids)} unique structure IDs in the splits.")

    record_dicts = [r for r in record_dicts if r["id"].lower() in all_ids]
    all_records: list[Metadata] = [parse_record(r) for r in record_dicts]
    total_count = len(all_records)
    logger.info(f"Filtered to {total_count} records based on EquiBind splits")

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
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format = output_path.suffix.lower()
    dict_list = [r.to_dict() for r in all_records]
    if format == ".json":
        with open(output_path, "w") as f:
            json.dump(dict_list, f, indent=2)
    elif format in {".pkl", ".pickle"}:
        with open(output_path, "wb") as f:
            pickle.dump(dict_list, f)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    logger.info(
        f"Saved filtered manifest to {output_path} in "
        f"{time.time() - start_time:.2f} seconds."
    )

    # Check loading time
    start_time = time.time()
    if format in {".pkl", ".pickle"}:
        with open(output_path, "rb") as f:
            res = pickle.load(f)
    else:
        with open(output_path) as f:
            res = json.load(f)
    manifest: list[Metadata] = [Metadata.from_dict(r) for r in res]  # noqa: F841
    logger.info(
        f"Verified loading filtered manifest in {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
