"""Construct the subset of manifest file"""

import argparse
import json
import pathlib

from kfold.data.types.metadata import Metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Construct subset of manifest.")
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to the original manifest.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to the subset manifest to be created.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        choices=["complex-only", "non-nucleic", "pp-pl-only"],
    )

    args = parser.parse_args()

    return args


def main():
    """Construct the subset of manifest file."""
    args = parse_args()

    # Define filter function
    match args.filter:
        case "complex-only":
            print("Filtering out single-chain only structures.")

            def filter_func(m: Metadata) -> bool:
                # Include only if more than one chain
                return len(m.chains) > 1

        case "non-nucleic":
            print("Excluding NA-NA, NA-ligand complex structures.")

            def filter_func(m: Metadata) -> bool:
                # Exclude if any chain is nucleic acid
                return any(chain.ctype.is_nucleic_acid for chain in m.chains)

        case "pp-pl-only":
            print("Including only protein-protein and protein-ligand complex structures.")
            print("NOTE: This will exclude single-protein and single-ion complexes.")

            def filter_func(m: Metadata) -> bool:
                if any(chain.ctype.is_nucleic_acid for chain in m.chains):
                    # Exclude nucleic acid chains
                    return False
                if len([chain for chain in m.chains if not chain.ctype.is_ion]) < 2:
                    # Need at least two non-ion chains for PP or PL complex
                    return False
                return True

        case _:
            raise ValueError(f"Unknown filter option: {args.filter}")

    # Get metadatas
    print("Loading metadatas...")
    with open(args.input) as f:
        metadata_dicts: list[dict] = json.load(f)
    metadatas: list[Metadata] = [Metadata.from_dict(d) for d in metadata_dicts]
    total_count = len(metadatas)
    print(f"Total entries found: {total_count}")

    # Apply the filter
    metadatas = list(filter(filter_func, metadatas))

    # Report filtering results
    print(f"Filtered {total_count - len(metadatas)} nucleic-acid complexes ")
    print(f"Final count: {len(metadatas)}.")

    # Save the subset manifest
    print(f"Saving the subset manifest to {args.output}...")
    with open(args.output, "w") as f:
        json.dump([r.to_dict() for r in metadatas], f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
