import argparse
import json
import pathlib

import msgpack
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    metadata_csv_path: pathlib.Path = data_dir / "metadata.csv"
    df: pd.DataFrame = pd.read_csv(metadata_csv_path)
    original_count = len(df)

    if args.name == "RNA":
        df = df[(df["criterion_iptm"] >= 0.7) & (df["criterion_ipde"] <= 5.0)]
    elif args.name == "NatAb":
        df = df[df["criterion_1"] >= 0.3476]
    elif args.name.startswith("huMAP_v1"):
        df = df[
            (df["criterion_boltz_confidence"] > 0.8)
            | (df["criterion_aiupred_plddt"] > 0.8)
        ]
    constraint_count = len(df)
    print(
        f"Applied constraints to {args.name}: {constraint_count} entries out of "
        f"{original_count} ({100 * constraint_count / original_count:.2f}%)"
    )

    filtered_ids = set()
    for row in df.itertuples():
        filtered_ids.add(f"{row.data_idx}_{row.structure_idx}")

    manifest_path: pathlib.Path = data_dir / "manifest_all.msgpack"
    with open(manifest_path, "rb") as f:
        all_metadata_dicts = msgpack.unpack(f)

    # Filter metadata_dicts based on the filtered dataframe
    metadata_dicts = [md for md in all_metadata_dicts if md["id"] in filtered_ids]

    original_count = len(all_metadata_dicts)
    constraint_count = len(metadata_dicts)

    print(
        f"Filtered {constraint_count} entries out of {original_count} "
        f"({100 * constraint_count / original_count:.2f}%)"
    )

    # Save to json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    # Save to msgpack file (efficient and fast)
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadata_dicts, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    main()
