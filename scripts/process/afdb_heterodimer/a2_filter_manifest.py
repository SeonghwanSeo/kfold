"""Filter AFDB heterodimer manifests by ipSAE metadata.

This script expects ``metadata.csv`` to contain one row per AF-M model with:

    modelEntityId, max_ipSAE

It writes ``manifest_08.msgpack`` and, when ``manifest.json`` exists,
``manifest_08.json``.
"""

import argparse
import csv
import json
import pathlib

import msgpack

DATASET_NAME = "AFDB-heterodimer"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help=f"Root processed dataset directory. {DATASET_NAME}/ is appended.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=pathlib.Path,
        default=None,
        help=f"CSV path. Defaults to {DATASET_NAME}/metadata.csv.",
    )
    parser.add_argument(
        "--name_key",
        type=str,
        default="modelEntityId",
        help="CSV column matched against manifest entry id.",
    )
    parser.add_argument(
        "--score_key",
        type=str,
        default="max_ipSAE",
        help="CSV column used for filtering.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="-model_v1",
        help="Suffix appended to metadata ids before matching manifest ids.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Keep entries with score_key >= threshold.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="08",
        help="Output manifest suffix: manifest_<suffix>.*",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered manifests.",
    )
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> list[dict]:
    with path.open("rb") as f:
        return msgpack.unpack(f, raw=False)


def write_manifest_msgpack(path: pathlib.Path, manifest: list[dict]) -> None:
    with path.open("wb") as f:
        msgpack.pack(manifest, f)


def write_manifest_json(path: pathlib.Path, manifest: list[dict]) -> None:
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def load_passing_ids(
    metadata_csv: pathlib.Path,
    name_key: str,
    score_key: str,
    threshold: float,
    name_suffix: str,
) -> set[str]:
    passing_ids = set()
    with metadata_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{metadata_csv} has no CSV header.")
        missing_keys = {name_key, score_key} - set(reader.fieldnames)
        if missing_keys:
            raise KeyError(
                f"{metadata_csv} is missing required columns: {sorted(missing_keys)}"
            )

        for row in reader:
            sample_id = row[name_key].strip()
            if not sample_id:
                continue
            if name_suffix and not sample_id.endswith(name_suffix):
                sample_id = f"{sample_id}{name_suffix}"
            score_value = row[score_key].strip()
            if not score_value:
                continue
            if float(score_value) >= threshold:
                passing_ids.add(sample_id)

    return passing_ids


def check_writable(path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")


def filter_manifest(manifest: list[dict], passing_ids: set[str]) -> list[dict]:
    return [entry for entry in manifest if entry["id"] in passing_ids]


def main():
    args = parse_args()
    dataset_dir = args.data_dir / DATASET_NAME
    metadata_csv = args.metadata_csv or (dataset_dir / "metadata.csv")

    manifest_path = dataset_dir / "manifest.msgpack"
    manifest_json_path = dataset_dir / "manifest.json"
    output_msgpack_path = dataset_dir / f"manifest_{args.output_suffix}.msgpack"
    output_json_path = dataset_dir / f"manifest_{args.output_suffix}.json"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.msgpack not found: {manifest_path}")

    check_writable(output_msgpack_path, args.overwrite)
    if manifest_json_path.exists():
        check_writable(output_json_path, args.overwrite)

    passing_ids = load_passing_ids(
        metadata_csv,
        name_key=args.name_key,
        score_key=args.score_key,
        threshold=args.threshold,
        name_suffix=args.name_suffix,
    )
    manifest = load_manifest(manifest_path)
    filtered_manifest = filter_manifest(manifest, passing_ids)

    write_manifest_msgpack(output_msgpack_path, filtered_manifest)
    print(f"Wrote: {output_msgpack_path}")
    if manifest_json_path.exists():
        write_manifest_json(output_json_path, filtered_manifest)
        print(f"Wrote: {output_json_path}")

    manifest_ids = {entry["id"] for entry in manifest}
    matched_ids = manifest_ids & passing_ids
    print(f"Manifest entries: {len(manifest)}")
    print(f"Passing metadata ids: {len(passing_ids)}")
    print(f"Matched passing entries: {len(matched_ids)}")
    print(f"Filtered entries: {len(filtered_manifest)}")


if __name__ == "__main__":
    main()
