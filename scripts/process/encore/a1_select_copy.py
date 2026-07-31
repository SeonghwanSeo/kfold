"""Select confidence-filtered ENCORE rows and copy their Protenix CIF files.

Each prediction is a separate training entry. Predictions generated for the same
provider sample share the CIF data-block sample name as ``cluster_id`` so that
ClusterSampler normalizes their aggregate sampling mass by the passing seed count.
"""

import argparse
import csv
import os
import pathlib
import shutil
from collections import Counter

DATASET_NAME = "ENCORE"
DEFAULT_SOURCE_DIR = pathlib.Path("/cache/wykim_lab/icl_shwan/source/ENCORE")
DEFAULT_DATA_DIR = pathlib.Path(
    "/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset"
)
DEFAULT_MIN_IPTM = 0.6
DEFAULT_MAX_IPDE = 3.0
IPDE_COLUMN = "chain_pair_gpde_offdiag_mean"
PREDICTION_SUFFIX = "_predicted_by_protenix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=pathlib.Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--data_dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min_iptm", type=float, default=DEFAULT_MIN_IPTM)
    parser.add_argument("--max_ipde", type=float, default=DEFAULT_MAX_IPDE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def make_entry_id(data_idx: str, structure_idx: str) -> str:
    return f"encore_{data_idx}__structure_{structure_idx}"


def source_cif_path(source_dir: pathlib.Path, row: dict[str, str]) -> pathlib.Path:
    data_idx = row["data_idx"]
    shard = data_idx.split("_", 1)[0]
    return (
        source_dir / "holo" / shard / data_idx / f"structure_{row['structure_idx']}.cif"
    )


def read_sample_name(cif_path: pathlib.Path) -> str:
    with cif_path.open() as f:
        data_block = f.readline().strip()
    if not data_block.startswith("data_"):
        raise ValueError(f"Missing CIF data-block name in {cif_path}: {data_block!r}")
    sample_name = data_block.removeprefix("data_")
    if sample_name.endswith(PREDICTION_SUFFIX):
        sample_name = sample_name.removesuffix(PREDICTION_SUFFIX)
    if not sample_name:
        raise ValueError(f"Empty sample name in {cif_path}")
    return sample_name


def row_passes(row: dict[str, str], min_iptm: float, max_ipde: float) -> bool:
    return (
        int(float(row["distillation"])) == 1
        and float(row["iptm"]) >= min_iptm
        and float(row[IPDE_COLUMN]) <= max_ipde
    )


def select_rows(
    metadata_path: pathlib.Path,
    source_dir: pathlib.Path,
    min_iptm: float,
    max_ipde: float,
) -> tuple[list[dict[str, str]], list[str]]:
    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {metadata_path}")
        source_fieldnames = list(reader.fieldnames)
        source_rows = list(reader)

    required = {"data_idx", "structure_idx", "distillation", "iptm", IPDE_COLUMN}
    missing = required - set(source_fieldnames)
    if missing:
        raise KeyError(f"Missing ENCORE metadata columns: {sorted(missing)}")

    selected: list[dict[str, str]] = []
    seen_entry_ids: set[str] = set()
    for source_row in source_rows:
        if not row_passes(source_row, min_iptm, max_ipde):
            continue
        row = dict(source_row)
        entry_id = make_entry_id(row["data_idx"], row["structure_idx"])
        if entry_id in seen_entry_ids:
            raise ValueError(f"Duplicate selected entry id: {entry_id}")
        seen_entry_ids.add(entry_id)

        source_path = source_cif_path(source_dir, row)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        row.update(
            {
                "entry_id": entry_id,
                "cluster_id": read_sample_name(source_path),
                "cif_path": f"cif/{entry_id}.cif",
                "source_cif_path": str(source_path),
            }
        )
        selected.append(row)

    selected.sort(key=lambda row: row["entry_id"])
    return selected, source_fieldnames


def copy_cifs(
    rows: list[dict[str, str]], dataset_dir: pathlib.Path, overwrite: bool
) -> None:
    copied = 0
    existing = 0
    for row in rows:
        source = pathlib.Path(row["source_cif_path"])
        target = dataset_dir / row["cif_path"]
        if target.exists() and not overwrite:
            existing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        copied += 1
    print(f"CIF files copied: {copied}; already present: {existing}")


def write_metadata(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    source_fieldnames: list[str],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    added = ["entry_id", "cluster_id", "cif_path", "source_cif_path"]
    fieldnames = added + [name for name in source_fieldnames if name not in added]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_path = args.source_dir / "metadata.csv"
    rows, source_fieldnames = select_rows(
        metadata_path, args.source_dir, args.min_iptm, args.max_ipde
    )
    cluster_sizes = Counter(row["cluster_id"] for row in rows)
    print(f"Selected structures: {len(rows)}")
    print(f"Provider sample clusters: {len(cluster_sizes)}")
    print(
        "Passing structures/sample: "
        f"{dict(sorted(Counter(cluster_sizes.values()).items()))}"
    )
    print(
        f"Criteria: distillation == 1, ipTM >= {args.min_iptm}, ipDE <= {args.max_ipde}"
    )
    if args.dry_run:
        return

    dataset_dir = args.data_dir / DATASET_NAME
    copy_cifs(rows, dataset_dir, args.overwrite)
    write_metadata(dataset_dir / "metadata.csv", rows, source_fieldnames, args.overwrite)
    print(f"Wrote: {dataset_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
