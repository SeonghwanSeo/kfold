"""Select high-confidence BioGRID rows and stream matching CIFs from tar.zst.

The output sample id is ``{data_idx}__seed_{structure_idx}``, while
``cluster_id`` is the original ``data_idx``.  This lets ClusterSampler assign
the same cluster to all passing seeds for a PPI.
"""

import argparse
import csv
import os
import pathlib
import shutil
import subprocess
import tarfile
from collections import Counter

DATASET_NAME = "Biogrid"
DEFAULT_SOURCE_DIR = pathlib.Path("/cache/wykim_lab/icl_shwan/source/260519_biogrid")
DEFAULT_DATA_DIR = pathlib.Path(
    "/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=pathlib.Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root. {DATASET_NAME}/ is appended.",
    )
    parser.add_argument("--tiers", nargs="+", choices=("T2", "T3"), default=("T2", "T3"))
    parser.add_argument("--min_ipsae", type=float, default=0.7)
    parser.add_argument("--min_iptm", type=float, default=0.7)
    parser.add_argument("--min_pdockq2", type=float, default=0.49)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report selected rows without creating metadata or extracting CIFs.",
    )
    return parser.parse_args()


def make_entry_id(data_idx: str, structure_idx: str) -> str:
    return f"{data_idx}__seed_{structure_idx}"


def make_archive_member(data_idx: str, structure_idx: str) -> str:
    return (
        "inference_cutoff21/A_env0_unpaired/runs/"
        f"{data_idx}/seed_{structure_idx}/predictions/{data_idx}_sample_0.cif"
    )


def row_passes(
    row: dict[str, str], min_ipsae: float, min_iptm: float, min_pdockq2: float
) -> bool:
    return (
        float(row["criterion_ipsae_max"]) >= min_ipsae
        and float(row["criterion_iptm"]) >= min_iptm
        and float(row["criterion_pdockq2_fast_max"]) >= min_pdockq2
    )


def select_rows(
    source_dir: pathlib.Path,
    tiers: list[str] | tuple[str, ...],
    min_ipsae: float,
    min_iptm: float,
    min_pdockq2: float,
) -> tuple[list[dict[str, str]], list[str]]:
    selected: list[dict[str, str]] = []
    source_fieldnames: list[str] | None = None
    seen_entry_ids: set[str] = set()

    for tier in tiers:
        path = source_dir / f"{tier}_metadata_cutoff21.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            if source_fieldnames is None:
                source_fieldnames = list(reader.fieldnames)
            elif reader.fieldnames != source_fieldnames:
                raise ValueError(f"Metadata schemas differ: {path}")

            for row in reader:
                if not row_passes(row, min_ipsae, min_iptm, min_pdockq2):
                    continue
                entry_id = make_entry_id(row["data_idx"], row["structure_idx"])
                if entry_id in seen_entry_ids:
                    raise ValueError(f"Duplicate selected entry id: {entry_id}")
                seen_entry_ids.add(entry_id)
                row = dict(row)
                row.update(
                    {
                        "entry_id": entry_id,
                        "cluster_id": row["data_idx"],
                        "cif_path": f"cif/{entry_id}.cif",
                        "archive_member": make_archive_member(
                            row["data_idx"], row["structure_idx"]
                        ),
                    }
                )
                selected.append(row)

    assert source_fieldnames is not None
    selected.sort(key=lambda row: row["entry_id"])
    return selected, source_fieldnames


def extract_archive(
    archive_path: pathlib.Path,
    rows: list[dict[str, str]],
    dataset_dir: pathlib.Path,
    overwrite: bool,
) -> None:
    pending: dict[str, pathlib.Path] = {}
    for row in rows:
        target = dataset_dir / row["cif_path"]
        if target.exists() and not overwrite:
            continue
        pending[row["archive_member"]] = target

    if not pending:
        print(f"{archive_path.name}: all {len(rows)} selected CIFs already exist")
        return

    print(f"{archive_path.name}: extracting {len(pending)} CIFs")
    proc = subprocess.Popen(
        ["zstd", "-dc", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    completed_early = False
    extracted = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                target = pending.pop(member.name, None)
                if target is None:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                with source, tmp_path.open("wb") as out:
                    shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
                os.replace(tmp_path, target)
                extracted += 1
                if extracted % 100 == 0:
                    print(f"  extracted {extracted}/{extracted + len(pending)}")
                if not pending:
                    completed_early = True
                    break
    finally:
        proc.stdout.close()
        if completed_early and proc.poll() is None:
            proc.terminate()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()

    if pending:
        examples = sorted(pending)[:10]
        raise FileNotFoundError(
            f"Missing {len(pending)} selected members in {archive_path}: {examples}"
        )
    if not completed_early and return_code != 0:
        raise subprocess.CalledProcessError(return_code, proc.args, stderr=stderr)
    print(f"{archive_path.name}: extracted {extracted} CIFs")


def write_metadata(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    source_fieldnames: list[str],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["entry_id", "cluster_id", "cif_path", "archive_member"]
    fieldnames.extend(name for name in source_fieldnames if name not in fieldnames)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, source_fieldnames = select_rows(
        args.source_dir,
        args.tiers,
        args.min_ipsae,
        args.min_iptm,
        args.min_pdockq2,
    )
    tier_counts = Counter(row["data_source"] for row in rows)
    ppi_counts = Counter(row["cluster_id"] for row in rows)
    print(f"Selected structures: {len(rows)}")
    print(f"Selected PPIs: {len(ppi_counts)}")
    print(f"Tier counts: {dict(sorted(tier_counts.items()))}")
    print(f"Passing seeds/PPI: {dict(sorted(Counter(ppi_counts.values()).items()))}")
    if args.dry_run:
        return

    dataset_dir = args.data_dir / DATASET_NAME
    for tier in args.tiers:
        tier_rows = [row for row in rows if row["data_source"] == tier]
        archive_path = args.source_dir / f"{tier}_inference_cutoff21.tar.zst"
        extract_archive(archive_path, tier_rows, dataset_dir, args.overwrite)

    write_metadata(dataset_dir / "metadata.csv", rows, source_fieldnames, args.overwrite)
    print(f"Wrote: {dataset_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
