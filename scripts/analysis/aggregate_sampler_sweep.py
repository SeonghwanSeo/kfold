"""Create provenance-preserving summaries for an ECSI sampler sweep.

The input directory contains one run root per sweep wave. This script never
collapses OST and DockQv2 rows into one modality: each raw scorer file remains
a separate metric surface with its own hash and target-level outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


LIGAND_KEY = ("pdb_id", "native_chain_id_1", "native_chain_id_2", "ligand_id")
INTERFACE_KEY = ("pdb_id", "interface_chain_id_1", "interface_chain_id_2")
DOCKQ_THRESHOLDS = {
    "dockq_acc": 0.23,
    "dockq_med": 0.49,
    "dockq_high": 0.80,
}
METRIC_SOURCES = ("ost", "dockqv2")
PRIMARY_SOURCE_OVERRIDES = {
    "protein_dna": "dockqv2",
    "protein_rna": "dockqv2",
}
SAMPLER_SETTINGS = (
    "gamma_power",
    "sampler_mode",
    "sampler_ode_type",
    "sampler_switch_time",
    "sampler_sde_atom_classes",
    "churn_factor",
    "churn_max_multiplier",
    "churn_end_time",
    "churn_max_time",
    "churn_step_fraction",
    "churn_step_power",
    "ode_step_power",
    "stepwarp_power",
    "svgd_step",
)

TARGET_FIELDS = (
    "root",
    "arm",
    "modality",
    "metric_source",
    "is_primary_surface",
    "metric_name",
    "target_key",
    "sample_count",
    "oracle_success",
    "rank_success",
    "oracle_dockq",
    "rank_dockq",
    "oracle_rmsd",
    "oracle_lddtpli",
    "rank_rmsd",
    "rank_lddtpli",
    "key_columns",
    "raw_csv",
    "raw_sha256",
    "raw_row_count",
    "config_sha256",
    "manifest_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "target_query_count",
    "scored_marker",
    *SAMPLER_SETTINGS,
)
SUMMARY_FIELDS = (
    "root",
    "arm",
    "modality",
    "metric_source",
    "is_primary_surface",
    "metric_name",
    "target_count",
    "oracle_success_count",
    "rank_success_count",
    "oracle_rank_gap",
    "sample_count_min",
    "sample_count_max",
    "key_columns",
    "raw_csv",
    "raw_sha256",
    "raw_row_count",
    "config_sha256",
    "manifest_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "target_query_count",
    "scored_marker",
    *SAMPLER_SETTINGS,
)
PAIRED_FIELDS = (
    "baseline_root",
    "baseline_arm",
    "candidate_root",
    "candidate_arm",
    "modality",
    "metric_source",
    "is_primary_surface",
    "metric_name",
    "outcome",
    "paired_target_count",
    "baseline_success_count",
    "candidate_success_count",
    "candidate_only_success_count",
    "baseline_only_success_count",
    "two_sided_exact_mcnemar_p",
    "baseline_unpaired_target_count",
    "candidate_unpaired_target_count",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an immutable input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(row: Mapping[str, str], column: str, path: Path) -> float:
    """Read one finite numerical metric or fail rather than silently dropping it."""
    raw = row.get(column)
    if raw is None or raw == "":
        raise ValueError(f"{path}: missing required {column!r} value")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {column!r} value {raw!r}")
    return value


def metric_surface_from_path(path: Path) -> tuple[str, str] | None:
    """Extract ``(modality, scorer)`` from an interface evaluator filename."""
    for source in METRIC_SOURCES:
        suffix = f"_{source}.csv"
        if path.name.startswith("interface_") and path.name.endswith(suffix):
            modality = path.name[len("interface_") : -len(suffix)]
            return modality, source
    return None


def primary_source_for_modality(modality: str) -> str:
    """Use DockQv2 for nucleic-acid surfaces and OST elsewhere."""
    return PRIMARY_SOURCE_OVERRIDES.get(modality, "ost")


def stable_target_key(
    row: Mapping[str, str],
    columns: Sequence[str],
    path: Path,
) -> str:
    """Encode a full target identifier without losing the source column names."""
    missing = [column for column in columns if row.get(column) in (None, "")]
    if missing:
        raise ValueError(f"{path}: missing target-key columns {missing}")
    return json.dumps({column: row[column] for column in columns}, sort_keys=True)


def read_raw_rows(path: Path) -> list[dict[str, str]]:
    """Read one raw scorer CSV and require a real header and at least one row."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV has no metric rows")
    return rows


def required_columns(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    path: Path,
) -> None:
    """Fail early if a scorer export cannot support the requested metric surface."""
    available = set(rows[0])
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def group_rows(
    rows: Iterable[Mapping[str, str]],
    key_columns: Sequence[str],
    path: Path,
) -> dict[str, list[Mapping[str, str]]]:
    """Group raw samples by their lossless JSON target key."""
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[stable_target_key(row, key_columns, path)].append(row)
    return grouped


def base_target_record(
    *,
    run_root: str,
    arm: str,
    modality: str,
    metric_source: str,
    metric_name: str,
    target_key: str,
    sample_count: int,
    key_columns: Sequence[str],
    raw_path: Path,
    raw_sha256: str,
    raw_row_count: int,
    config_sha256: str,
    manifest_sha256: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    target_query_count: int | None,
    scored_marker: bool,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build fields shared by every target-level metric row."""
    record: dict[str, Any] = {
        "root": run_root,
        "arm": arm,
        "modality": modality,
        "metric_source": metric_source,
        "is_primary_surface": metric_source == primary_source_for_modality(modality),
        "metric_name": metric_name,
        "target_key": target_key,
        "sample_count": sample_count,
        "key_columns": json.dumps(key_columns),
        "raw_csv": str(raw_path),
        "raw_sha256": raw_sha256,
        "raw_row_count": raw_row_count,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "target_query_count": target_query_count,
        "scored_marker": scored_marker,
    }
    record.update({setting: settings.get(setting) for setting in SAMPLER_SETTINGS})
    return record


def protein_ligand_records(
    *,
    grouped: Mapping[str, Sequence[Mapping[str, str]]],
    base_kwargs: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    """Return PL joint-success target records for oracle and confidence rank."""
    records: list[dict[str, Any]] = []
    for target_key, samples in grouped.items():
        ranked = max(samples, key=lambda row: parse_float(row, "ranking_score", path))
        rmsds = [parse_float(row, "rmsd", path) for row in samples]
        lddtplis = [parse_float(row, "lddt-pli", path) for row in samples]
        oracle_success = any(
            rmsd < 2.0 and lddtpli > 0.8
            for rmsd, lddtpli in zip(rmsds, lddtplis, strict=True)
        )
        rank_rmsd = parse_float(ranked, "rmsd", path)
        rank_lddtpli = parse_float(ranked, "lddt-pli", path)
        record = base_target_record(
            metric_name="pl_joint",
            target_key=target_key,
            sample_count=len(samples),
            **base_kwargs,
        )
        record.update(
            {
                "oracle_success": oracle_success,
                "rank_success": rank_rmsd < 2.0 and rank_lddtpli > 0.8,
                "oracle_rmsd": min(rmsds),
                "oracle_lddtpli": max(lddtplis),
                "rank_rmsd": rank_rmsd,
                "rank_lddtpli": rank_lddtpli,
            }
        )
        records.append(record)
    return records


def docking_records(
    *,
    grouped: Mapping[str, Sequence[Mapping[str, str]]],
    base_kwargs: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    """Return one target record per DockQ success threshold."""
    records: list[dict[str, Any]] = []
    for target_key, samples in grouped.items():
        scores = [parse_float(row, "dockq_score", path) for row in samples]
        ranked = max(samples, key=lambda row: parse_float(row, "ranking_score", path))
        oracle_dockq = max(scores)
        rank_dockq = parse_float(ranked, "dockq_score", path)
        for metric_name, threshold in DOCKQ_THRESHOLDS.items():
            record = base_target_record(
                metric_name=metric_name,
                target_key=target_key,
                sample_count=len(samples),
                **base_kwargs,
            )
            record.update(
                {
                    "oracle_success": oracle_dockq >= threshold,
                    "rank_success": rank_dockq >= threshold,
                    "oracle_dockq": oracle_dockq,
                    "rank_dockq": rank_dockq,
                }
            )
            records.append(record)
    return records


def settings_from_config(path: Path) -> dict[str, Any]:
    """Read only sampler settings from one resolved arm configuration."""
    payload = yaml.safe_load(path.read_text())
    try:
        head = payload["model"]["diffusion_head"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing model.diffusion_head") from error
    if not isinstance(head, Mapping):
        raise ValueError(f"{path}: model.diffusion_head is not a mapping")
    settings: dict[str, Any] = {}
    for setting in SAMPLER_SETTINGS:
        value = head.get(setting)
        if isinstance(value, (list, tuple)):
            settings[setting] = json.dumps(value)
        else:
            settings[setting] = value
    return settings


def collect_target_records(sweep_root: Path) -> list[dict[str, Any]]:
    """Collect target-level results while retaining every scorer's provenance."""
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(sweep_root.glob("*/sweep_manifest.json")):
        run_path = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        arms = manifest.get("arms")
        if not isinstance(arms, list) or not all(isinstance(arm, str) for arm in arms):
            raise ValueError(f"{manifest_path}: arms must be a list of strings")
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{manifest_path}: missing checkpoint identity")
        checkpoint_path = checkpoint.get("path")
        checkpoint_sha256 = checkpoint.get("sha256")
        if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_sha256, str):
            raise ValueError(f"{manifest_path}: invalid checkpoint identity")
        targets = manifest.get("targets")
        target_query_count = (
            targets.get("query_count") if isinstance(targets, Mapping) else None
        )
        if target_query_count is not None and not isinstance(target_query_count, int):
            raise ValueError(f"{manifest_path}: targets.query_count must be an integer")
        manifest_sha256 = sha256_file(manifest_path)

        for arm in sorted(arms):
            config_path = run_path / "configs" / f"{arm}.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Missing resolved config for {run_path.name}/{arm}"
                )
            settings = settings_from_config(config_path)
            config_sha256 = sha256_file(config_path)
            raw_dir = run_path / "evaluation" / arm / "raw"
            if not raw_dir.is_dir():
                continue

            for raw_path in sorted(raw_dir.glob("interface_*.csv")):
                surface = metric_surface_from_path(raw_path)
                if surface is None:
                    continue
                modality, metric_source = surface
                rows = read_raw_rows(raw_path)
                key_columns = (
                    LIGAND_KEY if modality == "protein_ligand" else INTERFACE_KEY
                )
                required = [*key_columns, "ranking_score"]
                required.extend(
                    ("rmsd", "lddt-pli")
                    if modality == "protein_ligand"
                    else ("dockq_score",)
                )
                required_columns(rows, required, raw_path)
                grouped = group_rows(rows, key_columns, raw_path)
                base_kwargs = {
                    "run_root": run_path.name,
                    "arm": arm,
                    "modality": modality,
                    "metric_source": metric_source,
                    "key_columns": key_columns,
                    "raw_path": raw_path.relative_to(sweep_root),
                    "raw_sha256": sha256_file(raw_path),
                    "raw_row_count": len(rows),
                    "config_sha256": config_sha256,
                    "manifest_sha256": manifest_sha256,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha256,
                    "target_query_count": target_query_count,
                    "scored_marker": (
                        run_path / "evaluation" / arm / ".scored"
                    ).is_file(),
                    "settings": settings,
                }
                if modality == "protein_ligand":
                    records.extend(
                        protein_ligand_records(
                            grouped=grouped,
                            base_kwargs=base_kwargs,
                            path=raw_path,
                        )
                    )
                else:
                    records.extend(
                        docking_records(
                            grouped=grouped,
                            base_kwargs=base_kwargs,
                            path=raw_path,
                        )
                    )
    return records


def summary_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize target records without co-mingling scorer surfaces."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    group_fields = (
        "root",
        "arm",
        "modality",
        "metric_source",
        "is_primary_surface",
        "metric_name",
        "key_columns",
        "raw_csv",
        "raw_sha256",
        "raw_row_count",
        "config_sha256",
        "manifest_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "target_query_count",
        "scored_marker",
        *SAMPLER_SETTINGS,
    )
    for record in records:
        grouped[tuple(record[field] for field in group_fields)].append(record)

    summaries: list[dict[str, Any]] = []
    for group_key, group_records in sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        summary = dict(zip(group_fields, group_key, strict=True))
        oracle_count = sum(bool(record["oracle_success"]) for record in group_records)
        rank_count = sum(bool(record["rank_success"]) for record in group_records)
        sample_counts = [int(record["sample_count"]) for record in group_records]
        summary.update(
            {
                "target_count": len(group_records),
                "oracle_success_count": oracle_count,
                "rank_success_count": rank_count,
                "oracle_rank_gap": oracle_count - rank_count,
                "sample_count_min": min(sample_counts),
                "sample_count_max": max(sample_counts),
            }
        )
        summaries.append(summary)
    return summaries


def exact_two_sided_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant target pairs."""
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, count)
        for count in range(min(candidate_only, baseline_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_root: str,
    baseline_arm: str,
    candidate_root: str,
    candidate_arm: str,
) -> list[dict[str, Any]]:
    """Compare two arms on shared target keys for each isolated scorer surface."""
    all_records = list(records)
    surfaces: dict[
        tuple[str, str, bool, str],
        dict[str, dict[str, Mapping[str, Any]]],
    ] = {}
    identities = {
        "baseline": (baseline_root, baseline_arm),
        "candidate": (candidate_root, candidate_arm),
    }
    for label, identity in identities.items():
        for record in all_records:
            if (record["root"], record["arm"]) != identity:
                continue
            surface = (
                str(record["modality"]),
                str(record["metric_source"]),
                bool(record["is_primary_surface"]),
                str(record["metric_name"]),
            )
            target_values = surfaces.setdefault(surface, {}).setdefault(label, {})
            target_key = str(record["target_key"])
            if target_key in target_values:
                raise ValueError(
                    "Duplicate target metric for "
                    f"{identity}/{surface}/{target_key}; scorer rows were co-mingled."
                )
            target_values[target_key] = record

    results: list[dict[str, Any]] = []
    for surface, surface_records in sorted(surfaces.items()):
        baseline = surface_records.get("baseline", {})
        candidate = surface_records.get("candidate", {})
        if not baseline or not candidate:
            continue
        shared = sorted(set(baseline) & set(candidate))
        modality, metric_source, is_primary_surface, metric_name = surface
        for outcome, field in (("oracle", "oracle_success"), ("rank", "rank_success")):
            candidate_only = sum(
                not bool(baseline[key][field]) and bool(candidate[key][field])
                for key in shared
            )
            baseline_only = sum(
                bool(baseline[key][field]) and not bool(candidate[key][field])
                for key in shared
            )
            results.append(
                {
                    "baseline_root": baseline_root,
                    "baseline_arm": baseline_arm,
                    "candidate_root": candidate_root,
                    "candidate_arm": candidate_arm,
                    "modality": modality,
                    "metric_source": metric_source,
                    "is_primary_surface": is_primary_surface,
                    "metric_name": metric_name,
                    "outcome": outcome,
                    "paired_target_count": len(shared),
                    "baseline_success_count": sum(
                        bool(baseline[key][field]) for key in shared
                    ),
                    "candidate_success_count": sum(
                        bool(candidate[key][field]) for key in shared
                    ),
                    "candidate_only_success_count": candidate_only,
                    "baseline_only_success_count": baseline_only,
                    "two_sided_exact_mcnemar_p": exact_two_sided_mcnemar_p(
                        candidate_only,
                        baseline_only,
                    ),
                    "baseline_unpaired_target_count": len(
                        set(baseline) - set(candidate)
                    ),
                    "candidate_unpaired_target_count": len(
                        set(candidate) - set(baseline)
                    ),
                }
            )
    return results


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a stable CSV with all declared columns, including empty metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--targets-out", required=True, type=Path)
    parser.add_argument(
        "--pair",
        nargs=4,
        metavar=("BASELINE_ROOT", "BASELINE_ARM", "CANDIDATE_ROOT", "CANDIDATE_ARM"),
        help="Emit oracle and rank McNemar rows for this exact pair.",
    )
    parser.add_argument(
        "--paired-out",
        type=Path,
        help="Destination for --pair output; required when --pair is supplied.",
    )
    args = parser.parse_args()
    if args.pair is not None and args.paired_out is None:
        parser.error("--paired-out is required with --pair")
    return args


def main() -> int:
    args = parse_args()
    records = collect_target_records(args.sweep_root)
    if not records:
        raise ValueError(f"{args.sweep_root}: no scorer CSVs found under run manifests")

    write_csv(args.targets_out, TARGET_FIELDS, records)
    write_csv(args.summary_out, SUMMARY_FIELDS, summary_rows(records))
    if args.pair is not None:
        baseline_root, baseline_arm, candidate_root, candidate_arm = args.pair
        paired = paired_rows(
            records,
            baseline_root=baseline_root,
            baseline_arm=baseline_arm,
            candidate_root=candidate_root,
            candidate_arm=candidate_arm,
        )
        write_csv(args.paired_out, PAIRED_FIELDS, paired)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
