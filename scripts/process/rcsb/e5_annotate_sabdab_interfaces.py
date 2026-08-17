"""Annotate RCSB validation protein-protein interfaces with descriptors."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass

import msgpack
import zstandard

import kfold.constants as C

ANTIBODY_ANTIGEN_DESCRIPTOR = "antibody_antigen"
HOMO_DESCRIPTOR = "homo"
HETERO_DESCRIPTOR = "hetero"
MANAGED_DESCRIPTORS = {
    ANTIBODY_ANTIGEN_DESCRIPTOR,
    HOMO_DESCRIPTOR,
    HETERO_DESCRIPTOR,
}


@dataclass(frozen=True, slots=True)
class SAbDabRow:
    antibody_chain_ids: frozenset[str]
    antigen_chain_ids: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add homo/hetero and antibody-antigen descriptors to an RCSB manifest."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="RCSB dataset directory containing manifest.msgpack.",
    )
    parser.add_argument(
        "--sabdab_path",
        type=pathlib.Path,
        required=True,
        help="SAbDab summary CSV with PDB/Hchain/Lchain/antigen_chain columns.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print descriptor statistics without writing manifests.",
    )
    return parser.parse_args()


def normalize_pdb_id(pdb_id: str) -> str:
    """Normalize SAbDab PDB IDs to K-Fold manifest IDs."""
    pdb_id = pdb_id.strip().lower()
    if pdb_id.startswith("pdb_"):
        pdb_id = pdb_id[len("pdb_") :]
    if len(pdb_id) == 8 and pdb_id.startswith("0000"):
        pdb_id = pdb_id[4:]
    return pdb_id


def split_chain_ids(chain_ids: str | None) -> frozenset[str]:
    """Split a pipe-delimited SAbDab chain field, ignoring missing values."""
    if chain_ids is None:
        return frozenset()
    chain_ids = chain_ids.strip()
    if not chain_ids or chain_ids.upper() == "NA":
        return frozenset()
    return frozenset(value.strip() for value in chain_ids.split("|") if value.strip())


def load_sabdab_rows(path: pathlib.Path) -> dict[str, list[SAbDabRow]]:
    """Load row-level antibody and antigen chain identifiers by PDB ID."""
    rows_by_pdb_id: dict[str, list[SAbDabRow]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"PDB", "Hchain", "Lchain", "antigen_chain"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required SAbDab columns: {sorted(missing)}")

        for row in reader:
            antibody_ids = split_chain_ids(row["Hchain"]) | split_chain_ids(row["Lchain"])
            antigen_ids = split_chain_ids(row["antigen_chain"])
            if not antibody_ids or not antigen_ids:
                continue
            pdb_id = normalize_pdb_id(row["PDB"])
            rows_by_pdb_id[pdb_id].append(
                SAbDabRow(
                    antibody_chain_ids=antibody_ids,
                    antigen_chain_ids=antigen_ids,
                )
            )
    return dict(rows_by_pdb_id)


class ChainResolver:
    """Resolve SAbDab chain IDs to protein asym IDs in one manifest entry."""

    def __init__(self, metadata: dict) -> None:
        self.chains_by_asym_id = {chain["asym_id"]: chain for chain in metadata["chains"]}
        self.auth_lookup: dict[str, set[int]] = defaultdict(set)
        self.name_lookup: dict[str, set[int]] = defaultdict(set)
        for chain in metadata["chains"]:
            if chain["type"] != C.ChainType.PROTEIN.value:
                continue
            if auth_asym_id := chain.get("auth_asym_id"):
                self.auth_lookup[auth_asym_id].add(chain["asym_id"])
            if name := chain.get("name"):
                self.name_lookup[name].add(chain["asym_id"])

    def resolve(self, chain_id: str) -> tuple[set[int], str | None]:
        """Resolve one chain ID, preferring auth_asym_id over name."""
        asym_ids = self.auth_lookup.get(chain_id)
        if not asym_ids:
            asym_ids = self.name_lookup.get(chain_id)
        if not asym_ids:
            return set(), "unresolved"

        entity_ids = {
            self.chains_by_asym_id[asym_id]["entity_id"] for asym_id in asym_ids
        }
        if len(entity_ids) > 1:
            return set(), "ambiguous"
        return set(asym_ids), None


def annotate_metadata(
    metadata: dict,
    sabdab_rows: list[SAbDabRow],
    stats: Counter[str],
) -> None:
    """Replace managed protein-protein descriptors on one metadata entry."""
    chains_by_asym_id = {chain["asym_id"]: chain for chain in metadata["chains"]}
    for interface in metadata["interfaces"]:
        descriptors = interface.get("descriptors", [])
        retained_descriptors = [
            descriptor
            for descriptor in descriptors
            if descriptor not in MANAGED_DESCRIPTORS
        ]
        stats["existing_descriptors_removed"] += len(descriptors) - len(
            retained_descriptors
        )
        if retained_descriptors:
            interface["descriptors"] = retained_descriptors
        else:
            interface.pop("descriptors", None)

        asym_id1, asym_id2 = interface["asym_ids"]
        chain1 = chains_by_asym_id[asym_id1]
        chain2 = chains_by_asym_id[asym_id2]
        if not (
            chain1["type"] == C.ChainType.PROTEIN.value
            and chain2["type"] == C.ChainType.PROTEIN.value
        ):
            continue
        if chain1["entity_id"] == chain2["entity_id"]:
            interface.setdefault("descriptors", []).append(HOMO_DESCRIPTOR)
            stats["protein_protein_homo_interfaces"] += 1
            if interface.get("is_low_homology", False):
                stats["low_homology_protein_protein_homo_interfaces"] += 1
        else:
            interface.setdefault("descriptors", []).append(HETERO_DESCRIPTOR)
            stats["protein_protein_hetero_interfaces"] += 1
            if interface.get("is_low_homology", False):
                stats["low_homology_protein_protein_hetero_interfaces"] += 1

    if not sabdab_rows:
        return

    stats["sabdab_entries_in_manifest"] += 1
    stats["sabdab_rows_in_manifest"] += len(sabdab_rows)
    resolver = ChainResolver(metadata)
    antibody_antigen_pairs: set[tuple[int, int]] = set()

    for row in sabdab_rows:
        antibody_asym_ids: set[int] = set()
        antigen_asym_ids: set[int] = set()
        for chain_id in row.antibody_chain_ids:
            resolved, error = resolver.resolve(chain_id)
            antibody_asym_ids.update(resolved)
            if error:
                stats[f"{error}_antibody_chain_ids"] += 1
        for chain_id in row.antigen_chain_ids:
            resolved, error = resolver.resolve(chain_id)
            antigen_asym_ids.update(resolved)
            if error:
                stats[f"{error}_antigen_chain_ids"] += 1

        antibody_antigen_pairs.update(
            tuple(sorted((antibody_asym_id, antigen_asym_id)))
            for antibody_asym_id in antibody_asym_ids
            for antigen_asym_id in antigen_asym_ids
            if antibody_asym_id != antigen_asym_id
        )

    entry_annotated = False
    low_homology_entry_annotated = False
    for interface in metadata["interfaces"]:
        asym_id1, asym_id2 = interface["asym_ids"]
        chain1 = chains_by_asym_id[asym_id1]
        chain2 = chains_by_asym_id[asym_id2]
        if not (
            chain1["type"] == C.ChainType.PROTEIN.value
            and chain2["type"] == C.ChainType.PROTEIN.value
        ):
            continue
        if tuple(sorted(interface["asym_ids"])) not in antibody_antigen_pairs:
            continue

        interface.setdefault("descriptors", []).append(ANTIBODY_ANTIGEN_DESCRIPTOR)
        stats["antibody_antigen_interfaces"] += 1
        entry_annotated = True
        if interface.get("is_low_homology", False):
            stats["low_homology_antibody_antigen_interfaces"] += 1
            low_homology_entry_annotated = True

    if entry_annotated:
        stats["antibody_antigen_entries"] += 1
    if low_homology_entry_annotated:
        stats["low_homology_antibody_antigen_entries"] += 1


def annotate_manifest(
    metadata_dicts: list[dict],
    rows_by_pdb_id: dict[str, list[SAbDabRow]],
) -> tuple[list[dict], Counter[str]]:
    """Annotate every entry and return serializable metadata plus statistics."""
    annotated_dicts = copy.deepcopy(metadata_dicts)
    stats: Counter[str] = Counter(manifest_entries=len(annotated_dicts))
    for metadata in annotated_dicts:
        annotate_metadata(metadata, rows_by_pdb_id.get(metadata["id"], []), stats)
    return annotated_dicts, stats


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Atomically replace a file with bytes written in the same directory."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    try:
        with temporary_path.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if existing_mode is not None:
            temporary_path.chmod(existing_mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_manifests(data_dir: pathlib.Path, metadata_dicts: list[dict]) -> None:
    """Write matching msgpack and zstd-compressed JSON manifests."""
    msgpack_data = msgpack.packb(metadata_dicts, use_bin_type=True)
    json_data = json.dumps(metadata_dicts, indent=2).encode("utf-8")
    json_zst_data = zstandard.ZstdCompressor(level=3).compress(json_data)

    assert msgpack.unpackb(msgpack_data, raw=False) == metadata_dicts
    assert json.loads(zstandard.ZstdDecompressor().decompress(json_zst_data)) == (
        metadata_dicts
    )

    atomic_write(data_dir / "manifest.json.zst", json_zst_data)
    atomic_write(data_dir / "manifest.msgpack", msgpack_data)


def main() -> None:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.msgpack"
    with manifest_path.open("rb") as f:
        metadata_dicts: list[dict] = msgpack.unpack(f, raw=False)

    rows_by_pdb_id = load_sabdab_rows(args.sabdab_path)
    annotated_dicts, stats = annotate_manifest(metadata_dicts, rows_by_pdb_id)
    print(json.dumps(dict(sorted(stats.items())), indent=2))

    if not args.dry_run:
        write_manifests(args.data_dir, annotated_dicts)
        print(f"Updated {args.data_dir / 'manifest.msgpack'}")
        print(f"Updated {args.data_dir / 'manifest.json.zst'}")


if __name__ == "__main__":
    main()
