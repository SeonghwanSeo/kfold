"""Add group-level apo complex candidates to apo_lookup.msgpack."""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

import msgpack
from tqdm import tqdm

from kfold.data.types.metadata import Metadata

SAMPLE_RE = re.compile(r"^protein_multimer_sample_(?P<sample_id>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach SAbDab-derived antibody complex apo groups to "
            "apo_lookup.msgpack.  The lookup stores one candidate per multimer "
            "sample and does not split H/L into separate metadata records."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to processed dataset root.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="RCSB split to process.",
    )
    parser.add_argument(
        "--sabdab_csv",
        type=pathlib.Path,
        default=None,
        help=(
            "CSV with pdb_id, asym_id_Hchain, asym_id_Lchain columns. "
            "Defaults to rcsb-*/sequences/sabdab_heavy_light_pairs.csv."
        ),
    )
    parser.add_argument(
        "--archive_dir",
        type=pathlib.Path,
        default=None,
        help="Optional apo_complex archive root. Defaults to rcsb-*/apo_complex.",
    )
    parser.add_argument(
        "--lookup_path",
        type=pathlib.Path,
        default=None,
        help="Input/output apo_lookup.msgpack path.",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=None,
        help="Optional output msgpack path. Defaults to overwriting lookup_path.",
    )
    parser.add_argument(
        "--kind",
        default="ab_complex",
        help="Complex group kind written to lookup.",
    )
    parser.add_argument(
        "--archive_kind",
        default="ab",
        help="Subdirectory under apo_complex/ to scan.",
    )
    parser.add_argument(
        "--source",
        default="ab_complex",
        help="Candidate source. Must match apo_complex.lmdb key source.",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Also write a JSON copy next to the msgpack output.",
    )
    parser.add_argument(
        "--keep_existing_kind",
        action="store_true",
        help="Keep existing complex groups of the same kind instead of replacing them.",
    )
    return parser.parse_args()


def strip_structure_suffix(path: pathlib.Path) -> str:
    name = path.name
    for suffix in (".zst", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for suffix in (".pdb", ".cif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def parse_sample_id(path: pathlib.Path) -> str:
    match = SAMPLE_RE.match(strip_structure_suffix(path))
    if match is None:
        raise ValueError(f"Unexpected apo complex file name: {path.name}")
    return match.group("sample_id")


def load_manifest(path: pathlib.Path) -> dict[str, Metadata]:
    with path.open("rb") as f:
        metadata_dicts = msgpack.unpack(f, raw=False)
    return {m.id: m for m in (Metadata.from_dict(d) for d in metadata_dicts)}


def load_lookup(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        return msgpack.unpack(f, raw=False, strict_map_key=False)


def normalize_entry(entry_lookup: dict) -> dict:
    if "chains" in entry_lookup or "complex_groups" in entry_lookup:
        chains = entry_lookup.get("chains", {})
        groups = entry_lookup.get("complex_groups", [])
    else:
        chains = entry_lookup
        groups = []
    return {"chains": {str(k): v for k, v in chains.items()}, "complex_groups": groups}


def read_sabdab_pairs(path: pathlib.Path) -> dict[str, list[tuple[str, str]]]:
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"pdb_id", "asym_id_Hchain", "asym_id_Lchain"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required SAbDab columns: {sorted(missing)}")
        for row in reader:
            pdb_id = row["pdb_id"].strip().lower()
            h_asym = row["asym_id_Hchain"].strip()
            l_asym = row["asym_id_Lchain"].strip()
            if pdb_id and h_asym and l_asym:
                pairs[pdb_id].append((h_asym, l_asym))
    return dict(pairs)


def chain_label_to_asym_id(metadata: Metadata) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for chain in metadata.chains:
        if chain.label_asym_id is not None:
            mapping[chain.label_asym_id] = chain.asym_id
        mapping[str(chain.asym_id)] = chain.asym_id
    return mapping


def find_candidates(
    archive_root: pathlib.Path,
    archive_kind: str,
    source: str,
    pdb_id: str,
    group_id: str,
) -> list[dict]:
    group_dir = archive_root / archive_kind / pdb_id[1:3] / pdb_id / group_id
    if not group_dir.exists():
        return []

    paths: list[pathlib.Path] = []
    patterns = (
        "protein_multimer_sample_*.pdb",
        "protein_multimer_sample_*.pdb.gz",
        "protein_multimer_sample_*.pdb.zst",
        "protein_multimer_sample_*.cif",
        "protein_multimer_sample_*.cif.gz",
        "protein_multimer_sample_*.cif.zst",
    )
    for pattern in patterns:
        paths.extend(group_dir.glob(pattern))

    candidates = []
    for path in sorted(paths):
        sample_id = parse_sample_id(path)
        candidates.append(
            {
                "source": source,
                "name": f"{pdb_id}_{group_id}_sample{sample_id}",
                "chain_type": "protein",
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    sabdab_csv = (
        args.sabdab_csv or data_dir / "sequences" / "sabdab_heavy_light_pairs.csv"
    )
    archive_root = args.archive_dir or data_dir / "apo_complex"
    lookup_path = args.lookup_path or data_dir / "apo_lookup.msgpack"
    out_path = args.out_path or lookup_path

    metadata_by_id = load_manifest(data_dir / "manifest.msgpack")
    lookup = load_lookup(lookup_path)
    pairs_by_pdb_id = read_sabdab_pairs(sabdab_csv)

    stats = {
        "pairs": 0,
        "missing_entry": 0,
        "missing_chain": 0,
        "missing_archive": 0,
        "groups_added": 0,
        "candidates_added": 0,
    }

    for pdb_id, pairs in tqdm(sorted(pairs_by_pdb_id.items()), desc="Adding groups"):
        metadata = metadata_by_id.get(pdb_id)
        if metadata is None:
            stats["missing_entry"] += len(pairs)
            continue

        entry = normalize_entry(lookup.get(pdb_id, {}))
        if not args.keep_existing_kind:
            entry["complex_groups"] = [
                group
                for group in entry["complex_groups"]
                if group.get("kind") != args.kind
            ]

        label_to_asym = chain_label_to_asym_id(metadata)
        seen_groups = {
            (group.get("kind"), group.get("group_id"))
            for group in entry["complex_groups"]
        }

        for h_label, l_label in pairs:
            stats["pairs"] += 1
            group_id = f"{h_label}_{l_label}"
            if h_label not in label_to_asym or l_label not in label_to_asym:
                stats["missing_chain"] += 1
                continue

            candidates = find_candidates(
                archive_root,
                args.archive_kind,
                args.source,
                pdb_id,
                group_id,
            )
            if not candidates:
                stats["missing_archive"] += 1
                continue

            group_key = (args.kind, group_id)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)

            h_asym_id = int(label_to_asym[h_label])
            l_asym_id = int(label_to_asym[l_label])
            entry["complex_groups"].append(
                {
                    "kind": args.kind,
                    "chain_type": "protein",
                    "apo_uid": h_asym_id,
                    "asym_ids": [h_asym_id, l_asym_id],
                    "label_asym_ids": [h_label, l_label],
                    "group_id": group_id,
                    "candidates": candidates,
                }
            )
            stats["groups_added"] += 1
            stats["candidates_added"] += len(candidates)

        lookup[pdb_id] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        msgpack.pack(lookup, f)
    if args.write_json:
        json_path = out_path.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(lookup, f, indent=2)
        print(f"Wrote JSON lookup: {json_path}")

    print(f"Wrote lookup: {out_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
