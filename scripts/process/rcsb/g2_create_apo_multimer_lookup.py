"""Create apo_multimer_lookup.msgpack from SAbDab H/L mapping."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import msgpack
from tqdm import tqdm

from kfold.data.types.metadata import Metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build apo_multimer_lookup.msgpack from SAbDab heavy/light pairs. "
            "The output is runtime metadata only; coordinates live in "
            "apo_multimer_lmdb and prior_multimer_lmdb."
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
        help="CSV with pdb_id/asym_id_Hchain/asym_id_Lchain columns.",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=None,
        help="Output msgpack path. Defaults to rcsb-*/apo_multimer_lookup.msgpack.",
    )
    parser.add_argument(
        "--kind",
        default="antibody_hl",
        help="Multimer group kind written to lookup.",
    )
    parser.add_argument(
        "--source",
        default="prot_m_sampler",
        help="Multimer LMDB source name.",
    )
    parser.add_argument(
        "--source_dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional {pdb_id}_{Hlabel}_{Llabel}/ source directory filter. "
            "When provided, groups without a matching directory are skipped."
        ),
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Also write a JSON copy next to the msgpack output.",
    )
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> dict[str, Metadata]:
    with path.open("rb") as f:
        metadata_dicts = msgpack.unpack(f, raw=False)
    return {m.id: m for m in (Metadata.from_dict(d) for d in metadata_dicts)}


def read_sabdab_pairs(path: pathlib.Path) -> dict[str, list[dict[str, str]]]:
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"pdb_id", "asym_id_Hchain", "asym_id_Lchain"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required SAbDab columns: {sorted(missing)}")
        for row in reader:
            pdb_id = row["pdb_id"].strip().lower()
            h_label = row["asym_id_Hchain"].strip()
            l_label = row["asym_id_Lchain"].strip()
            if pdb_id and h_label and l_label:
                pairs[pdb_id].append(
                    {
                        "h_label": h_label,
                        "l_label": l_label,
                    }
                )
    return dict(pairs)


def chain_label_to_asym_id(metadata: Metadata) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for chain in metadata.chains:
        if chain.label_asym_id is not None:
            mapping[chain.label_asym_id] = chain.asym_id
        mapping[str(chain.asym_id)] = chain.asym_id
    return mapping


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    sabdab_csv = (
        args.sabdab_csv or data_dir / "sequences" / "sabdab_heavy_light_pairs.csv"
    )
    out_path = args.out_path or data_dir / "apo_multimer_lookup.msgpack"

    metadata_by_id = load_manifest(data_dir / "manifest.msgpack")
    pairs_by_pdb_id = read_sabdab_pairs(sabdab_csv)

    lookup: dict[str, list[dict]] = {}
    stats = {
        "pairs": 0,
        "missing_entry": 0,
        "missing_chain": 0,
        "missing_source_dir": 0,
        "duplicate_group": 0,
        "groups_added": 0,
    }

    for pdb_id, pairs in tqdm(
        sorted(pairs_by_pdb_id.items()), desc="Building apo_multimer_lookup"
    ):
        metadata = metadata_by_id.get(pdb_id)
        if metadata is None:
            stats["missing_entry"] += len(pairs)
            continue

        label_to_asym = chain_label_to_asym_id(metadata)
        seen_groups: set[tuple[int, int]] = set()
        groups: list[dict] = []

        for pair in pairs:
            stats["pairs"] += 1
            h_label = pair["h_label"]
            l_label = pair["l_label"]
            if h_label not in label_to_asym or l_label not in label_to_asym:
                stats["missing_chain"] += 1
                continue

            h_asym_id = int(label_to_asym[h_label])
            l_asym_id = int(label_to_asym[l_label])
            group_key = (h_asym_id, l_asym_id)
            if group_key in seen_groups:
                stats["duplicate_group"] += 1
                continue
            seen_groups.add(group_key)

            name = f"{pdb_id}_{h_label}_{l_label}"
            if args.source_dir is not None and not (args.source_dir / name).exists():
                stats["missing_source_dir"] += 1
                continue
            groups.append(
                {
                    "kind": args.kind,
                    "chain_type": "protein",
                    "apo_uid": h_asym_id,
                    "asym_ids": [h_asym_id, l_asym_id],
                    "source": args.source,
                    "name": name,
                    "group_id": f"{h_label}_{l_label}",
                }
            )
            stats["groups_added"] += 1

        if groups:
            lookup[pdb_id] = groups

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
