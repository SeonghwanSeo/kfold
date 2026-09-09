"""Export SAbDab H/L groups as group_id,entry_id,asym_ids for AtlasFold-m."""

import argparse
import csv
import io
import pathlib
from collections import defaultdict

import lmdb
from tqdm import tqdm

from kfold.data.types.metadata import ChainInfo
from kfold.data.types.structure import Chain, RefStructure
from kfold.training.preprocess.apo_preparation import GROUP_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sabdab_path",
        type=pathlib.Path,
        required=True,
        help="SAbDab CSV: PDB,Hchain,Lchain",
    )
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out_path", type=pathlib.Path, required=True)
    return parser.parse_args()


def normalize_pdb_id(pdb_id: str) -> str:
    """Normalize SAbDab/RCSB PDB IDs to the processed LMDB key format."""
    pdb_id = pdb_id.strip().lower()
    if pdb_id.startswith("pdb_"):
        pdb_id = pdb_id[len("pdb_") :]
    if len(pdb_id) == 8 and pdb_id.startswith("0000"):
        pdb_id = pdb_id[4:]
    return pdb_id


def is_missing_chain_id(chain_id: str | None) -> bool:
    if chain_id is None:
        return True
    chain_id = chain_id.strip()
    return chain_id == "" or chain_id.upper() == "NA"


def load_sabdab_pairs(
    sabdab_path: pathlib.Path,
) -> dict[str, list[tuple[str, str]]]:
    """Read SAbDab H/L author-chain pairs grouped by normalized PDB ID."""
    pairs_by_pdb_id: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with sabdab_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"PDB", "Hchain", "Lchain"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing required SAbDab columns: {sorted(missing_fields)}")

        for row in reader:
            hchain = row["Hchain"].strip()
            lchain = row["Lchain"].strip()
            if is_missing_chain_id(hchain) or is_missing_chain_id(lchain):
                continue

            pdb_id = normalize_pdb_id(row["PDB"])
            pairs_by_pdb_id[pdb_id].append((hchain, lchain))

    return dict(pairs_by_pdb_id)


def build_auth_chain_lookup(
    struct: RefStructure,
) -> dict[str, tuple[ChainInfo, Chain]]:
    """Build an author-chain lookup from protein chains only.

    RCSB author chain IDs are sometimes reused for bound/covalent ligands.
    Those ligand chains should not make an antibody H/L protein chain ambiguous.
    Symmetry-expanded protein copies are collapsed when their label_asym_id and
    sequence are identical.
    """
    asym_id_to_chain: dict[int, Chain] = {chain.asym_id: chain for chain in struct.chains}
    candidates: dict[str, list[tuple[ChainInfo, Chain]]] = defaultdict(list)
    for chain_info in struct.metadata.chains:
        chain = asym_id_to_chain.get(chain_info.asym_id)
        if chain is None or not chain.ctype.is_protein:
            continue

        keys = {chain_info.auth_asym_id, chain_info.name}
        for key in keys:
            if key is not None:
                candidates[key].append((chain_info, chain))

    lookup = {}
    for key, values in candidates.items():
        labels = {
            chain_info.label_asym_id or str(chain_info.asym_id)
            for chain_info, _ in values
        }
        sequences = {chain.get_sequence(map_to_standard=True) for _, chain in values}
        if len(labels) == 1 and len(sequences) == 1:
            lookup[key] = values[0]

    return lookup


def extract_pair_rows(
    pdb_id: str,
    pairs: list[tuple[str, str]],
    struct: RefStructure,
) -> tuple[list[dict[str, str]], int]:
    auth_to_chain = build_auth_chain_lookup(struct)

    rows = []
    skipped_pairs = 0
    used_chains = set()
    for hchain_auth_id, lchain_auth_id in sorted(set(pairs)):
        hchain_entry = auth_to_chain.get(hchain_auth_id)
        lchain_entry = auth_to_chain.get(lchain_auth_id)
        if hchain_entry is None or lchain_entry is None:
            skipped_pairs += 1
            continue

        hchain_info, hchain = hchain_entry
        lchain_info, lchain = lchain_entry

        asym_ids = (hchain_info.asym_id, lchain_info.asym_id)
        if hchain.entity_id == lchain.entity_id or used_chains.intersection(asym_ids):
            skipped_pairs += 1
            continue
        used_chains.update(asym_ids)
        rows.append(
            {
                "group_id": f"{pdb_id}_{asym_ids[0]}_{asym_ids[1]}",
                "entry_id": pdb_id,
                "asym_ids": ";".join(map(str, asym_ids)),
            }
        )

    return rows, skipped_pairs


def main() -> None:
    args = parse_args()

    pairs_by_pdb_id = load_sabdab_pairs(args.sabdab_path)
    total_pairs = sum(len(pairs) for pairs in pairs_by_pdb_id.values())
    print(
        f"Loaded {total_pairs} SAbDab heavy/light pairs "
        f"from {len(pairs_by_pdb_id)} PDB entries."
    )

    lmdb_path = args.data_dir / f"rcsb-{args.split}" / "structure.lmdb"
    if not lmdb_path.exists():
        raise FileNotFoundError(lmdb_path)
    output_rows = []
    missing_structures = 0
    skipped_pairs = 0
    with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
        for pdb_id, pairs in tqdm(
            sorted(pairs_by_pdb_id.items()), desc="Extracting SAbDab pairs"
        ):
            with env.begin() as txn:
                data = txn.get(pdb_id.encode())
            if data is None:
                missing_structures += len(pairs)
                continue
            struct = RefStructure.load_npz(io.BytesIO(data))
            rows, n_skipped = extract_pair_rows(pdb_id, pairs, struct)
            output_rows.extend(rows)
            skipped_pairs += n_skipped

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.out_path}")
    print(f"Pairs with missing LMDB entries ignored: {missing_structures}")
    print(f"Pairs without matching protein chains ignored: {skipped_pairs}")


if __name__ == "__main__":
    main()
