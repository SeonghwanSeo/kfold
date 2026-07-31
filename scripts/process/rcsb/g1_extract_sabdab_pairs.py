"""Extract SAbDab heavy/light chain pairs from processed RCSB LMDBs.

The SAbDab summary stores author chain IDs in the Hchain/Lchain columns.
This script maps them to RCSB label_asym_id values using the metadata stored in
processed K-Fold structure LMDBs and writes heavy/light sequences for pairs that
exist in the processed data.
"""

import argparse
import csv
import io
import pathlib
from collections import defaultdict
from collections.abc import Iterable

import lmdb
from tqdm import tqdm

from kfold.data.types.metadata import ChainInfo
from kfold.data.types.structure import Chain, RefStructure

DEFAULT_DATA_DIR = pathlib.Path("/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3")
DEFAULT_SABDAB_PATH = pathlib.Path("assets/sabdab_summary_all.csv")
DEFAULT_OUTPUT_PATH = pathlib.Path("scripts/process/rcsb/sabdab_heavy_light_pairs.csv")
OUTPUT_FIELDNAMES = [
    "pdb_id",
    "asym_id_Hchain",
    "asym_id_Lchain",
    "sequence_Hchain",
    "sequence_Lchain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV of SAbDab heavy/light pairs from RCSB LMDB data."
    )
    parser.add_argument(
        "--sabdab_path",
        type=pathlib.Path,
        default=DEFAULT_SABDAB_PATH,
        help="Path to SAbDab summary CSV.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        default=DEFAULT_DATA_DIR,
        help="Processed K-Fold data root containing rcsb-*/structure.lmdb.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="RCSB splits to search under data_dir, in priority order.",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
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


def open_lmdbs(
    data_dir: pathlib.Path,
    splits: Iterable[str],
) -> dict[str, lmdb.Environment]:
    envs = {}
    for split in splits:
        lmdb_path = data_dir / f"rcsb-{split}" / "structure.lmdb"
        if not lmdb_path.exists():
            print(f"Skipping missing LMDB: {lmdb_path}")
            continue
        envs[split] = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
        )
    if not envs:
        raise FileNotFoundError(f"No structure.lmdb found under {data_dir}")
    return envs


def load_structure(
    pdb_id: str,
    envs: dict[str, lmdb.Environment],
) -> tuple[str, RefStructure] | None:
    key = pdb_id.encode("utf-8")
    for split, env in envs.items():
        with env.begin(write=False) as txn:
            data = txn.get(key)
        if data is None:
            continue
        return split, RefStructure.load_npz(io.BytesIO(data))
    return None


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
    for hchain_auth_id, lchain_auth_id in pairs:
        hchain_entry = auth_to_chain.get(hchain_auth_id)
        lchain_entry = auth_to_chain.get(lchain_auth_id)
        if hchain_entry is None or lchain_entry is None:
            skipped_pairs += 1
            continue

        hchain_info, hchain = hchain_entry
        lchain_info, lchain = lchain_entry

        rows.append(
            {
                "pdb_id": pdb_id,
                "asym_id_Hchain": hchain_info.label_asym_id or str(hchain_info.asym_id),
                "asym_id_Lchain": lchain_info.label_asym_id or str(lchain_info.asym_id),
                "sequence_Hchain": hchain.get_sequence(map_to_standard=True),
                "sequence_Lchain": lchain.get_sequence(map_to_standard=True),
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

    envs = open_lmdbs(args.data_dir, args.splits)
    print(f"Searching LMDB splits: {', '.join(envs.keys())}")

    output_rows: list[dict[str, str]] = []
    missing_structures = 0
    skipped_pairs = 0

    for pdb_id, pairs in tqdm(
        sorted(pairs_by_pdb_id.items()),
        desc="Extracting SAbDab pairs",
    ):
        loaded = load_structure(pdb_id, envs)
        if loaded is None:
            missing_structures += len(pairs)
            continue

        _, struct = loaded
        rows, n_skipped = extract_pair_rows(pdb_id, pairs, struct)
        output_rows.extend(rows)
        skipped_pairs += n_skipped

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    for env in envs.values():
        env.close()

    print(f"Wrote {len(output_rows)} rows to {args.out_path}")
    print(f"Pairs with missing LMDB entries ignored: {missing_structures}")
    print(f"Pairs without matching protein chains ignored: {skipped_pairs}")


if __name__ == "__main__":
    main()
