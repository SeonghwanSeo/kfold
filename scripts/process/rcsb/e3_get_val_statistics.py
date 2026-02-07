import argparse
import io
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import TypeVar

import lmdb
import msgpack
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import Chain, RefStructure

_T = TypeVar("_T")
ChainType = C.ChainType
SubChainType = C.SubChainType

InterfaceType = tuple[ChainType, ChainType]
SubInterfaceType = tuple[SubChainType, SubChainType]


# Helper functions
def norm_key(key1: _T, key2: _T) -> tuple[_T, _T]:
    """Return a normalized tuple of two keys."""
    return (key1, key2) if key1 <= key2 else (key2, key1)


def parse_args():
    parser = argparse.ArgumentParser(description="Get validation set statistics.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb-val"

    # Get entry IDs to include
    print("Loading entry IDs...")
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        metadata_dicts = msgpack.unpack(f, strict_map_key=False)
    manifest: list[Metadata] = [Metadata.from_dict(d) for d in metadata_dicts]
    print(f"Total entries in manifest: {len(manifest)}")

    n_chains_per_type: dict[ChainType, int] = defaultdict(int)
    n_eval_chains_per_type: dict[ChainType, int] = defaultdict(int)
    n_ifaces_per_type: dict[InterfaceType, int] = defaultdict(int)
    n_eval_ifaces_per_type: dict[InterfaceType, int] = defaultdict(int)

    n_chains_per_subtype: dict[SubChainType, int] = defaultdict(int)
    n_eval_chains_per_subtype: dict[SubChainType, int] = defaultdict(int)
    n_ifaces_per_subtype: dict[SubInterfaceType, int] = defaultdict(int)
    n_eval_ifaces_per_subtype: dict[SubInterfaceType, int] = defaultdict(int)

    entry_per_chain_type: dict[ChainType, set[str]] = defaultdict(set)
    entry_per_iface_type: dict[InterfaceType, set[str]] = defaultdict(set)
    entry_per_chain_subtype: dict[SubChainType, set[str]] = defaultdict(set)
    entry_per_iface_subtype: dict[SubInterfaceType, set[str]] = defaultdict(set)

    earlest_release_date = datetime.max
    latest_release_date = datetime.min

    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(str(lmdb_path), readonly=True)
    with env.begin(write=False) as txn:
        for m in tqdm(manifest):
            entry_id = m.id
            data = txn.get(entry_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Entry ID {entry_id} not found in LMDB database.")
            with io.BytesIO(data) as f:
                struct = RefStructure.load_npz(f)

            # Update date
            release_date = datetime.fromisoformat(m.exp.release_date)
            earlest_release_date = min(earlest_release_date, release_date)
            latest_release_date = max(latest_release_date, release_date)

            # Collect type info
            asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in struct.chains}
            for cm in m.chains:
                c = asym_id_to_chain[cm.asym_id]

                n_chains_per_type[c.ctype] += 1
                n_chains_per_subtype[c.subtype] += 1
                if cm.is_low_homology:
                    n_eval_chains_per_type[c.ctype] += 1
                    n_eval_chains_per_subtype[c.subtype] += 1
                    entry_per_chain_type[c.ctype].add(entry_id)
                    entry_per_chain_subtype[c.subtype].add(entry_id)

            for iface in m.interfaces:
                asym_id_1, asym_id_2 = iface.asym_ids

                c1 = asym_id_to_chain[asym_id_1]
                c2 = asym_id_to_chain[asym_id_2]
                ctype = norm_key(c1.ctype, c2.ctype)
                subtype = norm_key(c1.subtype, c2.subtype)

                n_ifaces_per_type[ctype] += 1
                n_ifaces_per_subtype[subtype] += 1
                if iface.is_low_homology:
                    n_eval_ifaces_per_type[ctype] += 1
                    n_eval_ifaces_per_subtype[subtype] += 1
                    entry_per_iface_type[ctype].add(entry_id)
                    entry_per_iface_subtype[subtype].add(entry_id)

    env.close()

    print("Release date range:")
    print(f"  Earliest: {earlest_release_date.date().isoformat()}")
    print(f"  Latest: {latest_release_date.date().isoformat()}")
    print()
    print("Final chain type statistics:")
    for ctype in sorted(n_chains_per_type.keys()):
        v1 = n_eval_chains_per_type[ctype]
        v2 = n_chains_per_type[ctype]
        print(f"  {ctype}: {v1} / {v2}")
    print()
    print("Final interface type statistics:")
    for ctypes in sorted(n_ifaces_per_type.keys()):
        key = f"{ctypes[0]}-{ctypes[1]}"
        v1 = n_eval_ifaces_per_type[ctypes]
        v2 = n_ifaces_per_type[ctypes]
        print(f"  {key}: {v1} / {v2}")
    print()

    print("Final chain subtype statistics:")
    for subtype in sorted(n_chains_per_subtype.keys()):
        v1 = n_eval_chains_per_subtype[subtype]
        v2 = n_chains_per_subtype[subtype]
        print(f"  {subtype}: {v1} / {v2}")
    print()
    print("Final interface subtype statistics:")
    for subtypes in sorted(n_ifaces_per_subtype.keys()):
        key = f"{subtypes[0]}-{subtypes[1]}"
        v1 = n_eval_ifaces_per_subtype[subtypes]
        v2 = n_ifaces_per_subtype[subtypes]
        print(f"  {key}: {v1} / {v2}")
    print()

    print("Entries per chain type:")
    for ctype in sorted(entry_per_chain_type.keys()):
        n_entries = len(entry_per_chain_type[ctype])
        print(f"  {ctype}: {n_entries}")
    print()
    print("Entries per interface type:")
    for ctypes in sorted(entry_per_iface_type.keys()):
        key = f"{ctypes[0]}-{ctypes[1]}"
        n_entries = len(entry_per_iface_type[ctypes])
        print(f"  {key}: {n_entries}")
    print()
    print("Entries per chain subtype:")
    for subtype in sorted(entry_per_chain_subtype.keys()):
        n_entries = len(entry_per_chain_subtype[subtype])
        print(f"  {subtype}: {n_entries}")
    print()
    print("Entries per interface subtype:")
    for subtypes in sorted(entry_per_iface_subtype.keys()):
        key = f"{subtypes[0]}-{subtypes[1]}"
        n_entries = len(entry_per_iface_subtype[subtypes])
        print(f"  {key}: {n_entries}")
    print()


if __name__ == "__main__":
    main()
