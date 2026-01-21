"""concatenate multiple files into one file."""

import argparse
import json
import pathlib
from collections import defaultdict

import lmdb
import msgpack

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure


def parse_args():
    parser = argparse.ArgumentParser(description="Construct validation set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    args = parser.parse_args()

    return args


def main():
    """Main function to extract sequences from npz files."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    # Get entry IDs to include
    print("Loading entry IDs...")
    key_path: pathlib.Path = data_dir / "validation_ids.txt"
    print(key_path.absolute())
    with open(key_path) as f:
        entry_ids: list[str] = sorted(set(line.strip().lower() for line in f.readlines()))
    print(f"Total entry IDs to include: {len(entry_ids)}")

    # Get npz files
    npz_dir: pathlib.Path = data_dir / "npz"
    npz_path_dict: dict[str, pathlib.Path] = {p.stem: p for p in npz_dir.rglob("*.npz")}

    # Create lmdb environment (expected size of rcsb training set: ~20GB)
    print("Creating LMDB database...")
    metadatas: list[Metadata] = []
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=1024 * 1024 * 1024,  # 1 GB
    )
    with env.begin(write=True) as txn:
        for entry_id in entry_ids:
            key = entry_id.encode()
            npz_path = npz_path_dict.get(entry_id)
            if npz_path is None:
                print(f"Warning: NPZ file not found for {entry_id}, skipping.")
                continue
            # Read the npz file as bytes
            with open(npz_path, "rb") as f:
                value_bytes = f.read()
            # Put (key, value) pair into the transaction
            txn.put(key, value_bytes)
            # Load structure to get metadata
            # WARN: this does not include the cluster ID info.
            struct = RefStructure.load_npz(npz_path)
            metadatas.append(struct.metadata)
    env.close()

    print(f"Successfully created LMDB at {lmdb_path}")
    print(f"Total entries written: {len(metadatas)}")

    # Save metadatas to a single manifest file.
    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]

    # Save to a msgpack file (efficient and fast)
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadata_dicts, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")

    # Save to a json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    chains_per_ctype: dict[C.ChainType, int] = defaultdict(int)
    interfaces_per_ctype: dict[tuple[C.ChainType, C.ChainType], int] = defaultdict(int)

    def norm_ctype(c1: C.ChainType, c2: C.ChainType) -> tuple[C.ChainType, C.ChainType]:
        return (c1, c2) if c1.value <= c2.value else (c2, c1)

    for m in metadatas:
        # Check chain composition
        for cm in m.chains:
            chains_per_ctype[cm.ctype] += 1

        for im in m.interfaces:
            ctype1 = m.get_chain_by_asym_id(im.asym_ids[0]).ctype
            ctype2 = m.get_chain_by_asym_id(im.asym_ids[1]).ctype
            ctype_pair = norm_ctype(ctype1, ctype2)
            interfaces_per_ctype[ctype_pair] += 1

    print("Composition statistics:")
    print("Final chain type statistics:")
    for ctype in sorted(chains_per_ctype.keys()):
        print(f"  {ctype}: {chains_per_ctype[ctype]}")
    print()

    print("Final interface type statistics:")
    for ctypes in sorted(interfaces_per_ctype.keys()):
        key = f"{ctypes[0]}-{ctypes[1]}"
        print(f"  {key}: {interfaces_per_ctype[ctypes]}")


if __name__ == "__main__":
    main()
