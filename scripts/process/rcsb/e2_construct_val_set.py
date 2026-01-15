"""concatenate multiple files into one file."""

import argparse
import json
import pathlib
import pickle

import lmdb

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
    key_path: pathlib.Path = data_dir / "validation_pdb_ids.txt"
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

    # Save to a pickle file (efficient)
    manifest_path: pathlib.Path = data_dir / "manifest.pkl"
    with open(manifest_path, "wb") as f:
        pickle.dump(metadata_dicts, f)
    print(f"Saved manifest (pickle) to {manifest_path}")

    # Save to a json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    n_proteins = 0
    n_rna = 0
    n_dna = 0
    n_ligands = 0

    n_protein_protein = 0
    n_rna_rna = 0
    n_dna_dna = 0
    n_dna_rna = 0

    n_protein_rna = 0
    n_protein_dna = 0
    n_protein_ligand = 0

    n_rna_ligand = 0

    n_dna_ligand = 0

    for metadata in metadatas:
        ctypes = [chain.ctype for chain in metadata.chains]
        if any(ctype.is_protein for ctype in ctypes):
            n_proteins += 1
        if any(ctype.is_rna for ctype in ctypes):
            n_rna += 1
        if any(ctype.is_dna for ctype in ctypes):
            n_dna += 1
        if any(ctype.is_ligand for ctype in ctypes):
            n_ligands += 1

        if sum(1 for ctype in ctypes if ctype.is_protein) >= 2:
            n_protein_protein += 1
        if sum(1 for ctype in ctypes if ctype.is_rna) >= 2:
            n_rna_rna += 1
        if sum(1 for ctype in ctypes if ctype.is_dna) >= 2:
            n_dna_dna += 1
        if (
            sum(1 for ctype in ctypes if ctype.is_dna) >= 1
            and sum(1 for ctype in ctypes if ctype.is_rna) >= 1
        ):
            n_dna_rna += 1

        if any(ctype.is_protein for ctype in ctypes) and any(
            ctype.is_rna for ctype in ctypes
        ):
            n_protein_rna += 1
        if any(ctype.is_protein for ctype in ctypes) and any(
            ctype.is_dna for ctype in ctypes
        ):
            n_protein_dna += 1

        if any(ctype.is_protein for ctype in ctypes) and any(
            ctype.is_ligand for ctype in ctypes
        ):
            n_protein_ligand += 1
        if any(ctype.is_rna for ctype in ctypes) and any(
            ctype.is_ligand for ctype in ctypes
        ):
            n_rna_ligand += 1
        if any(ctype.is_dna for ctype in ctypes) and any(
            ctype.is_ligand for ctype in ctypes
        ):
            n_dna_ligand += 1

    print("Composition statistics:")
    print(f"Number of entries with protein: {n_proteins}")
    print(f"Number of entries with RNA: {n_rna}")
    print(f"Number of entries with DNA: {n_dna}")
    print(f"Number of entries with ligands: {n_ligands}")

    print(f"Number of entries with protein-protein interactions: {n_protein_protein}")
    print(f"Number of entries with RNA-RNA interactions: {n_rna_rna}")
    print(f"Number of entries with DNA-DNA interactions: {n_dna_dna}")
    print(f"Number of entries with DNA-RNA interactions: {n_dna_rna}")

    print(f"Number of entries with protein-RNA interactions: {n_protein_rna}")
    print(f"Number of entries with protein-DNA interactions: {n_protein_dna}")
    print(f"Number of entries with protein-ligand interactions: {n_protein_ligand}")

    print(f"Number of entries with RNA-ligand interactions: {n_rna_ligand}")
    print(f"Number of entries with DNA-ligand interactions: {n_dna_ligand}")


if __name__ == "__main__":
    main()
