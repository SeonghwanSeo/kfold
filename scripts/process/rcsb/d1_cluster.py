"""Construct RCSB training set with cluster IDs.

1. Extract sequences from npz files. (invalid targets were filtered)
2. Deduplicate sequences based on hash values.
3. Cluster sequences using MMseqs2.

- Protein: 40% homology
- Short protein (<10 residues): 100% homology
- DNA: 100% homology
- RNA: 100% homology
- Small molecules: 100% homology (same CCD ID)

Procedure:
1. Extract sequences from npz files.
2. Deduplicate sequences based on hash values.
3. Cluster sequences using MMseqs2.
4. Construct mapping from (PDB ID, entity ID) to cluster ID.
5. Save updated metadata with cluster IDs.
"""

import argparse
import json
import multiprocessing
import os
import pathlib
from typing import NamedTuple

import msgpack
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.training.preprocess.prepared_sequences import save_sequences
from kfold.utils.mmseqs2 import run_mmseqs2_cluster


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster training set sequences using MMseqs2.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default="mmseqs",
        help="MMseqs2 executable.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    return args


class Seq(NamedTuple):
    pdb_id: str
    entity_id: int
    sequence: str
    ctype: C.ChainType

    @property
    def id(self) -> str:
        return f"{self.pdb_id}_{self.entity_id}"

    @property
    def ctype_str(self) -> str:
        return self.ctype.name.lower()


def parse_npz(
    npz_path: pathlib.Path,
) -> tuple[list[Seq], Metadata]:
    """Parse a NPZ file and return sequences and metadata."""
    struct: RefStructure = RefStructure.load_npz(npz_path)
    metadata: Metadata = struct.metadata
    name = struct.id
    sequences: dict[int, Seq] = {}
    for c in struct.chains:
        if c.entity_id in sequences:
            continue  # Skip duplicate chains
        if c.ctype.is_polymer:
            seq = c.get_sequence(map_to_standard=True)
        else:
            seq = "-".join(c.get_ccd_sequence())
        sequences[c.entity_id] = Seq(
            pdb_id=name,
            entity_id=c.entity_id,
            sequence=seq,
            ctype=c.ctype,
        )
    sequences: list[Seq] = [sequences[eid] for eid in sorted(sequences.keys())]
    return sequences, metadata


def run_clustering(all_sequences: list[Seq], mmseqs: str) -> dict[str, dict[str, str]]:
    """Main function to process and cluster sequences."""

    # Sequence -> representative ID mapping
    protein_to_repr_id: dict[str, str] = {}
    short_protein_to_repr_id: dict[str, str] = {}
    dna_to_repr_id: dict[str, str] = {}
    rna_to_repr_id: dict[str, str] = {}
    ligand_to_repr_id: dict[str, str] = {}

    # Collect sequences
    for seq in all_sequences:
        sequence = seq.sequence
        if seq.ctype.is_polymer:
            # Use first occurrence as representative
            if seq.ctype.is_protein and len(sequence) >= 10:
                category = protein_to_repr_id
            elif seq.ctype.is_protein and len(sequence) < 10:
                category = short_protein_to_repr_id
            elif seq.ctype.is_dna:
                category = dna_to_repr_id
            else:
                category = rna_to_repr_id
            if sequence not in category:
                repr_id = f"{seq.pdb_id}_{seq.entity_id}"
                category[sequence] = repr_id
        else:
            # Use CCD code as representative for ligands
            if sequence not in ligand_to_repr_id:
                ligand_to_repr_id[sequence] = sequence

    # check the number of unique sequences
    print("Unique sequences summary:")
    print(f"  Proteins (>=10 aa): {len(protein_to_repr_id)}")
    print(f"  Short Proteins (<10 aa): {len(short_protein_to_repr_id)}")
    print(f"  DNAs: {len(dna_to_repr_id)}")
    print(f"  RNAs: {len(rna_to_repr_id)}")
    print(f"  Ligands: {len(ligand_to_repr_id)}")

    # Perform clustering using MMseqs2
    # Save the sequences
    print()
    uniq_proteins: list[tuple[str, str]] = sorted(
        [(repr_id, seq) for seq, repr_id in protein_to_repr_id.items()]
    )
    protein_cluster_map = run_mmseqs2_cluster(
        uniq_proteins,
        min_sequence_identity=0.4,
        chain_type="protein",
        verbose=1,
        print_cmd=True,
        mmseqs2_exec=mmseqs,
    )

    # Construct Sequence to Cluster ID mapping
    # Protein sequences use 40% homology clustering
    protein_clusters: dict[str, str] = {
        seq: protein_cluster_map[repr_id] for seq, repr_id in protein_to_repr_id.items()
    }
    # Short proteins, DNA, RNA, and small molecules use 100% homology clustering
    short_protein_clusters: dict[str, str] = short_protein_to_repr_id
    dna_clusters: dict[str, str] = dna_to_repr_id
    rna_clusters: dict[str, str] = rna_to_repr_id
    ligand_clusters: dict[str, str] = ligand_to_repr_id

    print("Total clusters: ")
    print(f"  Proteins (>=10 aa): {len(set(protein_clusters.values()))}")
    print(f"  Short Proteins (<10 aa): {len(set(short_protein_clusters.values()))}")
    print(f"  DNAs: {len(set(dna_clusters.values()))}")
    print(f"  RNAs: {len(set(rna_clusters.values()))}")
    print(f"  Ligands: {len(set(ligand_clusters.values()))}")

    # Return clustering mapping
    cluster_mapping: dict[str, dict[str, str]] = {
        "protein": protein_clusters | short_protein_clusters,
        "dna": dna_clusters,
        "rna": rna_clusters,
        "ligand": ligand_clusters,
    }
    return cluster_mapping


def update_metadata(
    metadata: Metadata,
    sequences: list[Seq],
    mapping: dict[str, dict[str, str]],
) -> None:
    """Update metadata with cluster IDs."""
    entity_id_to_cluster: dict[int, str] = {}
    asym_id_to_entity_id: dict[int, int] = {}

    for seq in sequences:
        cluster_id = mapping[seq.ctype_str][seq.sequence]
        entity_id_to_cluster[seq.entity_id] = cluster_id

    for cm in metadata.chains:
        cm.cluster_id = entity_id_to_cluster[cm.entity_id]
        asym_id_to_entity_id[cm.asym_id] = cm.entity_id

    for iface in metadata.interfaces:
        asym_id_1, asym_id_2 = iface.asym_ids
        entity_id_1 = asym_id_to_entity_id[asym_id_1]
        entity_id_2 = asym_id_to_entity_id[asym_id_2]
        cluster_id_1 = entity_id_to_cluster[entity_id_1]
        cluster_id_2 = entity_id_to_cluster[entity_id_2]
        iface.cluster_id = "|".join(sorted([cluster_id_1, cluster_id_2]))


def main():
    """Main function to construct RCSB training set."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb-train"

    # Step 1. Extract sequences from npz files
    npz_dir = data_dir / "npz"
    npz_files = sorted(npz_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} preprocessed files to process.")
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_npz, npz_files),
                total=len(npz_files),
                desc="Extracting sequences",
            )
        )
    # Step 2. Collect all chain sequences
    metadata_dict: dict[str, Metadata] = {}
    all_sequences: list[Seq] = []
    entry_sequences: dict[str, list[Seq]] = {}
    for seqs, m in results:
        metadata_dict[m.id] = m
        all_sequences.extend(seqs)
        entry_sequences[m.id] = seqs
    all_pdb_ids: list[str] = sorted(metadata_dict.keys())

    print("Sequence extraction completed.")
    print(f"Total structures processed: {len(metadata_dict)}")
    print(f"Total extracted sequences: {len(all_sequences)}")
    del results  # free memory

    # Apo inputs must use entity IDs/sequences after structure preprocessing.
    all_sequences.sort(key=lambda seq: (seq.pdb_id, seq.entity_id))
    save_sequences(
        [
            (f"{seq.pdb_id}|{seq.entity_id}|{seq.ctype_str}", seq.sequence)
            for seq in all_sequences
        ],
        data_dir,
    )

    # Step 3. Cluster sequences using MMseqs2
    print("Starting sequence clustering...")
    cluter_mapping: dict[str, dict[str, str]] = run_clustering(all_sequences, args.mmseqs)

    # Step 4. Update metadata with cluster IDs
    for pdb_id in tqdm(all_pdb_ids, desc="Populating cluster IDs in metadata"):
        update_metadata(
            metadata_dict[pdb_id],
            entry_sequences[pdb_id],
            cluter_mapping,
        )
    # Sort metadatas by PDB ID
    metadatas: list[Metadata] = [metadata_dict[pdb_id] for pdb_id in all_pdb_ids]

    # Step 5. Save updated metadata with cluster IDs
    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]

    # Save to json file (human-readable)
    manifest_path: pathlib.Path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")

    # Save to msgpack file (efficient and fast)
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadata_dicts, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")


if __name__ == "__main__":
    main()
