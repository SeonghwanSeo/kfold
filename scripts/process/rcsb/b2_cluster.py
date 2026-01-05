"""Cluster training PDB set.

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
5. Save updated metadata with cluster IDs as JSON files.
"""

import argparse
import functools
import gc
import multiprocessing
import os
import pathlib
import subprocess

import pandas as pd
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.misc import hash_seq


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB CCD data.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Working directory containing preprocessed npz/ folder.",
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


def get_category(ctype: C.ChainType) -> str:
    """Generate a unique key for an entity."""
    if ctype.is_polymer:
        return ctype.name.lower()
    else:
        return "mol"


def get_entity_key(name: str, entity_id: int) -> str:
    """Generate a unique key for an entity."""
    return f"{name}_{entity_id}"


def get_sequence(chain: Chain) -> str:
    """Retrieve the sequence for a given chain."""
    if chain.ctype.is_polymer:
        sequence: str = chain.get_sequence().upper()
        if chain.ctype.is_protein:
            # Map ambiguous/non-standard amino acids to standard ones
            sequence = (
                sequence.replace("B", "D")
                .replace("Z", "E")
                .replace("U", "C")
                .replace("J", "X")
                .replace("O", "X")
            )
        return sequence
    else:
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        return ":".join(ccd_sequence)


def parse_cif(npz_path: pathlib.Path) -> dict[tuple[str, str], str]:
    """Parse a CIF file and return a gemmi.cif.Document object.
    NOTE: This is just a template function. Additional filtering can be
    inserted as needed, e.g., date cutoff, number of chains, etc.
    """
    # Load reference structure
    try:
        ref_struct: RefStructure = RefStructure.load_npz(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return {}
    name = ref_struct.metadata.id
    sequences: dict[tuple[str, str], str] = {}
    for c in ref_struct.chains:
        key = (get_category(c.ctype), get_entity_key(name, c.entity_id))
        if key in sequences:
            continue  # Skip duplicate chains
        sequences[key] = get_sequence(c)
    return sequences


def save_metadata(
    npz_path: pathlib.Path, mapping: dict[str, dict[str, str]], out_dir: pathlib.Path
):
    """Retrieve and save metadata for given npz file."""
    # Load reference structure
    try:
        ref_struct: RefStructure = RefStructure.load_npz(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return {}
    metadata = ref_struct.metadata

    entity_id_to_cluster: dict[int, str] = {}

    for c in ref_struct.chains:
        if c.entity_id in entity_id_to_cluster:
            continue  # Skip duplicate chains
        category = get_category(c.ctype)
        seq = get_sequence(c)
        seq_hash = hash_seq(seq)
        cluster_id = mapping[category][seq_hash]
        entity_id_to_cluster[c.entity_id] = cluster_id

    del ref_struct  # free memory

    asym_id_to_chain_meta = {c.asym_id: c for c in metadata.chains}

    # Update chain and interface cluster IDs
    for c in metadata.chains:
        c.cluster_id = entity_id_to_cluster[c.entity_id]

    for iface in metadata.interfaces:
        asym_id_1, asym_id_2 = iface.asym_ids
        chain_1 = asym_id_to_chain_meta[asym_id_1]
        chain_2 = asym_id_to_chain_meta[asym_id_2]
        cluster_id_1 = entity_id_to_cluster[chain_1.entity_id]
        cluster_id_2 = entity_id_to_cluster[chain_2.entity_id]
        iface.cluster_id = "|".join(sorted([cluster_id_1, cluster_id_2]))

    name = metadata.id
    out_path = out_dir / name[1:3] / f"{name}_meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.save_json(out_path)


def main():
    """Main function to process and cluster sequences."""
    args = parse_args()

    # Prepare partial function for multiprocessing
    npz_dir: pathlib.Path = args.data_dir / "npz"
    npz_paths = sorted(npz_dir.rglob("*.npz"))
    print(f"Found {len(npz_paths)} preprocessed files to process.")
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_cif, npz_paths),
                total=len(npz_paths),
                desc="Extracting sequences",
            )
        )
    # Collect all chain sequences
    seq_dict: dict[tuple[str, str], str] = {}
    for res in results:
        seq_dict.update(res)
    print("Sequence extraction completed.")
    print(f"Total extracted sequences: {len(seq_dict)}")
    del results  # free memory

    # Extract unique sequences with representative IDs
    proteins: dict[str, str] = {}
    short_proteins: dict[str, str] = {}
    dnas: dict[str, str] = {}
    rnas: dict[str, str] = {}
    mols: dict[str, str] = {}

    for (ctype, key), sequence in sorted(seq_dict.items()):
        # Determine which dictionary to use
        if ctype == "protein" and len(sequence) >= 10:
            category_dict = proteins
        elif ctype == "protein" and len(sequence) < 10:
            category_dict = short_proteins
        elif ctype == "dna":
            category_dict = dnas
        elif ctype == "rna":
            category_dict = rnas
        elif ctype == "mol":
            category_dict = mols
        else:
            raise ValueError(f"Unknown chain type: {ctype}")
        if sequence not in category_dict:
            category_dict[sequence] = key  # Use first occurrence as representative

    # check the number of unique sequences
    print("Unique sequences summary:")
    print(f"  Proteins (>=10 aa): {len(proteins)}")
    print(f"  Short Proteins (<10 aa): {len(short_proteins)}")
    print(f"  DNAs: {len(dnas)}")
    print(f"  RNAs: {len(rnas)}")
    print(f"  Small Molecules: {len(mols)}")

    # check the hashing is safe
    for category, category_dict in [
        ("protein", proteins),
        ("short_protein", short_proteins),
        ("dna", dnas),
        ("rna", rnas),
        ("mol", mols),
    ]:
        hash_set = set()
        for seq in category_dict.keys():
            seq_hash = hash_seq(seq)
            if seq_hash in hash_set:
                raise ValueError(f"Hash collision detected in category {category}")
            hash_set.add(seq_hash)

    # Perform clustering using MMseqs2 (if installed)
    # Save the sequences
    cluster_dir: pathlib.Path = args.data_dir / "clustered"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    with open(cluster_dir / "proteins.fasta", "w") as f:
        for h in sorted(proteins.keys()):
            f.write(f">{proteins[h]}\n{h}\n")
    with open(cluster_dir / "short_proteins.fasta", "w") as f:
        for h in sorted(short_proteins.keys()):
            f.write(f">{short_proteins[h]}\n{h}\n")
    with open(cluster_dir / "dnas.fasta", "w") as f:
        for h in sorted(dnas.keys()):
            f.write(f">{dnas[h]}\n{h}\n")
    with open(cluster_dir / "rnas.fasta", "w") as f:
        for h in sorted(rnas.keys()):
            f.write(f">{rnas[h]}\n{h}\n")
    with open(cluster_dir / "small_molecules.fasta", "w") as f:
        for h in sorted(mols.keys()):
            f.write(f">{mols[h]}\n{h}\n")

    print(f"Saved extracted sequences to {cluster_dir}")
    cmd_str = (
        f"{args.mmseqs} "
        f"easy-cluster "
        f"{cluster_dir / 'proteins.fasta'} "
        f"{cluster_dir / 'mmseq2_out'} "
        f"{cluster_dir / 'tmp/'} "
        "--min-seq-id 0.4 "
        "--dbtype 1"
    )
    print("Running MMseqs2 clustering with command:")
    print(cmd_str)

    subprocess.run(
        cmd_str,
        shell=True,
        check=True,
    )

    # Load mmseq2 clustering output
    cluster_out = pd.read_csv(
        cluster_dir / "mmseq2_out_cluster.tsv",
        sep="\t",
        header=None,
        names=["cluster_id", "seq_id"],
    )
    protein_cluster_map: dict[str, str] = {}
    for _, row in cluster_out.iterrows():
        cluster_id = row["cluster_id"]
        seq_id = row["seq_id"]
        protein_cluster_map[seq_id] = cluster_id

    # Construct Sequence Hash to Cluster ID mapping
    # Protein sequences use 40% homology clustering
    protein_clustering: dict[str, str] = {
        hash_seq(k): protein_cluster_map[v] for k, v in proteins.items()
    }
    # Short proteins, DNA, RNA, and small molecules use 100% homology clustering
    short_protein_clustering: dict[str, str] = {
        hash_seq(k): v for k, v in short_proteins.items()
    }
    dna_clustering: dict[str, str] = {hash_seq(k): v for k, v in dnas.items()}
    rna_clustering: dict[str, str] = {hash_seq(k): v for k, v in rnas.items()}
    mol_clustering: dict[str, str] = {hash_seq(k): v for k, v in mols.items()}
    cluster_mapping: dict[str, dict[str, str]] = {
        "protein": protein_clustering | short_protein_clustering,
        "dna": dna_clustering,
        "rna": rna_clustering,
        "mol": mol_clustering,
    }
    print("Total clusters: ")
    print(f"  Proteins (>=10 aa): {len(set(protein_clustering.values()))}")
    print(f"  Short Proteins (<10 aa): {len(set(short_protein_clustering.values()))}")
    print(f"  DNAs: {len(set(dna_clustering.values()))}")
    print(f"  RNAs: {len(set(rna_clustering.values()))}")
    print(f"  Small Molecules: {len(set(mol_clustering.values()))}")

    # Garbage collection
    del seq_dict
    del proteins, short_proteins, dnas, rnas, mols
    del cluster_out, protein_cluster_map
    gc.collect()

    # Save updated metadata with cluster IDs
    metadata_dir = args.data_dir / "metadata/"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    partial_func = functools.partial(
        save_metadata, mapping=cluster_mapping, out_dir=metadata_dir
    )
    # for npz_path in tqdm(npz_paths, desc="Saving metadata"):
    #     partial_func(npz_path)
    with multiprocessing.Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(partial_func, npz_paths, chunksize=100),
                total=len(npz_paths),
                desc="Saving metadata",
            )
        )
    print("Metadata saving completed.")


if __name__ == "__main__":
    main()
