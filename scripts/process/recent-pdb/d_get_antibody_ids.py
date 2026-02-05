"""Construct antibody-antigen test set from recent PDB entries."""

import argparse
import hashlib
import pathlib
from collections import defaultdict
from typing import TypeVar

import msgpack
import numpy as np
import pandas as pd

from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure

# Constant
MAX_TOKENS = 1280


_T = TypeVar("_T")


# Helper functions
def norm_key(key1: _T, key2: _T) -> tuple[_T, _T]:
    """Return a normalized tuple of two keys."""
    return (key1, key2) if key1 <= key2 else (key2, key1)


def get_rng(key: str) -> np.random.Generator:
    """Get a random number generator seeded by the given key."""
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct validation set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to mmCIF files directory.",
    )
    parser.add_argument(
        "--sabdab_path",
        type=pathlib.Path,
        required=True,
        help="Path to the SAbDab TSV file.",
    )
    args = parser.parse_args()

    return args


def sample_entries(
    entries: list[Metadata],
    antibody_dict: dict[str, set[str]],
    antigen_dict: dict[str, set[str]],
) -> list[Metadata]:
    """Sample entries by interface clustering.

    Parameters
    ----------
    entries : list[Metadata]
        List of Metadata entries to sample from.
    antibody_dict : dict[str, set[str]]
        Dictionary mapping PDB IDs to sets of antibody chain IDs.
    antigen_dict : dict[str, set[str]]
        Dictionary mapping PDB IDs to sets of antigen chain IDs.

    Returns
    -------
    sampled_entries : list[Metadata]
        List of sampled Metadata entries.
    """
    # Get low-homology interfaces involving at least one representative cluster
    interface_clusters: dict[str, set[str]] = defaultdict(set)
    for m in entries:
        for im in m.interfaces:
            assert im.cluster_id is not None
            if not im.is_low_homology:
                # Not a low-homology interface
                continue

            antigen_ids = antigen_dict[m.id]
            antibody_ids = antibody_dict[m.id]

            asym_id1, asym_id2 = im.asym_ids
            cm1 = m.get_chain_by_asym_id(asym_id1)
            cm2 = m.get_chain_by_asym_id(asym_id2)
            if not (cm1.ctype.is_protein and cm2.ctype.is_protein):
                # Not a protein-protein interface
                continue

            auth_id1, auth_id2 = cm1.auth_asym_id, cm2.auth_asym_id
            if not (
                (auth_id1 in antibody_ids and auth_id2 in antigen_ids)
                or (auth_id2 in antibody_ids and auth_id1 in antigen_ids)
            ):
                # Skip non-antibody-antigen interfaces
                continue

            eid1, eid2 = norm_key(cm1.entity_id, cm2.entity_id)
            im_key = f"{m.id}:{eid1}-{eid2}"
            interface_clusters[im.cluster_id].add(im_key)

    # Sample one interface from each cluster
    sampled_interfaces: list[str] = []
    for cluster_id, ifaces in interface_clusters.items():
        rng = get_rng(cluster_id)
        sampled = sorted(ifaces)[rng.integers(len(ifaces))]
        sampled_interfaces.append(sampled)

    sampled_pdb_ids = set(im.split(":")[0] for im in sampled_interfaces)
    sampled_entries = [m for m in entries if m.id in sampled_pdb_ids]

    # Remove duplicate entries with same set of interface clusters
    entry_clusters: dict[str, frozenset[str]] = dict()
    for m in sampled_entries:
        antigen_ids = antigen_dict[m.id]
        antibody_ids = antibody_dict[m.id]
        all_interface_clusters: set[str] = set()
        for im in m.interfaces:
            assert im.cluster_id is not None
            asym_id1, asym_id2 = im.asym_ids
            cm1 = m.get_chain_by_asym_id(asym_id1)
            cm2 = m.get_chain_by_asym_id(asym_id2)
            auth_id1, auth_id2 = cm1.auth_asym_id, cm2.auth_asym_id
            if not (cm1.ctype.is_protein and cm2.ctype.is_protein):
                # Skip non-protein interfaces
                continue
            auth_id1, auth_id2 = cm1.auth_asym_id, cm2.auth_asym_id
            if not (
                (auth_id1 in antibody_ids and auth_id2 in antigen_ids)
                or (auth_id2 in antibody_ids and auth_id1 in antigen_ids)
            ):
                # Skip non-antibody-antigen interfaces
                continue
            all_interface_clusters.add(im.cluster_id)
        entry_clusters[m.id] = frozenset(all_interface_clusters)

    prune_entries: list[Metadata] = []
    visited_clusters: set[frozenset[str]] = set()
    for m in sampled_entries:
        clusters = entry_clusters[m.id]
        if clusters not in visited_clusters:
            visited_clusters.add(clusters)
            prune_entries.append(m)
    sampled_entries = prune_entries

    return sampled_entries


def main():
    """Main function to construct antibody-antigen test set from recent PDB entries."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    # ============================================================
    # Stage 1: Load Antibody-Antigen Entries from SAbDab
    # ============================================================
    print("Stage 1: Load Antibody-Antigen Entries from SAbDab")

    sabdab_path: pathlib.Path = args.sabdab_path
    df = pd.read_csv(sabdab_path, sep="\t")
    print(f"Total entries in SAbDab: {len(df)}")

    # Collect PDB IDs
    all_pdb_ids = set(k for k in df["pdb"].unique().tolist())

    # Filter for entries with heavy chain and antigen type
    df = df.dropna(subset=["Hchain", "antigen_type"])
    print(f"Entries with Hchain and antigen type: {len(df)}")

    # Filter for protein antigens only
    def is_pure_protein_peptide(type_str: str) -> bool:
        current_types = set(t.strip() for t in type_str.split("|"))
        return current_types.issubset({"protein", "peptide"})

    df = df[df["antigen_type"].apply(is_pure_protein_peptide)]
    assert len(df) > 0, "No antibody-antigen entries found in SAbDab."
    print(f"Entries with protein antigens: {len(df)}")
    print("[Debug] Antigen types found:", df["antigen_type"].unique())
    print()

    # antibody-antigen complexes
    igg_df = df.dropna(subset=["Lchain"])
    igg_pdb_ids: set[str] = set(k for k in igg_df["pdb"].unique().tolist())

    # nanobody-antigen complexes
    vhh_df = df[df["Lchain"].isna() & df["Hchain"].notna()]
    vhh_pdb_ids: set[str] = set(k for k in vhh_df["pdb"].unique().tolist())

    # For simplicity, drop overlaps
    igg_pdb_ids, vhh_pdb_ids = igg_pdb_ids - vhh_pdb_ids, vhh_pdb_ids - igg_pdb_ids
    igg_vhh_pdb_ids = igg_pdb_ids | vhh_pdb_ids

    print("Total unique antibody-antigen PDB IDs:", len(all_pdb_ids))
    print(f"Unique PDB IDs with antibody-antigen complexes: {len(igg_pdb_ids)}")
    print(f"Unique PDB IDs with nanobody-antigen complexes: {len(vhh_pdb_ids)}")
    print()

    # Collect all antibody/antigen chain IDs
    antibody_dict: dict[str, set[str]] = defaultdict(set)
    antigen_dict: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        pdb_id = row.pdb
        if pd.notna(hchain_id := row["Hchain"]):
            for v in hchain_id.split("|"):
                antibody_dict[pdb_id].add(v.strip())
        if pd.notna(lchain_id := row["Lchain"]):
            for v in lchain_id.split("|"):
                antibody_dict[pdb_id].add(v.strip())
        if pd.notna(achain_id := row["antigen_chain"]):
            for v in achain_id.split("|"):
                antigen_dict[pdb_id].add(v.strip())

    # ============================================================
    # Stage 2: Load Recent PDB Entries with simple filtering
    # ============================================================
    print("Stage 2: Load Recent PDB Entries")

    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"

    with open(manifest_path, "rb") as f:
        manifest_dicts: list[dict] = msgpack.unpack(f, raw=False)
    all_metadatas: list[Metadata] = [Metadata.from_dict(d) for d in manifest_dicts]
    all_metadatas.sort(key=lambda x: x.id)
    print(f"Total entries in recent PDB: {len(all_metadatas)}")

    # Filter with antibody-antigen PDB IDs
    metadatas: list[Metadata] = [m for m in all_metadatas if m.id in igg_vhh_pdb_ids]
    print(f"Entries with antibody-antigen complexes: {len(metadatas)}")

    # Filter with DNA/RNA chains
    metadatas: list[Metadata] = [
        m for m in metadatas if not any(c.ctype.is_nucleic_acid for c in m.chains)
    ]
    print(f"Entries after DNA/RNA filtering: {len(metadatas)}")

    # Filter with size constraint
    metadatas = [
        m for m in metadatas if sum(c.num_tokens for c in m.chains) <= MAX_TOKENS
    ]
    print(f"Entries after size filtering: {len(metadatas)}")

    # Filter with unknown residues
    npz_dir: pathlib.Path = data_dir / "npz/"
    _metadatas: list[Metadata] = []
    for m in metadatas:
        pdb_id = m.id
        npz_file_path = npz_dir / pdb_id[1:3] / f"{pdb_id}.npz"
        ref_struct = RefStructure.load_npz(npz_file_path)
        if all(
            c.get_sequence().count("X") == 0 for c in ref_struct.chains if c.is_protein
        ):
            _metadatas.append(m)
    metadatas = _metadatas
    print(f"Entries after unknown residue filtering: {len(metadatas)}")
    print()

    # ============================================================
    # Stage 3: Filter for Antibody-Antigen Entries
    # ============================================================
    print("Stage 3: Filter for Antibody-Antigen Entries")
    igg_entries: list[Metadata] = [m for m in metadatas if m.id in igg_pdb_ids]
    vhh_entries: list[Metadata] = [m for m in metadatas if m.id in vhh_pdb_ids]
    antibody_entries: list[Metadata] = igg_entries + vhh_entries
    print(f"IgG entries in recent PDB: {len(igg_entries)}")
    print(f"VHH entries in recent PDB: {len(vhh_entries)}")
    print(f"Total antibody-antigen entries in recent PDB: {len(antibody_entries)}")
    print()

    # ============================================================
    # Stage 4: Filter and Sample Entries
    # ============================================================
    print("Stage 4: Sample Antibody-Protein entries")
    sampled_igg_entries = sample_entries(igg_entries, antibody_dict, antigen_dict)
    sampled_vhh_entries = sample_entries(vhh_entries, antibody_dict, antigen_dict)
    sampled_entries = sampled_igg_entries + sampled_vhh_entries
    print(f"Total IgG entries sampled: {len(sampled_igg_entries)}")
    print(f"Total VHH entries sampled: {len(sampled_vhh_entries)}")
    print(f"Total antibody-antigen entries sampled: {len(sampled_entries)}")
    print()

    # ============================================================
    # Stage 4: Save Sampled PDB IDs and Metadata
    # ============================================================
    print("Stage 4: Save Sampled PDB IDs and Metadata")

    # Save sampled PDB IDs
    save_dir = data_dir / "assets/"
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb_id_path: pathlib.Path = save_dir / "antibody_antigen_ids.txt"
    sampled_pdb_ids = [m.id for m in sampled_entries]
    with open(pdb_id_path, "w") as w:
        for pid in sampled_pdb_ids:
            w.write(f"{pid}\n")
    print(f"Saved sampled PDB IDs to: {pdb_id_path}")

    metadata_path: pathlib.Path = save_dir / "antibody_antigen_metadata.csv"

    n_vhh: int = 0
    n_igg: int = 0
    n_interfaces: int = 0
    all_interface_clusters: set[str] = set()
    with open(metadata_path, "w") as w:
        w.write(
            "pdb_id,type,chain_id_1,chain_id_2,cluster_id_1,cluster_id_2,interface_cluster_id\n"
        )
        for m in sampled_entries:
            entry_type = "VHH" if m.id in vhh_pdb_ids else "IgG"
            if entry_type == "VHH":
                n_vhh += 1
            else:
                n_igg += 1

            antigen_ids = antigen_dict[m.id]
            antibody_ids = antibody_dict[m.id]

            for im in m.interfaces:
                asym_id1, asym_id2 = im.asym_ids
                cm1 = m.get_chain_by_asym_id(asym_id1)
                cm2 = m.get_chain_by_asym_id(asym_id2)

                if not (cm1.ctype.is_protein and cm2.ctype.is_protein):
                    # Skip non-protein interfaces
                    continue

                auth_id1, auth_id2 = cm1.auth_asym_id, cm2.auth_asym_id
                if not (
                    (auth_id1 in antibody_ids and auth_id2 in antigen_ids)
                    or (auth_id2 in antibody_ids and auth_id1 in antigen_ids)
                ):
                    # Skip non-antibody-antigen interfaces
                    continue

                if not im.is_low_homology:
                    # Skip non-low-homology interfaces
                    continue

                label_id1, label_id2 = cm1.label_asym_id, cm2.label_asym_id
                all_interface_clusters.add(im.cluster_id)  # type: ignore
                n_interfaces += 1
                w.write(
                    f"{m.id},{entry_type},{label_id1},{label_id2},{cm1.cluster_id},{cm2.cluster_id},{im.cluster_id}\n"
                )
    print(f"Saved sampled metadata to: {metadata_path}")

    # Final summary
    print(f"Total antibody-antigen interfaces collected: {n_interfaces}")
    print(f"Total unique interface clusters collected: {len(all_interface_clusters)}")


if __name__ == "__main__":
    main()
