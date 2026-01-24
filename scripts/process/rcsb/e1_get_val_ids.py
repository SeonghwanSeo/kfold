"""Construct validation set by filtering and clustering RCSB PDB entries.

Intermediate results:
--- Cluster-based sampling ---
# Multimer:
  Protein-Protein: 1707 -> 600
  Protein-DNA: 402 -> 200
  Protein-RNA: 184 -> 184
  Protein-Ligand: 2115 -> 500
  DNA-DNA: 293 -> 100
  DNA-RNA: 31 -> 31
  DNA-Ligand: 67 -> 50
  RNA-RNA: 42 -> 42
  RNA-Ligand: 16 -> 16
  Ligand-Ligand: 293 -> 0

# Monomer:
  DNA: 17
  RNA: 25

--- Final sampling ---
Multimer entries: 1265
Monomer entries: 42
Total entries: 1303
Final entries: 1280
"""

import argparse
import hashlib
import multiprocessing
import os
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any, NamedTuple, TypeVar

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.utils.io.fasta import read_fasta
from kfold.utils.mmseqs2 import run_mmseqs2_cluster, run_mmseqs2_search
from kfold.utils.rcsb_api import fetch_ranking_model_fit

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

_T = TypeVar("_T")


# Helper functions
def norm_key(key1: _T, key2: _T) -> tuple[_T, _T]:
    """Return a normalized tuple of two keys."""
    return (key1, key2) if key1 <= key2 else (key2, key1)


def get_rng(key: str) -> np.random.Generator:
    """Get a random number generator seeded by the given key."""
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


# Constants
VERBOSE = 0
INIT_MAX_TOKENS = 2560
FINAL_MAX_TOKENS = 2048
SEQUENCE_IDENTITY_THRESHOLD = 0.40
TANIMOTO_SIMILARITY_THRESHOLD = 0.85
NUM_INTERFACE_SAMPLES: dict[tuple[C.ChainType, C.ChainType], int] = {
    norm_key(C.ChainType.PROTEIN, C.ChainType.PROTEIN): 600,
    norm_key(C.ChainType.PROTEIN, C.ChainType.DNA): 200,
    norm_key(C.ChainType.PROTEIN, C.ChainType.RNA): 200,
    norm_key(C.ChainType.PROTEIN, C.ChainType.LIGAND): 500,
    norm_key(C.ChainType.DNA, C.ChainType.DNA): 100,
    norm_key(C.ChainType.DNA, C.ChainType.RNA): 50,
    norm_key(C.ChainType.DNA, C.ChainType.LIGAND): 50,
    norm_key(C.ChainType.RNA, C.ChainType.RNA): 50,
    norm_key(C.ChainType.RNA, C.ChainType.LIGAND): 50,
    norm_key(C.ChainType.LIGAND, C.ChainType.LIGAND): 0,
}
NUM_MONOMER_SAMPLES: dict[C.ChainType, int] = {}
FINAL_VALIDATION_SET_SIZE = 1280


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


def get_polymer_homologs(
    ctype: C.ChainType,
    queries: list[Seq],
    targets: list[Seq],
    mmseqs: str,
    sequence_identity: float = SEQUENCE_IDENTITY_THRESHOLD,
) -> dict[str, set[str]]:
    """Return high homology sequences from queries against targets."""
    assert ctype.is_polymer, "Polymer homology search only supports polymers."
    print(f"Getting {ctype.name} homologs with sequence identity >= 40%...")

    queries = [seq for seq in queries if seq.ctype == ctype]
    targets = [seq for seq in targets if seq.ctype == ctype]

    assert len(queries) > 0, f"No query sequences for chain type {ctype.name}."
    assert len(targets) > 0, f"No target sequences for chain type {ctype.name}."

    # Extract unique sequences for efficiency
    seq_to_query_ids: dict[str, set[str]] = defaultdict(set)
    for seq in queries:
        seq_to_query_ids[seq.sequence].add(seq.id)
    uniq_queries: dict[str, str] = {
        f"query-{i}": seq for i, seq in enumerate(sorted(seq_to_query_ids))
    }

    seq_to_target_ids: dict[str, set[str]] = defaultdict(set)
    for seq in targets:
        seq_to_target_ids[seq.sequence].add(seq.id)
    uniq_targets: dict[str, str] = {
        f"target-{i}": seq for i, seq in enumerate(sorted(seq_to_target_ids))
    }

    # Run MMseqs2 search
    homolog_out: dict[str, set[str]] = run_mmseqs2_search(
        uniq_queries,
        uniq_targets,
        chain_type=ctype.name.lower(),
        min_sequence_identity=sequence_identity,
        verbose=VERBOSE,
        print_cmd=(VERBOSE > 0),
        mmseqs2_exec=mmseqs,
    )

    # Map back to original IDs
    results: dict[str, set[str]] = {seq.id: set() for seq in queries}
    for quid, tuids in homolog_out.items():
        query_ids = seq_to_query_ids[uniq_queries[quid]]
        for tuid in tuids:
            target_ids = seq_to_target_ids[uniq_targets[tuid]]
            for qid in query_ids:
                results[qid].update(target_ids)

    # Manually add identical sequences
    for seq in queries:
        if seq.sequence in seq_to_target_ids:
            for target_id in seq_to_target_ids[seq.sequence]:
                results[seq.id].add(target_id)

    # print statistics
    n_low_homology = sum(1 for v in results.values() if len(v) == 0)
    print(f"Total query {ctype.name} chains: {len(results)}")
    print(f"Total target {ctype.name} chains: {len(targets)}")
    print(f"Low homology {ctype.name} chains: {n_low_homology}")
    return results


def get_ligand_homologs(
    queries: list[Seq],
    targets: list[Seq],
    ccd: CCD,
    tanimoto_threshold: float = TANIMOTO_SIMILARITY_THRESHOLD,
) -> dict[str, set[str]]:
    """Return high homology ligands from queries against targets."""
    print(f"Getting ligand homologs with Tanimoto threshold {tanimoto_threshold}...")
    queries = [seq for seq in queries if seq.ctype.is_ligand]
    targets = [seq for seq in targets if seq.ctype.is_ligand]

    assert len(queries) > 0, "No query sequences for ligands."
    assert len(targets) > 0, "No target sequences for ligands."

    target_ccd_to_ids: dict[str, set[str]] = defaultdict(set)
    for seq in targets:
        for code in seq.sequence.split("-"):
            target_ccd_to_ids[code].add(seq.id)
    print(f"Target ligand CCDs loaded: {len(target_ccd_to_ids)}")

    # Filter ligands with tanimoto similarity >= 0.85 to training set
    target_fps: list[Any] = []
    target_fp_ccds: list[str] = []
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    for code in sorted(target_ccd_to_ids):
        mol = ccd[code].mol
        Chem.SanitizeMol(mol, catchErrors=True)
        try:
            fp = fpgen.GetFingerprint(mol)
        except Exception:
            if VERBOSE:
                print(f"Error computing fingerprint for CCD {code}")
            continue
        target_fps.append(fp)
        target_fp_ccds.append(code)
    assert len(target_fps) > 0, "No valid target ligand fingerprints computed."
    print(
        f"Target ligand fingerprints computed: "
        f"{len(target_fps)} out of {len(target_ccd_to_ids)}"
    )

    # Load query ligand CCDs
    query_ccd_to_ids: dict[str, set[str]] = defaultdict(set)
    for seq in queries:
        for code in seq.sequence.split("-"):
            query_ccd_to_ids[code].add(seq.id)
    print(f"Query ligand CCDs loaded: {len(query_ccd_to_ids)}")

    # Compute homology
    results: dict[str, set[str]] = {seq.id: set() for seq in queries}
    for code in tqdm(sorted(query_ccd_to_ids), desc="Tanimoto Similarity Check"):
        mol = ccd[code].mol
        Chem.SanitizeMol(mol, catchErrors=True)
        sim_ccds: list[str] = []
        try:
            # Sanitize molecule
            fp = fpgen.GetFingerprint(mol)
        except Exception as e:
            # If fingerprint computation fails, pass
            if VERBOSE:
                print(f"Error computing fingerprint for CCD {code}: {e}")
        else:
            similarities = BulkTanimotoSimilarity(fp, target_fps)
            for i, sim in enumerate(similarities):
                if sim >= tanimoto_threshold:
                    sim_ccds.append(target_fp_ccds[i])

        # Record all similar sequences
        for seq_id in query_ccd_to_ids[code]:
            for sim_ccd in sim_ccds:
                results[seq_id].update(target_ccd_to_ids[sim_ccd])

    # Manually add identical CCDs
    for code in query_ccd_to_ids:
        if code in target_ccd_to_ids:
            for seq_id in query_ccd_to_ids[code]:
                results[seq_id].update(target_ccd_to_ids[code])

    # print statistics
    n_low_homology = sum(1 for v in results.values() if len(v) == 0)
    print(f"Total query ligands: {len(query_ccd_to_ids)}")
    print(f"Low homology ligands (no similar in targets): {n_low_homology}")
    return results


def run_clustering(
    all_sequences: list[Seq],
    mmseqs: str,
    sequence_identity: float = SEQUENCE_IDENTITY_THRESHOLD,
) -> dict[str, dict[str, str]]:
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
    if len(uniq_proteins) > 0:
        protein_cluster_map = run_mmseqs2_cluster(
            uniq_proteins,
            min_sequence_identity=sequence_identity,
            chain_type="protein",
            verbose=VERBOSE,
            print_cmd=(VERBOSE > 0),
            mmseqs2_exec=mmseqs,
        )
    else:
        protein_cluster_map = {}

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


def filter_multier_interfaces(
    all_interfaces: list[tuple[Seq, Seq]],
    train_sequences: list[Seq],
    mmseqs: str,
    ccd: CCD,
) -> list[tuple[Seq, Seq]]:
    """Filter multimer interfaces according to homology and clustering."""

    print("=" * 50)
    print("Multimer Interface Filtering")
    print("Total interfaces before filtering:", len(all_interfaces))

    # ============================================================
    # Determine low homology interfaces
    # ============================================================
    print("\nStage 1-1: Get homology mappings for all sequences...")
    all_sequences: list[Seq] = []
    collected_ids: set[str] = set()
    for seq1, seq2 in all_interfaces:
        for seq in (seq1, seq2):
            if seq.id not in collected_ids:
                collected_ids.add(seq.id)
                all_sequences.append(seq)
    print(f"Total sequences in interfaces: {len(all_sequences)}")

    # Determine low homology polymers
    protein_homologs: dict[str, set[str]] = get_polymer_homologs(
        C.ChainType.PROTEIN, all_sequences, train_sequences, mmseqs
    )
    dna_homologs: dict[str, set[str]] = get_polymer_homologs(
        C.ChainType.DNA, all_sequences, train_sequences, mmseqs
    )
    rna_homologs: dict[str, set[str]] = get_polymer_homologs(
        C.ChainType.RNA, all_sequences, train_sequences, mmseqs
    )
    # ... and low homology ligands
    ligand_homologs: dict[str, set[str]] = get_ligand_homologs(
        all_sequences, train_sequences, ccd
    )

    # Combine homology results
    seq_homologs_map: dict[str, set[str]] = (
        protein_homologs | dna_homologs | rna_homologs | ligand_homologs
    )
    assert len(seq_homologs_map) == (
        len(protein_homologs)
        + len(dna_homologs)
        + len(rna_homologs)
        + len(ligand_homologs)
    ), "Homology map size mismatch."
    assert len(seq_homologs_map) == len(all_sequences), (
        f"Some sequences missing in homology map."
        f" Expected: {len(all_sequences)}, Got: {len(seq_homologs_map)}"
    )
    print("Homology search completed.")

    # ============================================================
    # Filter interfaces
    # ============================================================
    print("\nStage 1-2: Homology filtering of interfaces...")
    # Just keep pdb_id level homologs for interface filtering
    high_homology_pdbs: dict[str, set[str]] = {
        seq_id: set(h.split("_")[0] for h in homologs)
        for seq_id, homologs in seq_homologs_map.items()
    }
    # Filter out high homology interfaces, defined as interfaces contains
    # two chains with high homology to any target in training set.
    # In addition to AF3 protocol, also filter out ion - high homology chain interfaces.
    is_ion = lambda seq: seq.ctype.is_ligand and seq.sequence in C.ccd.IONS  # noqa

    n_high_homology = 0
    n_ion_leakage = 0
    filtered_interfaces: list[tuple[Seq, Seq]] = []
    for seq1, seq2 in tqdm(all_interfaces, desc="Homology Filtering"):
        # Multi-residue ligands should have been filtered out.
        assert seq1.sequence.count("-") == 0
        assert seq2.sequence.count("-") == 0
        # Check that there is any target with high homology to both chains
        train_pdb1 = high_homology_pdbs[seq1.id]
        train_pdb2 = high_homology_pdbs[seq2.id]
        if len(train_pdb1 & train_pdb2) > 0:
            n_high_homology += 1
            continue

        # Filter out high-homology chain - ion interfaces
        if (is_ion(seq1) and len(train_pdb2) > 0) or (
            is_ion(seq2) and len(train_pdb1) > 0
        ):
            n_ion_leakage += 1
            continue

        # Keep the interface
        filtered_interfaces.append((seq1, seq2))

    print(f"Total high homology interfaces filtered: {n_high_homology}")
    print(f"Total ion - high homology chain interfaces filtered: {n_ion_leakage}")
    print("Total interfaces after homology filtering:", len(filtered_interfaces))

    # ============================================================
    # Filter ligand interfaces by ranking model fit
    # ============================================================
    print("\nStage 1-3: Filtering ligand interfaces by ranking model fit...")

    # Collect ligand entities in interfaces
    ligand_entities: set[str] = set()
    for seq1, seq2 in filtered_interfaces:
        for seq in (seq1, seq2):
            if seq.ctype.is_ligand:
                ligand_entities.add(seq.id.upper())  # RCSB uses uppercase IDs

    # Fetch ranking model fit scores
    print(f"Fetching ranking model fit scores for {len(ligand_entities)} ligands...")
    ranking_model_fits: dict[str, float] = {}
    ligand_entities: list[str] = sorted(ligand_entities)
    for i in tqdm(range(0, len(ligand_entities), 1000)):
        ranking_model_fits |= fetch_ranking_model_fit(ligand_entities[i : i + 1000])
    print("Total ligands with ranking model fit scores:", len(ranking_model_fits))

    # Determine ligands to exclude
    excluding_ligands: set[str] = set(
        seq_id.lower()  # convert back to lowercase IDs
        for seq_id in ligand_entities
        if ranking_model_fits.get(seq_id, 0.0) < 0.5
    )
    del ligand_entities  # free up memory

    # Filter interfaces with low ranking model fit ligands
    filtered_interfaces: list[tuple[Seq, Seq]] = [
        iface
        for iface in filtered_interfaces
        if all(seq.id not in excluding_ligands for seq in iface)
    ]
    print("Total interfaces after ranking model fit filtering:", len(filtered_interfaces))

    # ============================================================
    # Clustering and sampling interfaces
    # ============================================================
    print("\nStage 2-1: Clustering interfaces...")
    # Collect all sequences
    all_sequences: list[Seq] = []
    for seq1, seq2 in filtered_interfaces:
        all_sequences.extend([seq1, seq2])
    # Run clustering
    seq_to_clusters: dict[str, dict[str, str]] = run_clustering(
        all_sequences, mmseqs=mmseqs
    )
    # Interface-level clustering
    interface_clusters: dict[str, list[tuple[Seq, Seq]]] = defaultdict(list)
    for seq1, seq2 in filtered_interfaces:
        cluster_id1 = seq_to_clusters[seq1.ctype_str][seq1.sequence]
        cluster_id2 = seq_to_clusters[seq2.ctype_str][seq2.sequence]
        cluster_id = ":".join(norm_key(cluster_id1, cluster_id2))
        interface_clusters[cluster_id].append((seq1, seq2))

    # Sample one interface per cluster
    print("\nStage 2-2: Sample one interface per cluster...")
    sampled_interfaces: list[tuple[Seq, Seq]] = []
    for cluster_id, interfaces in interface_clusters.items():
        n_cluster = len(interfaces)
        rng = get_rng(cluster_id)
        sampled_interfaces.append(interfaces[rng.integers(n_cluster)])
    print(f"Total interfaces after filtering and clustering: {len(sampled_interfaces)}")

    # ============================================================
    # Final sampling for each interface type
    # ============================================================
    print("\nStage 3: Final sampling interfaces for each interface type")
    interfaces_per_type = defaultdict(list)
    for seq1, seq2 in sampled_interfaces:
        ctypes = norm_key(seq1.ctype, seq2.ctype)
        interfaces_per_type[ctypes].append((seq1, seq2))
    del sampled_interfaces  # free up memory

    sampled_interfaces: list[tuple[Seq, Seq]] = []
    for ctypes in sorted(interfaces_per_type):
        key = f"{ctypes[0]}-{ctypes[1]}"
        interfaces = interfaces_per_type[ctypes]
        rng = get_rng(key)
        n_interfaces = len(interfaces)
        n_samples = min(NUM_INTERFACE_SAMPLES.get(ctypes, n_interfaces), n_interfaces)
        if n_samples == 0:
            pass
        elif n_interfaces == n_samples:
            sampled_interfaces.extend(interfaces)
        else:
            sampled_indices = rng.choice(len(interfaces), size=n_samples, replace=False)
            for idx in sampled_indices:
                sampled_interfaces.append(interfaces[idx])
        print(f"  {key}: {n_interfaces} -> {n_samples}")

    print("\nMultimer filtering completed.")
    print(f"Total interfaces after final sampling: {len(sampled_interfaces)}")
    return sampled_interfaces


def filter_monomers(
    all_polymers: list[Seq],
    train_sequences: list[Seq],
    mmseqs: str,
) -> list[Seq]:
    """Filter monomer chains according to homology and clustering."""

    print("\n" + "=" * 50)
    print("Polymer Monomer Filtering")
    print("Total polymers before filtering:", len(all_polymers))

    # ============================================================
    # Determine low homology polymers
    # ============================================================
    print("\nStage 1-1: Get homology mappings for all sequences...")
    dna_homologs: dict[str, set[str]] = get_polymer_homologs(
        C.ChainType.DNA, all_polymers, train_sequences, mmseqs
    )
    rna_homologs: dict[str, set[str]] = get_polymer_homologs(
        C.ChainType.RNA, all_polymers, train_sequences, mmseqs
    )

    # Combine homology results
    seq_homologs_map: dict[str, set[str]] = dna_homologs | rna_homologs
    assert len(seq_homologs_map) == (len(dna_homologs) + len(rna_homologs)), (
        "Homology map size mismatch."
    )
    assert len(seq_homologs_map) == len(all_polymers), (
        "Some sequences missing in homology map."
    )

    # ============================================================
    # Filter polymers
    # ============================================================
    print("\nStage 1-2: Homology filtering of polymers...")
    filtered_polymers: list[Seq] = [
        seq for seq in all_polymers if len(seq_homologs_map[seq.id]) == 0
    ]
    print("Total polymers after homology filtering:", len(filtered_polymers))

    # ============================================================
    # Collect all sequences
    # ============================================================
    print("\nStage 2: Collect sequences...")
    polymers_per_ctype = defaultdict(list)
    for seq in filtered_polymers:
        polymers_per_ctype[seq.ctype].append(seq)
    del filtered_polymers  # free up memory

    sampled_polymers: list[Seq] = []
    for ctype in sorted(polymers_per_ctype):
        polymers = polymers_per_ctype[ctype]
        sampled_polymers.extend(polymers)
        print(f"  {ctype}: {len(polymers)}")

    print("\nMonomer filtering completed.")
    print(f"Total polymers after final sampling: {len(sampled_polymers)}")
    return sampled_polymers


def read_npz_file(npz_file: pathlib.Path) -> dict:
    """Read NPZ file and extract sequences."""
    # Load structure
    struct: RefStructure = RefStructure.load_npz(npz_file)
    assert struct.num_chains > 0
    pdb_id = struct.id

    # Get entity sequences
    entity_sequences: dict[int, Seq] = {}
    for chain in struct.chains:
        entity_id = chain.entity_id
        if entity_id not in entity_sequences:
            if chain.is_polymer:
                # Mapping to standard residues
                sequence = chain.get_sequence(map_to_standard=True)
            else:
                # Handle multi-residue ligands (this will be filtered out later)
                sequence = "-".join(chain.get_ccd_sequence())
            seq = Seq(pdb_id, entity_id, sequence, chain.ctype)
            entity_sequences[entity_id] = seq

    # Extract monomers and interfaces
    all_interfaces: list[tuple[Seq, Seq]] = []
    all_monomers: list[Seq] = []

    # Interfaces
    m: Metadata = struct.metadata
    visited_iface_entities: set[tuple[int, int]] = set()
    for interface in m.interfaces:
        asym_id_1, asym_id_2 = interface.asym_ids
        c1: Chain = struct.get_chain_by_asym_id(asym_id_1)
        c2: Chain = struct.get_chain_by_asym_id(asym_id_2)
        eid1, eid2 = c1.entity_id, c2.entity_id
        key = norm_key(eid1, eid2)

        # Skip duplicate interfaces
        if key in visited_iface_entities:
            continue

        # Skip multi-residue non-polymers in interfaces
        if (c1.is_nonpolymer and c1.num_residues > 1) or (
            c2.is_nonpolymer and c2.num_residues > 1
        ):
            continue

        # Add interface
        seq1 = entity_sequences[c1.entity_id]
        seq2 = entity_sequences[c2.entity_id]
        all_interfaces.append((seq1, seq2))
        visited_iface_entities.add(key)

    # Monomers (Nucleic acids only)
    if struct.num_polymer_chains == 1:
        chain: Chain = next(c for c in struct.chains if c.is_polymer)
        assert chain.is_polymer, "Monomer chain must be polymer."
        seq = entity_sequences[chain.entity_id]
        # Add nucleic acid monomers only
        if chain.is_nucleic_acid:
            all_monomers.append(seq)

    return {
        "pdb_id": pdb_id,
        "num_tokens": struct.num_tokens,
        "monomers": all_monomers,
        "interfaces": all_interfaces,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Construct validation set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the training preprocessed data directory.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        required=True,
        help="Path to the CCD pickle file.",
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
        help="Number of parallel workers to use.",
    )
    args = parser.parse_args()

    return args


def main():
    """Main function to construct validation set.

    See AlphaFold3 Supplementary Section 5.8 for details: Multimer and Monomer selection.
    """
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    train_dir: pathlib.Path = args.train_data_dir

    # ======================================================================
    # Load training sequences
    # ======================================================================
    train_seqs: list[Seq] = []
    train_seq_fasta: pathlib.Path = train_dir / "sequences" / "all_sequences.fasta"
    for header, sequence in read_fasta(train_seq_fasta):
        pdb_id, entity_id, ctype_str = header.split("|")
        entity_id = int(entity_id)
        ctype = C.ChainType[ctype_str.upper()]
        if ctype.is_protein:
            # Map ambiguous amino acids to standard ones
            sequence = "".join(
                C.residue.PROTEIN_AMINO_ACID_MAPPING.get(aa, aa) for aa in sequence
            )
        train_seqs.append(Seq(pdb_id, entity_id, sequence, ctype))

    # ======================================================================
    # Load validation candidates from NPZ files
    # ======================================================================
    npz_files: list[pathlib.Path] = list((data_dir / "npz").rglob("*.npz"))
    print(f"Total NPZ files found: {len(npz_files)}")

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(read_npz_file, npz_files, chunksize=10),
                total=len(npz_files),
                desc="Processing NPZ files",
            )
        )

    # Collect all monomers and interfaces
    all_interfaces: list[tuple[Seq, Seq]] = []
    all_monomers: list[Seq] = []
    entry_size: dict[str, int] = {}
    for res in results:
        if res["num_tokens"] > INIT_MAX_TOKENS:
            continue
        entry_size[res["pdb_id"]] = res["num_tokens"]
        all_interfaces.extend(res["interfaces"])
        all_monomers.extend(res["monomers"])

    # sort interfaces and monomers
    all_interfaces.sort(key=lambda x: (x[0].pdb_id, x[0].entity_id, x[1].entity_id))
    all_monomers.sort(key=lambda x: (x.pdb_id, x.entity_id))

    print(f"Total entries processed: {len(entry_size)}.")
    print(f"Total interfaces collected: {len(all_interfaces)}")
    print(f"Total monomers collected: {len(all_monomers)}")

    # Print composition:
    print("Interface type statistics:")
    n_iface_types = defaultdict(int)
    for seq1, seq2 in all_interfaces:
        ctypes = norm_key(seq1.ctype, seq2.ctype)
        n_iface_types[ctypes] += 1
    for ctypes in sorted(n_iface_types.keys()):
        key = f"{ctypes[0]}-{ctypes[1]}"
        print(f"  {key}: {n_iface_types[ctypes]}")
    print()
    print("Monomer type statistics:")
    n_monomer_types = defaultdict(int)
    for seq in all_monomers:
        n_monomer_types[seq.ctype] += 1
    for ctype in sorted(n_monomer_types.keys()):
        print(f"  {ctype}: {n_monomer_types[ctype]}")
    print()

    # ======================================================================
    # Multimer filtering
    # ======================================================================
    ccd = CCD.load(args.ccd_path)
    samples_interfaces: list[tuple[Seq, Seq]] = filter_multier_interfaces(
        all_interfaces=all_interfaces,
        train_sequences=train_seqs,
        mmseqs=args.mmseqs,
        ccd=ccd,
    )
    del ccd  # free up memory

    # Collect PDB IDs from multimer filtering
    multimer_ids: set[str] = set()
    for seq1, seq2 in samples_interfaces:
        assert seq1.pdb_id == seq2.pdb_id, "Interface chains must belong to the same PDB."
        multimer_ids.add(seq1.pdb_id)
    print(f"Multimer PDB entries: {len(multimer_ids)}")

    # Filter with max token limit
    multimer_ids = {v for v in multimer_ids if entry_size[v] <= FINAL_MAX_TOKENS}
    print(f"Multimer PDB entries after token limit filtering: {len(multimer_ids)}")

    # ======================================================================
    # Monomer filtering
    # ======================================================================
    sampled_monomers: list[Seq] = filter_monomers(
        all_polymers=all_monomers,
        train_sequences=train_seqs,
        mmseqs=args.mmseqs,
    )
    # Collect PDB IDs from monomer filtering
    monomer_ids: set[str] = set()
    for seq in sampled_monomers:
        monomer_ids.add(seq.pdb_id)
    print(f"Monomer PDB entries: {len(monomer_ids)}")

    # Filter with max token limit
    monomer_ids = {v for v in monomer_ids if entry_size[v] <= FINAL_MAX_TOKENS}
    print(f"Monomer PDB entries after token limit filtering: {len(monomer_ids)}")

    # ======================================================================
    # Final validation set sampling
    # ======================================================================
    print("\n" + "=" * 50)
    sampled_ids = multimer_ids | monomer_ids
    if len(sampled_ids) > FINAL_VALIDATION_SET_SIZE:
        val_ids: list[str] = sorted(sampled_ids)
        sampled_indices = get_rng("final").choice(
            len(val_ids), size=FINAL_VALIDATION_SET_SIZE, replace=False
        )
        val_ids = [val_ids[i] for i in sorted(sampled_indices)]
    else:
        val_ids = sorted(sampled_ids)

    print("Validation Set Final Summary")
    print(f"Multimer entries: {len(multimer_ids)}")
    print(f"Monomer entries: {len(monomer_ids)}")
    print(f"Total entries: {len(sampled_ids)}")
    print(f"Final entries: {len(val_ids)}")

    # Save validation set PDB IDs
    val_ids_file: pathlib.Path = data_dir / "validation_ids.txt"
    with val_ids_file.open("w") as f:
        for pdb_id in sorted(val_ids):
            f.write(f"{pdb_id}\n")
    print(f"Validation set PDB IDs saved to: {val_ids_file}")

    # Summarize final validation set statistics
    print("\nExtracting final validation set statistics...")
    chains_per_ctype = defaultdict(int)
    interfaces_per_ctype = defaultdict(int)

    earlest_release_date = datetime.max
    latest_release_date = datetime.min
    npz_files: list[pathlib.Path] = [f for f in npz_files if f.stem in val_ids]
    for f in npz_files:
        struct: RefStructure = RefStructure.load_npz(f)
        pdb_id = struct.id

        # Update date
        release_date = datetime.fromisoformat(struct.metadata.exp.release_date)
        earlest_release_date = min(earlest_release_date, release_date)
        latest_release_date = max(latest_release_date, release_date)

        # Collect type info
        asym_id_to_type: dict[int, str] = {}
        for chain in struct.chains:
            ctype_str = str(chain.ctype)
            chains_per_ctype[ctype_str] += 1
            asym_id_to_type[chain.asym_id] = ctype_str
        m: Metadata = struct.metadata
        for interface in m.interfaces:
            asym_id_1, asym_id_2 = interface.asym_ids
            ctype1 = asym_id_to_type[asym_id_1]
            ctype2 = asym_id_to_type[asym_id_2]
            ctypes = norm_key(ctype1, ctype2)
            interfaces_per_ctype[ctypes] += 1

    print("Release date range:")
    print(f"  Earliest: {earlest_release_date.date().isoformat()}")
    print(f"  Latest: {latest_release_date.date().isoformat()}")
    print()
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
