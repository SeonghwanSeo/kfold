"""Construct test set by filtering RCSB PDB entries."""

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import pathlib
from collections import defaultdict
from typing import Any, NamedTuple, TypeVar

import msgpack
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import read_fasta
from kfold.utils.mmseqs2 import run_mmseqs2_cluster, run_mmseqs2_search

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
MAX_TOKENS = 5120
MAX_POLYMER_RESIDUES = 1280  # Same to AlphaFoldDB UniProt length limit
SEQUENCE_IDENTITY_THRESHOLD = 0.40
TANIMOTO_SIMILARITY_THRESHOLD = 0.85


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


class Interface(NamedTuple):
    seq1: Seq
    seq2: Seq

    @property
    def pdb_id(self) -> str:
        return self.seq1.pdb_id

    @property
    def id(self) -> str:
        eid1, eid2 = norm_key(self.seq1.entity_id, self.seq2.entity_id)
        return f"{self.pdb_id}_{eid1}:{eid2}"


def read_npz_file(npz_file: pathlib.Path, return_metadata: bool = False) -> dict:
    """Read NPZ file and extract sequences."""
    # Load structure
    struct: RefStructure = RefStructure.load_npz(npz_file)
    pdb_id = struct.id

    # Get entity sequences
    entity_sequences: dict[int, Seq] = {}
    for chain in struct.chains:
        entity_id = chain.entity_id
        if entity_id not in entity_sequences:
            if chain.is_polymer:
                sequence = chain.get_sequence(map_to_standard=True)
            else:
                sequence = "-".join(chain.get_ccd_sequence())
            seq = Seq(pdb_id, entity_id, sequence, chain.ctype)
            entity_sequences[entity_id] = seq
    all_chains: list[Seq] = list(entity_sequences.values())

    # Extract interfaces without duplicates
    all_interfaces: list[Interface] = []
    visited: set[tuple[int, int]] = set()
    m: Metadata = struct.metadata
    for interface in m.interfaces:
        asym_id_1, asym_id_2 = interface.asym_ids
        eid1 = struct.get_chain_by_asym_id(asym_id_1).entity_id
        eid2 = struct.get_chain_by_asym_id(asym_id_2).entity_id
        key = norm_key(eid1, eid2)
        if key in visited:
            continue
        visited.add(key)
        seq1 = entity_sequences[eid1]
        seq2 = entity_sequences[eid2]
        all_interfaces.append(Interface(seq1, seq2))

    out = {
        "pdb_id": pdb_id,
        "num_tokens": struct.num_tokens,
        "chains": all_chains,
        "interfaces": all_interfaces,
        "is_monomer": struct.num_polymer_chains == 1,
    }
    if return_metadata:
        out["metadata"] = m
    return out


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

    seq_to_target_pdbs: dict[str, set[str]] = defaultdict(set)
    for seq in targets:
        seq_to_target_pdbs[seq.sequence].add(seq.pdb_id)
    uniq_targets: dict[str, str] = {
        f"target-{i}": seq for i, seq in enumerate(sorted(seq_to_target_pdbs))
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
            target_pdbs = seq_to_target_pdbs[uniq_targets[tuid]]
            for qid in query_ids:
                results[qid].update(target_pdbs)

    # Manually add identical sequences
    for seq in queries:
        if seq.sequence in seq_to_target_pdbs:
            target_pdbs = seq_to_target_pdbs[seq.sequence]
            results[seq.id].update(target_pdbs)

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

    target_ccd_to_pdbs: dict[str, set[str]] = defaultdict(set)
    for seq in targets:
        for code in seq.sequence.split("-"):
            target_ccd_to_pdbs[code].add(seq.pdb_id)
    print(f"Target ligand CCDs loaded: {len(target_ccd_to_pdbs)}")

    # Filter ligands with tanimoto similarity >= 0.85 to training set
    target_fps: list[Any] = []
    target_fp_ccds: list[str] = []
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    for code in sorted(target_ccd_to_pdbs):
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
        f"{len(target_fps)} out of {len(target_ccd_to_pdbs)}"
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
                results[seq_id].update(target_ccd_to_pdbs[sim_ccd])

    # Manually add identical CCDs
    for code in query_ccd_to_ids:
        if code in target_ccd_to_pdbs:
            for seq_id in query_ccd_to_ids[code]:
                results[seq_id].update(target_ccd_to_pdbs[code])

    # print statistics
    n_low_homology = sum(1 for v in results.values() if len(v) == 0)
    print(f"Total query ligands: {len(query_ccd_to_ids)}")
    print(f"Low homology ligands (no similar in targets): {n_low_homology}")
    return results


def run_clustering(
    all_sequences: list[Seq],
    mmseqs: str,
    sequence_identity: float = SEQUENCE_IDENTITY_THRESHOLD,
) -> dict[str, str]:
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

    print()
    print("Total clusters: ")
    print(f"  Proteins (>=10 aa): {len(set(protein_clusters.values()))}")
    print(f"  Short Proteins (<10 aa): {len(set(short_protein_clusters.values()))}")
    print(f"  DNAs: {len(set(dna_clusters.values()))}")
    print(f"  RNAs: {len(set(rna_clusters.values()))}")
    print(f"  Ligands: {len(set(ligand_clusters.values()))}")

    # Return clustering mapping
    cluster_mapping: dict[C.ChainType, dict[str, str]] = {
        C.ChainType.PROTEIN: protein_clusters | short_protein_clusters,
        C.ChainType.DNA: dna_clusters,
        C.ChainType.RNA: rna_clusters,
        C.ChainType.LIGAND: ligand_clusters,
    }

    clustering: dict[str, str] = {
        seq.id: cluster_mapping[seq.ctype][seq.sequence] for seq in all_sequences
    }
    return clustering


def filter_multier_interfaces(
    all_interfaces: list[Interface],
    train_sequences: list[Seq],
    mmseqs: str,
    ccd: CCD,
) -> list[Interface]:
    """Filter multimer interfaces according to homology and clustering."""
    print("=" * 50)
    print("Multimer Interface Filtering")
    print("Total interfaces before filtering:", len(all_interfaces))

    # ============================================================
    # Pre-filter interfaces
    # ============================================================
    print("\nStage 1: Pre-filter interfaces...")
    is_ion = lambda seq: seq.ctype.is_ligand and seq.sequence in C.ccd.IONS  # noqa
    is_peptide = lambda seq: seq.ctype.is_protein and len(seq.sequence) < 16  # noqa
    interfaces: list[Interface] = []
    for iface in all_interfaces:
        seq1, seq2 = iface
        # Skip peptide-peptide interfaces
        if is_peptide(seq1) and is_peptide(seq2):
            continue
        # Skip ligand-ligand interfaces
        if seq1.ctype.is_ligand and seq2.ctype.is_ligand:
            continue
        # Skip multi-residue non-polymers in interfaces
        if "-" in seq1.sequence or "-" in seq2.sequence:
            continue
        # Skip ion in interfaces
        if is_ion(seq1) or is_ion(seq2):
            continue
        interfaces.append(iface)
    all_interfaces = interfaces
    print("Total interfaces after pre-filtering:", len(all_interfaces))

    # ============================================================
    # Determine low homology interfaces
    # ============================================================
    print("\nStage 2-1: Get homology mappings for all sequences...")
    all_sequences: list[Seq] = []
    collected_ids: set[str] = set()
    for seq1, seq2 in sorted(all_interfaces):
        for seq in (seq1, seq2):
            if seq.id not in collected_ids:
                collected_ids.add(seq.id)
                all_sequences.append(seq)
    del collected_ids
    print(f"Total sequences in interfaces: {len(all_sequences)}")

    # Determine low homology polymers
    homologs: dict[str, set[str]] = {}
    homologs |= get_polymer_homologs(
        C.ChainType.PROTEIN, all_sequences, train_sequences, mmseqs
    )
    homologs |= get_polymer_homologs(
        C.ChainType.DNA, all_sequences, train_sequences, mmseqs
    )
    homologs |= get_polymer_homologs(
        C.ChainType.RNA, all_sequences, train_sequences, mmseqs
    )
    # ... and low homology ligands
    homologs |= get_ligand_homologs(all_sequences, train_sequences, ccd)
    assert len(homologs) == len(all_sequences), "Homology mapping incomplete."
    print("Homology search completed.")

    # ============================================================
    # Filter interfaces
    # ============================================================
    print("\nStage 2-2: Homology filtering of interfaces...")
    # Filter out high homology interfaces, defined as interfaces contains
    # two chains with high homology to any target in training set.
    filtered_interfaces: list[Interface] = []
    for iface in tqdm(all_interfaces, desc="Homology Filtering"):
        train_pdb1 = homologs[iface.seq1.id]
        train_pdb2 = homologs[iface.seq2.id]
        if len(train_pdb1 & train_pdb2) > 0:
            # Skip if there is any target with homology to both chains
            continue
        if (is_peptide(iface.seq1) and len(train_pdb2) > 0) or (
            is_peptide(iface.seq2) and len(train_pdb1) > 0
        ):
            # Skip if one chain is a peptide and the other has homology
            continue
        filtered_interfaces.append(iface)
    print(
        f"Total interfaces after homology filtering: {len(filtered_interfaces)} "
        f"out of {len(all_interfaces)}"
    )

    return filtered_interfaces


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
    # Pre-filter monomers
    # ============================================================
    print("\nStage 1: Extract RNA polymers...")
    monomers: list[Seq] = []
    for seq in all_polymers:
        if seq.ctype.is_rna:
            monomers.append(seq)
    all_polymers = monomers
    print("Total polymers after pre-filtering:", len(all_polymers))

    # ============================================================
    # Determine low homology rnas
    # ============================================================
    print("\nStage 2-1: Get homology mappings for all sequences...")
    homologs: dict[str, set[str]] = {}
    homologs |= get_polymer_homologs(
        C.ChainType.RNA, all_polymers, train_sequences, mmseqs
    )
    assert len(homologs) == len(all_polymers), "Homology mapping incomplete."

    # ============================================================
    # Filter polymers
    # ============================================================
    print("\nStage 2-2: Homology filtering of polymers...")
    filtered_polymers: list[Seq] = [
        seq for seq in all_polymers if len(homologs[seq.id]) == 0
    ]
    print("Total polymers after homology filtering:", len(filtered_polymers))
    return filtered_polymers


def save_metadata(
    npz_files: list[pathlib.Path],
    save_dir: pathlib.Path,
    train_sequences: list[Seq],
    mmseqs: str,
    ccd: CCD,
    num_workers: int = 1,
) -> list[Metadata]:
    """Save metadata files with cluster ids"""
    print("=" * 50)
    print("Saving Metadata Files")
    print("Total entries:", len(npz_files))

    # ============================================================
    # Collect all sequences
    # ============================================================
    all_chains: list[Seq] = []
    all_interfaces: list[Interface] = []
    metadatas: list[Metadata] = []
    func = functools.partial(read_npz_file, return_metadata=True)
    with multiprocessing.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(func, npz_files, chunksize=10),
                total=len(npz_files),
                desc="Processing NPZ files",
            )
        )
    for res in results:
        all_chains.extend(res["chains"])
        all_interfaces.extend(res["interfaces"])
        metadatas.append(res["metadata"])

    seq_id_to_seq: dict[str, Seq] = {seq.id: seq for seq in all_chains}

    # ============================================================
    # Homology search
    # ============================================================
    print("\nStage 1: Homology search for all sequences...")
    all_sequences = all_chains
    print(f"Total sequences collected: {len(all_sequences)}")
    homologs: dict[str, set[str]] = {}
    homologs |= get_polymer_homologs(
        C.ChainType.PROTEIN, all_sequences, train_sequences, mmseqs
    )
    homologs |= get_polymer_homologs(
        C.ChainType.DNA, all_sequences, train_sequences, mmseqs
    )
    homologs |= get_polymer_homologs(
        C.ChainType.RNA, all_sequences, train_sequences, mmseqs
    )
    homologs |= get_ligand_homologs(all_sequences, train_sequences, ccd)
    assert len(homologs) == len(all_sequences), "Homology mapping incomplete."
    print("Homology search completed.")

    # ============================================================
    # Clustering
    # ============================================================
    print("\nStage 2: Clustering all sequences...")
    clusters: dict[str, str] = run_clustering(all_sequences, mmseqs)

    # ============================================================
    # Save metadata files
    # ============================================================
    print("\nStage 2: Saving metadata files...")
    for m in metadatas:
        # Update homology information
        for cm in m.chains:
            seq = seq_id_to_seq[f"{m.id}_{cm.entity_id}"]
            cm.cluster_id = clusters[seq.id]
            cm.is_low_homology = len(homologs[seq.id]) == 0

        for im in m.interfaces:
            asym_id_1, asym_id_2 = im.asym_ids
            cm1 = m.get_chain_by_asym_id(asym_id_1)
            cm2 = m.get_chain_by_asym_id(asym_id_2)
            seq1 = seq_id_to_seq[f"{m.id}_{cm1.entity_id}"]
            seq2 = seq_id_to_seq[f"{m.id}_{cm2.entity_id}"]
            cluster_id1, cluster_id2 = clusters[seq1.id], clusters[seq2.id]
            if seq1.ctype.is_polymer and seq2.ctype.is_ligand:
                im.cluster_id = f"{cluster_id1}"
            elif seq1.ctype.is_ligand and seq2.ctype.is_polymer:
                im.cluster_id = f"{cluster_id2}"
            else:
                cid1, cid2 = norm_key(cluster_id1, cluster_id2)
                im.cluster_id = f"{cid1}|{cid2}"
            im.is_low_homology = len(homologs[seq1.id] & homologs[seq2.id]) == 0

    metadata_dicts: list[dict] = [m.to_dict() for m in metadatas]
    # Save to a msgpack file (efficient and fast)
    manifest_path: pathlib.Path = save_dir / "manifest.msgpack"
    with open(manifest_path, "wb") as f:
        msgpack.pack(metadata_dicts, f)
    print(f"Saved manifest (msgpack) to {manifest_path}")

    # Save to a json file (human-readable)
    manifest_path: pathlib.Path = save_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata_dicts, f, indent=2)
    print(f"Saved manifest (json) to {manifest_path}")
    return metadatas


def parse_args():
    parser = argparse.ArgumentParser(description="Extract test pdb ids.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
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
    """Main function to construct test set.
    See AlphaFold3 Supplementary Section 5.8 for details: Multimer and Monomer selection.
    """
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    # ======================================================================
    # Load training sequences
    # ======================================================================
    train_seqs: list[Seq] = []
    train_seq_fasta: pathlib.Path = data_dir / "rcsb-train-sequences.fasta"
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
    # Load test candidates from NPZ files
    # ======================================================================
    npz_dir: pathlib.Path = data_dir / "npz"
    npz_files: list[pathlib.Path] = sorted(npz_dir.rglob("*.npz"))
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
    interfaces: list[Interface] = []
    monomers: list[Seq] = []
    n_entries: int = 0
    for res in results:
        if res["num_tokens"] > MAX_TOKENS:
            # Skip entries exceeding max token limit
            continue
        if any(
            len(seq.sequence) > MAX_POLYMER_RESIDUES
            for seq in res["chains"]
            if seq.ctype.is_polymer
        ):
            # Skip entries with long polymer chains
            continue

        n_entries += 1
        interfaces.extend(res["interfaces"])

        if res["is_monomer"]:
            polymer_chains = [seq for seq in res["chains"] if seq.ctype.is_polymer]
            assert len(polymer_chains) == 1, (
                "Monomer entry must have exactly one polymer chain."
            )
            monomers.extend(polymer_chains)
    del results  # free up memory

    # sort interfaces and monomers
    interfaces.sort(key=lambda x: x.id)
    monomers.sort(key=lambda x: x.id)

    print(f"Total entries processed: {n_entries}")
    print(f"Total interfaces collected: {len(interfaces)}")
    print(f"Total monomers collected: {len(monomers)}")

    # Print composition:
    print("Interface type statistics:")
    n_iface_types = defaultdict(int)
    for seq1, seq2 in interfaces:
        ctypes = norm_key(seq1.ctype, seq2.ctype)
        n_iface_types[ctypes] += 1
    for ctypes in sorted(n_iface_types.keys()):
        key = f"{ctypes[0]}-{ctypes[1]}"
        print(f"  {key}: {n_iface_types[ctypes]}")
    print()
    print("Monomer type statistics:")
    n_monomer_types = defaultdict(int)
    for seq in monomers:
        n_monomer_types[seq.ctype] += 1
    for ctype in sorted(n_monomer_types.keys()):
        print(f"  {ctype}: {n_monomer_types[ctype]}")
    print()

    # ======================================================================
    # Multimer filtering
    # ======================================================================
    ccd = CCD.load(args.ccd_path)
    samples_interfaces: list[Interface] = filter_multier_interfaces(
        all_interfaces=interfaces,
        train_sequences=train_seqs,
        mmseqs=args.mmseqs,
        ccd=ccd,
    )

    # Collect PDB IDs from multimer filtering
    multimer_ids: set[str] = set()
    for seq1, seq2 in samples_interfaces:
        assert seq1.pdb_id == seq2.pdb_id, "Interface chains must belong to the same PDB."
        multimer_ids.add(seq1.pdb_id)
    print(f"Multimer PDB entries: {len(multimer_ids)}")

    # ======================================================================
    # Monomer filtering
    # ======================================================================
    sampled_monomers: list[Seq] = filter_monomers(
        all_polymers=monomers,
        train_sequences=train_seqs,
        mmseqs=args.mmseqs,
    )
    # Collect PDB IDs from monomer filtering
    monomer_ids: set[str] = set()
    for seq in sampled_monomers:
        monomer_ids.add(seq.pdb_id)
    print(f"Monomer PDB entries: {len(monomer_ids)}")

    # ======================================================================
    # Final test set
    # ======================================================================
    print("\n" + "=" * 50)
    test_ids = multimer_ids | monomer_ids
    print("Validation Set Final Summary")
    print(f"Multimer entries: {len(multimer_ids)}")
    print(f"Monomer entries: {len(monomer_ids)}")
    print(f"Total entries: {len(test_ids)}")

    # Save test set PDB IDs
    test_ids_file: pathlib.Path = data_dir / "test_ids.txt"
    with test_ids_file.open("w") as f:
        for pdb_id in sorted(test_ids):
            f.write(f"{pdb_id}\n")

    # Save metadata files for the test set
    npz_files: list[pathlib.Path] = [f for f in npz_files if f.stem in test_ids]
    save_metadata(
        npz_files=npz_files,
        save_dir=data_dir,
        train_sequences=train_seqs,
        mmseqs=args.mmseqs,
        ccd=ccd,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
