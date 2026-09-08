"""
Create a lookup JSON file mapping RCSB PDB entries and their chains
to apo structures, sequence embedding, and structure embedding.

# Apo structure sources
The order of apo structures in the list indicates preference.
AFDB > ESMFold > PDB

TODO: PDB source is not implemented yet.

# Structure embedding selection
The most preferred apo structure is called the `ref_apo` structure,
which is used for structure embedding.

# Rules for different splits
Training set: Consider all available apo structures.
Validation/Test set: Consider only the most preferred apo structure.
"""

import argparse
import json
import pathlib
from functools import partial

import msgpack
import pandas as pd
from tqdm import tqdm

from kfold.data.types.metadata import Metadata
from kfold.data.utils.io.fasta import read_fasta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    args = parser.parse_args()
    return args


def load_sequence_map(
    all_sequence_fasta: pathlib.Path,
    protein_fasta: pathlib.Path,
) -> dict[tuple[str, int], str]:
    """Load sequence ID mapping."""
    # Load all sequences
    prot_seq_to_id: dict[str, str] = {
        seq: seq_id for seq_id, seq in read_fasta(protein_fasta)
    }
    protein_id_map: dict[tuple[str, int], str] = {}
    # Load sequence to id mapping
    for key, seq in read_fasta(all_sequence_fasta):
        pdb_id, entity_id_str, ctype_str = key.split("|")
        entity_id = int(entity_id_str)
        if ctype_str.lower() == "protein":
            seq_id = prot_seq_to_id[seq]
            protein_id_map[(pdb_id, entity_id)] = seq_id
    return protein_id_map


def load_afdb_map(path: pathlib.Path) -> dict[str, dict]:
    """Load AFDB ID mapping from CSV file."""
    assert path.suffix == ".csv"
    afdb_id_map: dict = {}
    df = pd.read_csv(path)
    df_dict = df.set_index(["pdb_id", "entity_id"]).to_dict("index")
    for (pdb_id, entity_id), row in df_dict.items():
        seq_st = row["seq_st"]
        seq_end = row["seq_end"]

        # uniprot id
        uniprot_id = row["uniprot_id"]
        uniprot_st = row["uniprot_st"]
        uniprot_end = row["uniprot_end"]

        if not (seq_end - seq_st) == (uniprot_end - uniprot_st):
            # There are some typos in the pdb to uniprot mapping file.
            print(f"Length mismatch in AFDB mapping: {row}")
            continue

        res_map = f"{seq_st}:{seq_end}->{uniprot_st}:{uniprot_end}"

        # Compute overlap ratio
        afdb_id_map[(pdb_id, entity_id)] = {
            "uniprot_id": uniprot_id,
            "source": "afdb",
            "res_map": res_map,
        }
    return afdb_id_map


def load_rcsb_apo_mapping(path: pathlib.Path) -> dict[str, dict[str, list[dict]]]:
    with path.open("rb") as f:
        return msgpack.unpack(f, raw=False, strict_map_key=False)


def _prepare_protein_lookup(
    entry_name: str,
    entity_id: int,
    polymer_seq_id_map: dict,
) -> list[dict[str, str]]:
    """Prepare lookup data for protein chains."""
    # Ensure sequence ID exists
    if (entry_name, entity_id) not in polymer_seq_id_map:
        raise ValueError(f"Sequence ID not found for {entry_name} entity {entity_id}")
    apo_infos: list[dict[str, str]] = []
    # ESMFold models are full-length and have a 1-to-1 residue mapping.
    seq_id = polymer_seq_id_map[(entry_name, entity_id)]
    esmfold_apo: dict[str, str] = {
        "source": "esmfold",
        "name": seq_id,
        "chain_type": "protein",
    }
    apo_infos.append(esmfold_apo)

    return apo_infos


def main():
    """Main function to process RCSB PDB entries and create lookup JSON."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb"

    print("Loading ID mappings...")
    seq_dir = data_dir / "sequences"
    rcsb_apo_mapping_path = seq_dir / "rcsb_apo_mapping.msgpack"
    if rcsb_apo_mapping_path.exists():
        RCSB_APO_MAPPING = load_rcsb_apo_mapping(rcsb_apo_mapping_path)
        SEQ_MAP = {}
        prepare_protein_lookup = None
        print(f"Using RCSB apo mapping: {rcsb_apo_mapping_path}")
    else:
        RCSB_APO_MAPPING = {}
        SEQ_MAP = load_sequence_map(
            seq_dir / "all_sequences.fasta",
            seq_dir / "uniq_sequences.fasta",
        )
        prepare_protein_lookup = partial(
            _prepare_protein_lookup,
            polymer_seq_id_map=SEQ_MAP,
        )

    # Load manifest
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        metadata_dicts: list[dict] = msgpack.unpack(f, raw=False)
    metadatas: list[Metadata] = [Metadata.from_dict(m) for m in metadata_dicts]

    stats: dict[str, int] = {
        "total_entries": 0,
        "polymer_chains": 0,
        "protein_chains": 0,
        "dna_chains": 0,
        "rna_chains": 0,
        "with_apo": 0,
        "without_apo": 0,
        "with_esmfold": 0,
        "with_prot_sampler": 0,
    }
    all_lookup: dict[str, dict[str, list[dict[str, str]]]] = {}
    for m in tqdm(metadatas, desc="Processing entries"):
        entry_name = m.id
        stats["total_entries"] += 1
        # === 2. Prepare lookup info for each entity === #
        visited_entities: set[int] = set()
        entry_info: dict[int, list[dict[str, str]]] = {}
        for c_m in m.chains:
            entity_id = c_m.entity_id
            if entity_id in visited_entities:
                continue
            visited_entities.add(entity_id)
            if not (c_m.ctype.is_protein or c_m.ctype.is_nucleic_acid):
                continue

            stats["polymer_chains"] += 1
            if c_m.ctype.is_protein:
                stats["protein_chains"] += 1
            elif c_m.ctype.is_dna:
                stats["dna_chains"] += 1
            elif c_m.ctype.is_rna:
                stats["rna_chains"] += 1

            if RCSB_APO_MAPPING:
                apo_dicts = RCSB_APO_MAPPING.get(entry_name, {}).get(str(entity_id), [])
            else:
                if not c_m.ctype.is_protein:
                    continue
                assert prepare_protein_lookup is not None
                apo_dicts = prepare_protein_lookup(entry_name, entity_id)
            # Update apo statistics
            if len(apo_dicts) > 0:
                stats["with_apo"] += 1
                has_esmfold = any(
                    apo_info["source"] == "esmfold" for apo_info in apo_dicts
                )
                if has_esmfold:
                    stats["with_esmfold"] += 1
                if any(
                    apo_info["source"].startswith("prot_sampler")
                    for apo_info in apo_dicts
                ):
                    stats["with_prot_sampler"] += 1
            else:
                stats["without_apo"] += 1

            entry_info[entity_id] = apo_dicts

        # Convert entity IDs to strings for JSON compatibility
        all_lookup[entry_name] = {str(k): v for k, v in entry_info.items()}

    # Print statistics
    print("Processing complete. Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save lookup
    lookup_path = data_dir / "apo_lookup.json"
    print(f"Saving lookup to {lookup_path}")
    with open(lookup_path, "w") as f:
        json.dump(all_lookup, f, indent=2)

    lookup_path = data_dir / "apo_lookup.msgpack"
    print(f"Saving lookup to {lookup_path} (msgpack format)")
    with open(lookup_path, "wb") as f:
        msgpack.pack(all_lookup, f)


if __name__ == "__main__":
    main()
