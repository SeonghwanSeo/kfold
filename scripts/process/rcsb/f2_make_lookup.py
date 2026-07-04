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


* apo_lookup.json format
```json
{
  "6oim": {
    "1": [
      {
        "source": "afdb"
        "name": "AF-P01116-F1-model_v6",
        "residue_map": "1:235->11:245",
      },
      {
        "source": "atlasfold"
        "name": "rcsb_protein_000020",
      },
    ],
    "2": [...]
  },
  "1a2c": {...}
}
```
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
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        choices=["train", "val", "test"],
        help="Data split to process (train/val/test).",
    )
    args = parser.parse_args()
    return args


def load_polymer_sequence_map(
    all_sequence_fasta: pathlib.Path,
    unique_sequence_fasta: pathlib.Path,
    ctype_name: str,
) -> dict[tuple[str, int], str]:
    """Load RCSB entity to unique polymer sequence ID mapping."""
    seq_to_id: dict[str, str] = {}
    entity_to_id: dict[tuple[str, int], tuple[str, str]] = {}
    for seq_id, seq in read_fasta(unique_sequence_fasta):
        seq_to_id[seq] = seq_id
        pdb_id, sep, entity_id_str = seq_id.rpartition("_")
        if sep and entity_id_str.isdigit():
            entity_to_id[(pdb_id.lower(), int(entity_id_str))] = (seq_id, seq)

    sequence_id_map: dict[tuple[str, int], str] = {}
    for key, seq in read_fasta(all_sequence_fasta):
        pdb_id, entity_id_str, ctype_str = key.split("|")
        entity_id = int(entity_id_str)
        if ctype_str.lower() == ctype_name:
            entity_match = entity_to_id.get((pdb_id, entity_id))
            if entity_match is not None:
                seq_id, expected_seq = entity_match
                if expected_seq != seq:
                    raise ValueError(
                        f"Sequence mismatch for {pdb_id}_{entity_id} in "
                        f"{unique_sequence_fasta}."
                    )
            else:
                seq_id = seq_to_id[seq]
            sequence_id_map[(pdb_id, entity_id)] = seq_id
    return sequence_id_map


def load_afdb_map(path: pathlib.Path) -> dict[str, dict]:
    """Load AFDB ID mapping from CSV file."""
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


def entity_apo_file_exists(
    apo_dir: pathlib.Path,
    entry_name: str,
    entity_id: int,
    filename: str,
) -> bool:
    """Check current sharded entity archive if it exists.

    If the dataset still uses the legacy apo/{source}/ layout, the shard
    directory will not exist and lookup generation keeps the legacy behavior.
    """
    shard_dir = apo_dir / entry_name[1:3]
    if not shard_dir.exists():
        return True
    return (shard_dir / entry_name / str(entity_id) / filename).exists()


def _prepare_protein_lookup(
    entry_name: str,
    entity_id: int,
    apo_dir: pathlib.Path,
    seq_id_map: dict[tuple[str, int], str],
    polymer_seq_id_map: dict,
    afdb_id_map: dict,
) -> list[dict[str, str]]:
    """Prepare lookup data for protein chains."""
    # Ensure sequence ID exists
    if (entry_name, entity_id) not in polymer_seq_id_map:
        raise ValueError(f"Sequence ID not found for {entry_name} entity {entity_id}")

    apo_infos: list[dict[str, str]] = []

    # First, add AFDB Apo Structure
    if (entry_name, entity_id) in afdb_id_map and entity_apo_file_exists(
        apo_dir, entry_name, entity_id, "af2.pdb"
    ):
        afdb_info = afdb_id_map[(entry_name, entity_id)]
        uniprot_id = afdb_info["uniprot_id"]
        uniprot_res_map = afdb_info["res_map"]
        afdb_id = f"AF-{uniprot_id}-F1-model_v6"
        afdb_apo = {
            "source": "afdb",
            "name": afdb_id,
            "residue_map": uniprot_res_map,
            "chain_type": "protein",
        }
        apo_infos.append(afdb_apo)

    # Next, add ESMFold Apo Structure
    # ESMFold models are full-length and have a 1-to-1 residue mapping.
    seq_id = polymer_seq_id_map[(entry_name, entity_id)]
    if entity_apo_file_exists(apo_dir, entry_name, entity_id, "esmfold2.pdb"):
        esmfold_apo: dict[str, str] = {
            "source": "esmfold",
            "name": seq_id,
            "chain_type": "protein",
        }
        apo_infos.append(esmfold_apo)

    if entity_apo_file_exists(apo_dir, entry_name, entity_id, "protein_apo.pdb.zst"):
        protein_apo: dict[str, str] = {
            "source": "protein_apo",
            "name": seq_id,
            "chain_type": "protein",
        }
        apo_infos.append(protein_apo)

    return apo_infos


def _prepare_dna_lookup(
    entry_name: str,
    entity_id: int,
    apo_dir: pathlib.Path,
    polymer_seq_id_map: dict[tuple[str, int], str],
) -> list[dict[str, str]]:
    """Prepare lookup data for deterministic DNA helix apo structures."""
    if (entry_name, entity_id) not in polymer_seq_id_map:
        raise ValueError(f"DNA sequence ID not found for {entry_name} entity {entity_id}")

    seq_id = polymer_seq_id_map[(entry_name, entity_id)]
    if not entity_apo_file_exists(apo_dir, entry_name, entity_id, "helix.pdb.zst"):
        return []
    return [
        {
            "source": "dna",
            "name": seq_id,
            "model": "single_helix",
            "chain_type": "dna",
        }
    ]


def _prepare_rna_lookup(
    entry_name: str,
    entity_id: int,
    apo_dir: pathlib.Path,
    polymer_seq_id_map: dict[tuple[str, int], str],
) -> list[dict[str, str]]:
    """Prepare lookup data for RNA apo sample structures."""
    if (entry_name, entity_id) not in polymer_seq_id_map:
        raise ValueError(f"RNA sequence ID not found for {entry_name} entity {entity_id}")

    seq_id = polymer_seq_id_map[(entry_name, entity_id)]
    apo_infos: list[dict[str, str]] = []
    for sample_id in range(5):
        filename = f"rna_apo_sample_{sample_id}.cif.zst"
        if not entity_apo_file_exists(apo_dir, entry_name, entity_id, filename):
            continue
        apo_infos.append(
            {
                "source": "rna_apo",
                "name": f"{seq_id}_sample_{sample_id}",
                "sample_id": str(sample_id),
                "chain_type": "rna",
            }
        )
    return apo_infos


def main():
    """Main function to process RCSB PDB entries and create lookup JSON."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"

    apo_dir = data_dir / "apo"

    print("Loading ID mappings...")
    seq_dir = data_dir / "sequences"
    SEQ_MAP = load_polymer_sequence_map(
        seq_dir / "all_sequences.fasta",
        seq_dir / "unique_protein_sequences.fasta",
        "protein",
    )
    DNA_SEQ_MAP = load_polymer_sequence_map(
        seq_dir / "all_sequences.fasta",
        seq_dir / "unique_dna_sequences.fasta",
        "dna",
    )
    RNA_SEQ_MAP = load_polymer_sequence_map(
        seq_dir / "all_sequences.fasta",
        seq_dir / "unique_rna_sequences.fasta",
        "rna",
    )

    afdb_map_path = seq_dir / "afdb_mapping.csv"
    if not afdb_map_path.exists():
        afdb_map_path = seq_dir / "afdb_mapping.csv.zst"
    AFDB_MAP = {}
    if afdb_map_path is not None and afdb_map_path.exists():
        AFDB_MAP = load_afdb_map(afdb_map_path)

    prepare_protein_lookup = partial(
        _prepare_protein_lookup,
        apo_dir=apo_dir,
        seq_id_map=SEQ_MAP,
        polymer_seq_id_map=SEQ_MAP,
        afdb_id_map=AFDB_MAP,
    )
    prepare_dna_lookup = partial(
        _prepare_dna_lookup,
        apo_dir=apo_dir,
        polymer_seq_id_map=DNA_SEQ_MAP,
    )
    prepare_rna_lookup = partial(
        _prepare_rna_lookup,
        apo_dir=apo_dir,
        polymer_seq_id_map=RNA_SEQ_MAP,
    )

    # Load manifest
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        metadata_dicts: list[dict] = msgpack.unpack(f, raw=False)
    metadatas: list[Metadata] = [Metadata.from_dict(m) for m in metadata_dicts]

    stats: dict[str, int] = {
        "total_entries": 0,
        "protein_chains": 0,
        "rna_chains": 0,
        "dna_chains": 0,
        "with_apo": 0,
        "without_apo": 0,
        "with_esmfold": 0,
        "without_esmfold": 0,
        "with_afdb": 0,
        "without_afdb": 0,
        "with_protein_apo": 0,
        "with_rna_apo": 0,
        "with_dna_helix": 0,
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
            if not (c_m.ctype.is_protein or c_m.ctype.is_rna or c_m.ctype.is_dna):
                continue

            if c_m.ctype.is_protein:
                stats["protein_chains"] += 1
                apo_dicts = prepare_protein_lookup(entry_name, entity_id)
            elif c_m.ctype.is_rna:
                stats["rna_chains"] += 1
                apo_dicts = prepare_rna_lookup(entry_name, entity_id)
            else:
                stats["dna_chains"] += 1
                apo_dicts = prepare_dna_lookup(entry_name, entity_id)
            # Update apo statistics
            if len(apo_dicts) > 0:
                stats["with_apo"] += 1
                if c_m.ctype.is_protein:
                    has_esmfold = any(
                        apo_info["source"] == "esmfold" for apo_info in apo_dicts
                    )
                    has_afdb = any(apo_info["source"] == "afdb" for apo_info in apo_dicts)
                    has_protein_apo = any(
                        apo_info["source"] == "protein_apo" for apo_info in apo_dicts
                    )
                    if has_esmfold:
                        stats["with_esmfold"] += 1
                    else:
                        stats["without_esmfold"] += 1
                    if has_afdb:
                        stats["with_afdb"] += 1
                    else:
                        stats["without_afdb"] += 1
                    if has_protein_apo:
                        stats["with_protein_apo"] += 1
                elif any(apo_info["source"] == "rna_apo" for apo_info in apo_dicts):
                    stats["with_rna_apo"] += 1
                elif any(apo_info["source"] == "dna" for apo_info in apo_dicts):
                    stats["with_dna_helix"] += 1

                # For validation/test set, keep the most preferred one.
                if args.split in ["val", "test"]:
                    if len(apo_dicts) > 0:
                        apo_dicts = [apo_dicts[0]]
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
