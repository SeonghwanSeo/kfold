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
        "source": "esmfold"
        "name": "rcsb_protein_000020",
      },
      {
        "source": "pdb"
        "name": "6oim_A",
        "residue_map": "5:250->5:250",
      }
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
    if (entry_name, entity_id) in afdb_id_map:
        afdb_info = afdb_id_map[(entry_name, entity_id)]
        uniprot_id = afdb_info["uniprot_id"]
        uniprot_res_map = afdb_info["res_map"]
        afdb_id = f"AF-{uniprot_id}-F1-model_v6"
        afdb_apo = {
            "source": "afdb",
            "name": afdb_id,
            "residue_map": uniprot_res_map,
        }
        apo_infos.append(afdb_apo)

    # Next, add ESMFold Apo Structure
    # ESMFold models are full-length and have a 1-to-1 residue mapping.
    seq_id = polymer_seq_id_map[(entry_name, entity_id)]
    esmfold_apo: dict[str, str] = {
        "source": "esmfold",
        "name": seq_id,
    }
    apo_infos.append(esmfold_apo)

    return apo_infos


def main():
    """Main function to process RCSB PDB entries and create lookup JSON."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"

    apo_dir = data_dir / "apo"

    print("Loading ID mappings...")
    seq_dir = data_dir / "sequences"
    SEQ_MAP = load_sequence_map(
        seq_dir / "all_sequences.fasta",
        seq_dir / "unique_protein_sequences.fasta",
    )

    afdb_map_path = seq_dir / "afdb_mapping.csv"
    if afdb_map_path is not None and afdb_map_path.exists():
        AFDB_MAP = load_afdb_map(afdb_map_path)

    prepare_protein_lookup = partial(
        _prepare_protein_lookup,
        apo_dir=apo_dir,
        seq_id_map=SEQ_MAP,
        polymer_seq_id_map=SEQ_MAP,
        afdb_id_map=AFDB_MAP,
    )

    # Load manifest
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        metadata_dicts: list[dict] = msgpack.unpack(f, raw=False)
    metadatas: list[Metadata] = [Metadata.from_dict(m) for m in metadata_dicts]

    stats: dict[str, int] = {
        "total_entries": 0,
        "protein_chains": 0,
        "with_apo": 0,
        "without_apo": 0,
        "with_esmfold": 0,
        "without_esmfold": 0,
        "with_afdb": 0,
        "without_afdb": 0,
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
            if not c_m.ctype.is_protein:
                continue

            stats["protein_chains"] += 1
            apo_dicts = prepare_protein_lookup(entry_name, entity_id)
            # Update apo statistics
            if len(apo_dicts) > 0:
                stats["with_apo"] += 1
                has_esmfold = any(
                    apo_info["source"] == "esmfold" for apo_info in apo_dicts
                )
                has_afdb = any(apo_info["source"] == "afdb" for apo_info in apo_dicts)
                if has_esmfold:
                    stats["with_esmfold"] += 1
                else:
                    stats["without_esmfold"] += 1
                if has_afdb:
                    stats["with_afdb"] += 1
                else:
                    stats["without_afdb"] += 1

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
