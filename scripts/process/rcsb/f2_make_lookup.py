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


* lookup.json format
```json
{
  "6oim": {
    "1": {
      "type": "protein",
      "seq_id": "rcsb_protein_000020",
      "seq_emb": {
        "path": "rcsb_protein_000020.pt",
        "residue_map": "1:250->1:250"
      },
      "struct_emb": {
        "path": "AF-P01116-F1-model_v6.pt",
        "residue_map": "1:235->11:245"
      },
      "apo": [
        {
          "source": "afdb"
          "name": "AF-P01116-F1-model_v6",
          "path": "AF-P01116-F1-model_v6.cif.gz",
          "residue_map": "1:235->11:245",
        },
        {
          "source": "esmfold"
          "name": "rcsb_protein_000020",
          "path": "rcsb_protein_000020.pdb.gz",
          "residue_map": "1:250->1:250",
        },
        {
          "source": "pdb"
          "name": "51d6-A",
          "path": "51d6-A.pdb.gz",
          "residue_map": "5:250->5:250",
        }
      ]
    },
    "2": {...}
  },
  "1a2c": {...}
}
```
"""

import argparse
import atexit
import io
import json
import multiprocessing
import pathlib
import shutil
from functools import partial
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import read_fasta

# --- Global variables for worker processes ---
_GLOBAL_POLYMER_SEQ_ID: dict = {}
_GLOBAL_AFDB_ID: dict = {}
_GLOBAL_LMDB_ENV: lmdb.Environment | None = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
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


def init_worker(seq_id_map: dict, afdb_id_map: dict, lmdb_path: pathlib.Path):
    """
    Initialize worker process with read-only shared data.
    This avoids pickling large dictionaries for every task.
    """
    global _GLOBAL_POLYMER_SEQ_ID, _GLOBAL_AFDB_ID, _GLOBAL_LMDB_ENV
    # Open a new read-only transaction for this worker
    # lock=False is safe for read-only and prevents potential locking issues in MP
    _GLOBAL_LMDB_ENV = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    _GLOBAL_POLYMER_SEQ_ID = seq_id_map
    _GLOBAL_AFDB_ID = afdb_id_map
    atexit.register(close_worker)


def close_worker():
    """Cleanup function to close LMDB environment."""
    global _GLOBAL_LMDB_ENV
    if _GLOBAL_LMDB_ENV is not None:
        _GLOBAL_LMDB_ENV.close()
        _GLOBAL_LMDB_ENV = None


def load_sequence_map(
    all_sequence_fasta: pathlib.Path,
    protein_fasta: pathlib.Path,
    dna_fasta: pathlib.Path,
    rna_fasta: pathlib.Path,
) -> dict[str, dict]:
    """Load sequence ID mapping."""
    # Load all sequences
    prot_seq_to_id: dict[str, str] = {
        seq: seq_id for seq_id, seq in read_fasta(protein_fasta)
    }
    dna_seq_to_id: dict[str, str] = {seq: seq_id for seq_id, seq in read_fasta(dna_fasta)}
    rna_seq_to_id: dict[str, str] = {seq: seq_id for seq_id, seq in read_fasta(rna_fasta)}

    seq_id_map: dict = {}

    # Load sequence to id mapping
    for key, seq in read_fasta(all_sequence_fasta):
        pdb_id, entity_id_str, ctype_str = key.split("|")
        entity_id = int(entity_id_str)
        ctype = C.ChainType[ctype_str.upper()]
        seqlen = len(seq)
        match ctype:
            case C.ChainType.PROTEIN:
                seq_id = prot_seq_to_id[seq]
            case C.ChainType.DNA:
                seq_id = dna_seq_to_id[seq]
            case C.ChainType.RNA:
                seq_id = rna_seq_to_id[seq]
            case _:
                # ligand
                continue

        # Construct residue mapping
        # Since apo structures are predicted from full sequences,
        # we assume full-length mapping here.
        seq_res = f"1:{seqlen}"
        apo_res = f"1:{seqlen}"
        res_map = f"{seq_res}->{apo_res}"

        seq_id_map[(pdb_id, entity_id)] = {
            "ctype": ctype,
            "sequence": seq,
            "seq_len": seqlen,
            "seq_id": seq_id,
            "res_map": res_map,
        }
    return seq_id_map


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


def process_batch(
    keys: list[bytes],
    apo_dir: pathlib.Path,
    split: str = "train",
):
    """
    Process a batch of LMDB keys.
    Returns the local lookup dictionary and statistics.
    """

    global _GLOBAL_LMDB_ENV, _GLOBAL_POLYMER_SEQ_ID, _GLOBAL_AFDB_ID

    env = _GLOBAL_LMDB_ENV
    assert env is not None, "LMDB environment is not initialized in worker."

    lookup: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = {
        "total_entries": 0,
        "nonprotein_chains": 0,
        "protein_chains": 0,
        "with_apo": 0,
        "without_apo": 0,
        "with_esmfold": 0,
        "without_esmfold": 0,
        "with_afdb": 0,
        "without_afdb": 0,
    }

    with env.begin() as txn:
        for key in keys:
            entry_name: str = key.decode()
            value = txn.get(key)
            # Key not found, skip
            if value is None:
                print(f"Warning: Entry {entry_name} not found in LMDB, skipping.")
                continue

            stats["total_entries"] += 1

            # === 1. Extract sequences for each entity === #
            # Load structure from npz bytes
            with io.BytesIO(value) as byte_stream:
                struct: RefStructure = RefStructure.load_npz(byte_stream)
            assert struct.id == entry_name, (
                f"Entry name mismatch: {struct.id} vs {entry_name}"
            )

            entity_dict: dict[int, tuple[C.ChainType, str]] = {}
            for chain in struct.chains:
                if chain.entity_id in entity_dict:
                    continue
                if chain.ctype.is_nonpolymer:
                    seq = "-".join(chain.get_ccd_sequence())
                else:
                    seq = chain.get_sequence(map_to_standard=True)
                entity_dict[chain.entity_id] = (chain.ctype, seq)

            # Cleanup
            del struct

            # === 2. Prepare lookup info for each entity === #
            entry_info: dict[int, dict[str, Any]] = {}
            for entity_id, (ctype, seq) in entity_dict.items():
                if ctype.is_nonpolymer:
                    stats["nonprotein_chains"] += 1
                    entity_lookup = _prepare_nonpolymer_lookup(ctype, seq)
                elif ctype.is_nucleic_acid:
                    stats["nonprotein_chains"] += 1
                    entity_lookup = _prepare_nucleic_acid_lookup(
                        ctype, entry_name, entity_id
                    )
                elif ctype.is_protein:
                    stats["protein_chains"] += 1
                    entity_lookup = _prepare_protein_lookup(
                        ctype, entry_name, entity_id, apo_dir
                    )
                    # Update apo statistics
                    if len(entity_lookup.get("apo", {})) > 0:
                        stats["with_apo"] += 1
                        has_esmfold = any(
                            apo_info["source"] == "esmfold"
                            for apo_info in entity_lookup["apo"]
                        )
                        has_afdb = any(
                            apo_info["source"] == "afdb"
                            for apo_info in entity_lookup["apo"]
                        )
                        if has_esmfold:
                            stats["with_esmfold"] += 1
                        else:
                            stats["without_esmfold"] += 1
                        if has_afdb:
                            stats["with_afdb"] += 1
                        else:
                            stats["without_afdb"] += 1

                        # Copy reference apo structures to ref_apo directory
                        if len(entity_lookup["apo"]) > 0:
                            ref_apo_info = entity_lookup["apo"][0]
                            src_path = (
                                apo_dir / ref_apo_info["source"] / ref_apo_info["path"]
                            )
                            dst_path = apo_dir / "ref_apo" / src_path.name
                            if not dst_path.exists():
                                shutil.copyfile(src_path, dst_path)

                        # For validation/test set, keep the most preferred one.
                        if split in ["val", "test"]:
                            if len(entity_lookup["apo"]) > 0:
                                entity_lookup["apo"] = entity_lookup["apo"][:1]
                    else:
                        stats["without_apo"] += 1
                else:
                    raise ValueError(f"Unknown chain type: {ctype}")

                entry_info[entity_id] = entity_lookup

            # Convert entity IDs to strings for JSON compatibility
            lookup[entry_name] = {str(k): v for k, v in entry_info.items()}

    return lookup, stats


def _prepare_nonpolymer_lookup(
    ctype: C.ChainType,
    seq: str,
) -> dict:
    """Prepare lookup data for non-polymer chains."""
    return {
        "type": ctype.name.lower(),
        "ccd": seq,
    }


def _prepare_nucleic_acid_lookup(
    ctype: C.ChainType,
    entry_name: str,
    entity_id: int,
) -> dict:
    """Prepare lookup data for nucleic acid chains."""
    global _GLOBAL_POLYMER_SEQ_ID

    # === 1. Get sequence ID === #
    # Ensure sequence ID exists
    if (entry_name, entity_id) not in _GLOBAL_POLYMER_SEQ_ID:
        raise ValueError(f"Sequence ID not found for {entry_name} entity {entity_id}")
    seq_info = _GLOBAL_POLYMER_SEQ_ID[(entry_name, entity_id)]

    # Verify chain type matches
    assert ctype == seq_info["ctype"], (
        f"Chain type mismatch for {entry_name} entity {entity_id}: "
        f"{ctype} vs {seq_info['ctype']}"
    )

    # Get sequence ID and residue mapping
    seq_id: str = seq_info["seq_id"]
    seq_res_map: str = seq_info["res_map"]

    # === 2. Prepare sequence embedding info === #
    seq_emb: dict[str, str] = {
        "path": f"{seq_id}.pt",
        "residue_map": seq_res_map,
    }
    return {
        "type": ctype.name.lower(),
        "seq_id": seq_id,
        "seq_emb": seq_emb,
    }


def _prepare_protein_lookup(
    ctype: C.ChainType,
    entry_name: str,
    entity_id: int,
    apo_dir: pathlib.Path,
) -> dict:
    """Prepare lookup data for protein chains."""
    global _GLOBAL_POLYMER_SEQ_ID, _GLOBAL_AFDB_ID

    # === 1. Get sequence ID === #
    # Ensure sequence ID exists
    if (entry_name, entity_id) not in _GLOBAL_POLYMER_SEQ_ID:
        raise ValueError(f"Sequence ID not found for {entry_name} entity {entity_id}")

    seq_info = _GLOBAL_POLYMER_SEQ_ID[(entry_name, entity_id)]

    # Verify chain type matches
    assert ctype == seq_info["ctype"], (
        f"Chain type mismatch for {entry_name} entity {entity_id}: "
        f"{ctype} vs {seq_info['ctype']}"
    )

    # Get sequence ID and residue mapping
    seq_id: str = seq_info["seq_id"]
    seq_res_map: str = seq_info["res_map"]

    # === 2. Prepare sequence embedding info === #
    seq_emb: dict[str, str] = {
        "path": f"{seq_id}.pt",
        "residue_map": seq_res_map,
    }

    # === 3. Prepare apo structure info === #
    apo_infos: list[dict[str, str]] = []

    # First, check AFDB Apo Structure
    if (entry_name, entity_id) in _GLOBAL_AFDB_ID:
        afdb_info = _GLOBAL_AFDB_ID[(entry_name, entity_id)]
        uniprot_id = afdb_info["uniprot_id"]
        uniprot_res_map = afdb_info["res_map"]

        afdb_id = f"AF-{uniprot_id}-F1-model_v6"
        afdb_path = apo_dir / "afdb" / f"{afdb_id}.pdb.gz"
        if afdb_path.exists():
            afdb_apo = {
                "source": "afdb",
                "name": afdb_id,
                "path": afdb_path.name,
                "residue_map": uniprot_res_map,
            }
            apo_infos.append(afdb_apo)

    # Then, check PDB Apo Structure
    seqlen = seq_info["seq_len"]
    esmfold_path = apo_dir / "esmfold" / f"{seqlen}/{seq_id}.pdb.gz"
    if esmfold_path.exists():
        esmfold_apo: dict[str, str] = {
            "source": "esmfold",
            "name": seq_id,
            "path": f"{seqlen}/{seq_id}.pdb.gz",
            "residue_map": seq_res_map,
        }
        apo_infos.append(esmfold_apo)

    # Finally, check PDB Apo Structure (not implemented yet)
    # TODO: Add PDB apo structure support here

    # === 4. Prepare structure embedding info === #
    if len(apo_infos) == 0:
        struct_emb = None
    else:
        # Use the most preferred apo structure for structure embedding
        ref_apo_info = apo_infos[0]
        struct_emb = {
            "path": f"{ref_apo_info['name']}.pt",
            "residue_map": ref_apo_info["residue_map"],
        }

    return {
        "type": ctype.name.lower(),
        "seq_id": seq_id,
        "seq_emb": seq_emb,
        "struct_emb": struct_emb,
        "apo": apo_infos,
    }


def main():
    """Main function to extract sequences from npz files using multiprocessing."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    lmdb_path = data_dir / "structure.lmdb"

    apo_dir = data_dir / "apo"
    ref_apo_dir = apo_dir / "ref_apo"
    ref_apo_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ID mappings...")
    seq_dir = data_dir / "sequences"
    seq_map = load_sequence_map(
        seq_dir / "all_sequences.fasta",
        seq_dir / "unique_protein_sequences.fasta",
        seq_dir / "unique_dna_sequences.fasta",
        seq_dir / "unique_rna_sequences.fasta",
    )

    afdb_map_path = seq_dir / "afdb_mapping.csv"
    afdb_map = {}
    if afdb_map_path is not None and afdb_map_path.exists():
        afdb_map = load_afdb_map(afdb_map_path)

    print("Retrieving keys from LMDB...")
    env = lmdb.open(str(lmdb_path), readonly=True, readahead=False, lock=False)
    with env.begin() as txn:
        # Collect all keys first (fast operation)
        keys = [key for key, _ in txn.cursor()]
    env.close()

    total_entries = len(keys)
    print(f"Total entries to process: {total_entries}")

    # Chunk keys for workers
    # Heuristic: Break into chunks to keep progress bar smooth but minimize overhead
    chunk_size = max(1, total_entries // (args.num_workers * 4))
    key_chunks = [keys[i : i + chunk_size] for i in range(0, total_entries, chunk_size)]

    print(f"Starting processing with {args.num_workers} workers...")

    func = partial(process_batch, apo_dir=apo_dir, split=args.split)

    # Initialize pool with shared read-only data
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(seq_map, afdb_map, lmdb_path),
    ) as pool:
        # Use imap_unordered for better responsiveness in tqdm
        results = list(
            tqdm(
                pool.imap_unordered(func, key_chunks),
                total=len(key_chunks),
                desc="Processing batches",
            )
        )

    # Aggregating results
    print("Aggregating results...")
    final_lookup = {}
    final_stats = {
        "total_entries": 0,
        "nonprotein_chains": 0,
        "protein_chains": 0,
        "with_esmfold": 0,
        "without_esmfold": 0,
        "with_afdb": 0,
        "without_afdb": 0,
        "with_apo": 0,
        "without_apo": 0,
    }
    for local_lookup, stats in results:
        final_lookup.update(local_lookup)
        for k in final_stats:
            final_stats[k] += stats[k]

    # Save lookup
    lookup_path = data_dir / "lookup.json"
    print(f"Saving lookup to {lookup_path}")
    with open(lookup_path, "w") as f:
        json.dump(final_lookup, f, indent=2)

    # Print statistics
    print("Processing complete. Statistics:")
    for k, v in final_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
