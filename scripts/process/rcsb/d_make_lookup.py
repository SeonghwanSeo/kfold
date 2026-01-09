"""
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
          "name": "rcsb_protein_000020",
          "path": "rcsb_protein_000020.pdb.gz",
          "residue_map": "1:250->1:250",
          "source": "esmfold"
        },
        {
          "name": "AF-P01116-F1-model_v6",
          "path": "AF-P01116-F1-model_v6.cif.gz",
          "residue_map": "1:235->11:245",
          "source": "afdb"
        },
        {
          "name": "51d6-A",
          "path": "51d6-A.pdb.gz",
          "residue_map": "5:250->5:250",
          "source": "pdb"
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

# --- Thresholds for AFDB mapping ---
# Minimum RCSB sequence length to consider (use ESMFold only)
AFDB_LENGTH_THRESHOLD = 16
# Minimum overlap ratio on RCSB side
AFDB_RCSB_OVERLAP_THRESHOLD = 0.8
# Minimum overlap ratio on Uniprot side
AFDB_UNIPROT_OVERLAP_THRESHOLD = 0.5

# --- Global variables for worker processes ---
_GLOBAL_SEQ_ID: dict = {}
_GLOBAL_AFDB_ID: dict = {}

_GLOBAL_LMDB_ENV: lmdb.Environment = None


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
    args = parser.parse_args()
    return args


def init_worker(seq_id_map: dict, afdb_id_map: dict, lmdb_path: pathlib.Path):
    """
    Initialize worker process with read-only shared data.
    This avoids pickling large dictionaries for every task.
    """
    global _GLOBAL_SEQ_ID, _GLOBAL_AFDB_ID, _GLOBAL_LMDB_ENV
    # Open a new read-only transaction for this worker
    # lock=False is safe for read-only and prevents potential locking issues in MP
    _GLOBAL_LMDB_ENV = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    _GLOBAL_SEQ_ID = seq_id_map
    _GLOBAL_AFDB_ID = afdb_id_map


def process_batch(keys: list[bytes], apo_dir: pathlib.Path):
    """
    Process a batch of LMDB keys.
    Returns the local lookup dictionary and statistics.
    """

    # Create reference apo directory paths
    ref_apo_dir = apo_dir / "ref_apo"
    ref_apo_dir.mkdir(parents=True, exist_ok=True)

    global _GLOBAL_LMDB_ENV, _GLOBAL_SEQ_ID, _GLOBAL_AFDB_ID

    env = _GLOBAL_LMDB_ENV

    lookup: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = {
        "total_entries": 0,
        "nonprotein_chains": 0,
        "protein_chains": 0,
        "with_esmfold": 0,
        "without_esmfold": 0,
        "with_afdb": 0,
        "without_afdb": 0,
    }
    with env.begin() as txn:
        for key in keys:
            value = txn.get(key)
            if value is None:
                continue

            stats["total_entries"] += 1

            # Deserialize
            with io.BytesIO(value) as byte_stream:
                ref_structure: RefStructure = RefStructure.load_npz(byte_stream)

            metadata = ref_structure.metadata
            entry_name: str = metadata.id

            # Process Chains
            entity_dict: dict[int, tuple[C.ChainType, str]] = {}
            for chain in ref_structure.chains:
                entity_id = chain.entity_id
                if entity_id in entity_dict:
                    continue
                if chain.ctype.is_nonpolymer:
                    seq = ":".join(chain.get_ccd_sequence())
                else:
                    seq = chain.get_sequence(map_to_standard=True)
                entity_dict[entity_id] = (chain.ctype, seq)

            # free memory explicitly for the object
            del ref_structure

            # Build Lookup Entry
            entry_info: dict[int, dict] = {}
            for entity_id, (ctype, _) in entity_dict.items():
                entity_lookup_data: dict[str, Any] = {"type": ctype.name.lower()}

                # Match Sequence Embedding
                if ctype.is_nonpolymer:
                    entry_info[entity_id] = entity_lookup_data
                    continue

                # Get Sequence ID
                if (entry_name, entity_id) not in _GLOBAL_SEQ_ID:
                    raise ValueError(
                        f"Sequence ID not found for {entry_name} entity {entity_id}"
                    )

                seq_info = _GLOBAL_SEQ_ID[(entry_name, entity_id)]
                assert ctype == seq_info["ctype"], (
                    f"Chain type mismatch for {entry_name} entity {entity_id}: "
                    f"{ctype} vs {seq_info['ctype']}"
                )

                # Sequence Embedding
                seq_id = seq_info["seq_id"]
                seq_res_map = seq_info["res_map"]
                seq_emb = {
                    "path": f"{seq_id}.pt",
                    "residue_map": seq_res_map,
                }

                if ctype.is_protein:
                    stats["protein_chains"] += 1
                    # ESMFold Apo Structure
                    apo_infos = []
                    esm_apo = {
                        "name": seq_id,
                        "path": f"{seq_id}.pdb.gz",
                        "residue_map": seq_res_map,
                        "source": "esmfold",
                    }
                    esmfold_struct = apo_dir / "esmfold" / esm_apo["path"]
                    if esmfold_struct.exists():
                        stats["with_esmfold"] += 1
                        apo_infos.append(esm_apo)
                    else:
                        stats["without_esmfold"] += 1

                    # Try to Match AFDB Structure
                    if (entry_name, entity_id) in _GLOBAL_AFDB_ID:
                        afdb_info = _GLOBAL_AFDB_ID[(entry_name, entity_id)]
                        uniprot_id = afdb_info["uniprot_id"]
                        uniprot_res_map = afdb_info["res_map"]

                        afdb_id = f"AF-{uniprot_id}-F1-model_v6"
                        afdb_apo = {
                            "name": afdb_id,
                            "path": f"{afdb_id}.pdb.gz",
                            "residue_map": uniprot_res_map,
                            "source": "afdb",
                        }

                        afdb_struct = apo_dir / "afdb" / afdb_apo["path"]
                        if afdb_struct.exists():
                            stats["with_afdb"] += 1
                            apo_infos.append(afdb_apo)
                        else:
                            stats["without_afdb"] += 1
                    else:
                        stats["without_afdb"] += 1

                    # Determine `ref_apo` source, which is the primary structure
                    # for structure embedding.
                    if len(apo_infos) == 0:
                        struct_emb = None
                    else:
                        if len(apo_infos) > 1:
                            # Prefer AFDB structure if available
                            ref_apo = apo_infos[1]
                        else:
                            ref_apo = apo_infos[0]
                        # Copy ref apo structure to ref_apo directory
                        ref_apo_src = apo_dir / ref_apo["source"] / ref_apo["path"]
                        ref_apo_dst = ref_apo_dir / ref_apo["path"]
                        if not ref_apo_dst.exists():
                            shutil.copyfile(ref_apo_src, ref_apo_dst)

                        struct_emb = {
                            "path": f"{ref_apo['name']}.pt",
                            "residue_map": ref_apo["residue_map"],
                        }
                else:
                    # Non-protein chains do not have structure embeddings
                    stats["nonprotein_chains"] += 1
                    apo_infos = []
                    struct_emb = None

                entity_lookup_data = {
                    "seq_id": seq_id,
                    "seq_emb": seq_emb,
                    "struct_emb": struct_emb,
                    "apo": apo_infos,
                }
                entry_info[entity_id] = entity_lookup_data

            # Convert entity IDs to strings for JSON compatibility
            lookup[entry_name] = {str(k): v for k, v in entry_info.items()}

    return lookup, stats


def load_sequence_map(path: pathlib.Path) -> dict[str, dict]:
    seq_id_map: dict = {}
    assert path.suffix == ".csv"
    df = pd.read_csv(path)
    for row in df.itertuples():
        pdb_id = row.pdb_id
        entity_id = row.entity_id
        ctype = row.type
        seqlen = row.length
        seq_id = row.seq_id

        # Construct residue mapping
        # Since apo structures are predicted from full sequences,
        # we assume full-length mapping here.
        seq_res = f"1:{seqlen}"
        apo_res = f"1:{seqlen}"
        res_map = f"{seq_res}->{apo_res}"

        seq_id_map[(pdb_id, entity_id)] = {
            "ctype": C.ChainType[ctype.upper()],
            "seq_id": seq_id,
            "res_map": res_map,
        }
    return seq_id_map


def load_afdb_map(path: pathlib.Path) -> dict[str, dict]:
    assert path.suffix == ".csv"
    afdb_id_map: dict = {}
    df = pd.read_csv(path)
    for row in df.itertuples():
        pdb_id = row.pdb_id
        entity_id = row.entity_id
        seq_len = row.seq_len
        seq_st = row.seq_st
        seq_end = row.seq_end

        # uniprot id
        uniprot_id = row.uniprot_id
        uniprot_len = row.uniprot_len
        uniprot_st = row.uniprot_st
        uniprot_end = row.uniprot_end

        if seq_len < AFDB_LENGTH_THRESHOLD:
            # Too short to consider
            continue

        if not (seq_end - seq_st) == (uniprot_end - uniprot_st):
            # There are some typos in the pdb to uniprot mapping file.
            print(f"Length mismatch in AFDB mapping: {row}")
            continue

        res_map = f"{seq_st}:{seq_end}->{uniprot_st}:{uniprot_end}"

        # Compute overlap ratio
        seq_overlap = (seq_end - seq_st + 1) / seq_len
        uniprot_overlap = (uniprot_end - uniprot_st + 1) / uniprot_len

        if (
            seq_overlap >= AFDB_RCSB_OVERLAP_THRESHOLD
            and uniprot_overlap >= AFDB_UNIPROT_OVERLAP_THRESHOLD
        ):
            afdb_id_map[(pdb_id, entity_id)] = {
                "uniprot_id": uniprot_id,
                "source": "afdb",
                "res_map": res_map,
            }
    return afdb_id_map


def main():
    """Main function to extract sequences from npz files using multiprocessing."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    lmdb_path = data_dir / "structure.lmdb"

    print("Loading ID mappings...")
    seq_map_path = data_dir / "sequences" / "sequence_id.csv"
    seq_map = load_sequence_map(seq_map_path)

    afdb_map_path = data_dir / "sequences" / "afdb_mapping.csv"
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

    func = partial(process_batch, apo_dir=data_dir / "apo")

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
