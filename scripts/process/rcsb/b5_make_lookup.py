"""
* lookup.json format
```json
{
  "6oim": {
    "1": {
      "type": "protein",
      "seq_emb": {
        "path": "uniq_protein_000020.pt",
        "residue_map": "1:250->1:250"
      },
      "struct_emb": {
        "path": "AF-P01116-F1-model_v6.pt",
        "residue_map": "1:235->11:245"
      },
      "apo": [
        {
          "name": "uniq_protein_000020-esmfold",
          "path": "uniq_protein_000020-esmfold.pdb.gz",
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
from typing import Any

import lmdb
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import RefStructure

# --- Global variables for worker processes ---
_GLOBAL_SEQ_TO_ID: dict = {}
_GLOBAL_STRUCT_TO_ID: dict = {}
_GLOBAL_LMDB_PATH: pathlib.Path | None = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--seq_id_path",
        type=pathlib.Path,
        help="Path to the file containing sequence IDs.",
    )
    parser.add_argument(
        "--struct_id_path",
        type=pathlib.Path,
        help="Path to the file containing structure IDs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
    )
    args = parser.parse_args()
    return args


def init_worker(seq_to_id: dict, struct_to_id: dict, lmdb_path: pathlib.Path):
    """
    Initialize worker process with read-only shared data.
    This avoids pickling large dictionaries for every task.
    """
    global _GLOBAL_SEQ_TO_ID, _GLOBAL_STRUCT_TO_ID, _GLOBAL_LMDB_PATH
    _GLOBAL_SEQ_TO_ID = seq_to_id
    _GLOBAL_STRUCT_TO_ID = struct_to_id
    _GLOBAL_LMDB_PATH = lmdb_path


def process_batch(keys: list[bytes]) -> tuple[dict, dict]:
    """
    Process a batch of LMDB keys.
    Returns the local lookup dictionary and statistics.
    """
    global _GLOBAL_SEQ_TO_ID, _GLOBAL_STRUCT_TO_ID, _GLOBAL_LMDB_PATH

    # Open a new read-only transaction for this worker
    # lock=False is safe for read-only and prevents potential locking issues in MP
    env = lmdb.open(
        str(_GLOBAL_LMDB_PATH),
        map_size=1 * 1024 * 1024 * 1024,  # 1 GB
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    local_lookup: dict[str, dict[str, Any]] = {}
    stats = {
        "seq_success": 0,
        "seq_fail": 0,
        "struct_success": 0,
        "struct_fail": 0,
    }

    with env.begin() as txn:
        for key in keys:
            value = txn.get(key)
            if value is None:
                continue

            # Deserialize
            with io.BytesIO(value) as byte_stream:
                ref_structure: RefStructure = RefStructure.load_npz(byte_stream)

            metadata = ref_structure.metadata
            entry_name: str = metadata.id

            # Process Chains
            entity_dict: dict[int, tuple[str, str]] = {}
            for chain in ref_structure.chains:
                entity_id = chain.entity_id
                if entity_id in entity_dict:
                    continue

                if chain.ctype.is_nonpolymer:
                    seq = ":".join(chain.get_ccd_sequence())
                    entity_dict[entity_id] = (seq, chain.ctype.name.lower())
                else:
                    sequence = chain.get_sequence()
                    standard_set = set()
                    unk = "X"

                    if chain.ctype.is_protein:
                        standard_set = set(C.residue.PROTEIN_AMINO_ACIDS)
                        unk = "X"
                        sequence = (
                            sequence.replace("B", "D").replace("Z", "E").replace("U", "C")
                        )
                    elif chain.ctype.is_rna:
                        standard_set = set(C.residue.RNA_BASES)
                        unk = "N"
                    elif chain.ctype.is_dna:
                        standard_set = set(C.residue.DNA_BASES)
                        unk = "N"

                    if standard_set:
                        sequence = "".join(
                            [v if v in standard_set else unk for v in sequence]
                        )

                    entity_dict[entity_id] = (sequence, chain.ctype.name.lower())

            # free memory explicitly for the object
            del ref_structure

            # Build Lookup Entry
            entry_lookup: dict[int, dict] = {}
            for entity_id, (seq, ctype) in entity_dict.items():
                entity_lookup_data: dict[str, Any] = {"type": ctype}

                # Match Sequence Embedding
                if (ctype, seq) in _GLOBAL_SEQ_TO_ID:
                    seq_id, seq_res_map = _GLOBAL_SEQ_TO_ID[(ctype, seq)]
                    stats["seq_success"] += 1
                    seq_emb = {
                        "path": f"{seq_id}.pt",
                        "residue_map": seq_res_map,
                    }
                    entity_lookup_data["seq_emb"] = seq_emb
                else:
                    if ctype in ["protein", "rna", "dna"]:
                        # Log only on failures to avoid clutter
                        stats["seq_fail"] += 1

                # Match Structure Embedding
                if (ctype, seq) in _GLOBAL_STRUCT_TO_ID:
                    stats["struct_success"] += 1
                    struct_id, struct_source, struct_res_map = _GLOBAL_STRUCT_TO_ID[
                        (ctype, seq)
                    ]
                    struct_emb = {
                        "path": f"{struct_id}.pt",
                        "residue_map": struct_res_map,
                    }
                    entity_lookup_data["struct_emb"] = struct_emb
                    apo_info = {
                        "name": struct_id,
                        "path": f"{struct_id}.pdb.gz",
                        "residue_map": struct_res_map,
                        "source": struct_source,
                    }
                    entity_lookup_data["apo"] = [apo_info]
                else:
                    if ctype == "protein":
                        stats["struct_fail"] += 1

                entry_lookup[entity_id] = entity_lookup_data

            # Convert entity IDs to strings for JSON compatibility
            local_lookup[entry_name] = {str(k): v for k, v in entry_lookup.items()}

    env.close()
    return local_lookup, stats


def load_sequence_ids(path: pathlib.Path) -> dict:
    seq_to_id = {}
    with open(path) as f:
        assert path.suffix == ".fasta"
        lines = f.readlines()
        assert len(lines) % 2 == 0, "Fasta file should have even number of lines."
        for i in range(0, len(lines), 2):
            header = lines[i].strip()
            sequence = lines[i + 1].strip()
            # Parse header
            key = header[1:]
            if "_protein_" in key:
                ctype = "protein"
            elif "_rna_" in key:
                ctype = "rna"
            elif "_dna_" in key:
                ctype = "dna"
            else:
                continue  # Skip unknown types

            res_map = f"1:{len(sequence)}->1:{len(sequence)}"
            seq_to_id[(ctype, sequence)] = (key, res_map)
    return seq_to_id


def load_structure_ids(path: pathlib.Path) -> dict:
    struct_to_id = {}
    with open(path) as f:
        assert path.suffix == ".csv"
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            seq, length, source, key, apo_res, res = parts
            res, apo_res = res.replace("-", ":"), apo_res.replace("-", ":")
            res_map = f"{res}->{apo_res}"
            struct_to_id[("protein", seq)] = (key, source, res_map)
    return struct_to_id


def main():
    """Main function to extract sequences from npz files using multiprocessing."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir
    lmdb_path = data_dir / "structure.lmdb"

    print("Loading ID mappings...")
    seq_to_id = load_sequence_ids(args.seq_id_path)
    struct_to_id = load_structure_ids(args.struct_id_path)

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

    final_lookup = {}
    global_stats = {
        "seq_success": 0,
        "seq_fail": 0,
        "struct_success": 0,
        "struct_fail": 0,
    }

    # Initialize pool with shared read-only data
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(seq_to_id, struct_to_id, lmdb_path),
    ) as pool:
        # Use imap_unordered for better responsiveness in tqdm
        results = list(
            tqdm(
                pool.imap_unordered(process_batch, key_chunks),
                total=len(key_chunks),
                desc="Processing batches",
            )
        )

    # Aggregating results
    print("Aggregating results...")
    for local_lookup, local_stats in results:
        final_lookup.update(local_lookup)
        for k, v in local_stats.items():
            global_stats[k] += v

    print(
        f"Sequence embedding: {global_stats['seq_success']} found, "
        f"{global_stats['seq_fail']} not found."
    )
    print(
        f"Structure embedding: {global_stats['struct_success']} found, "
        f"{global_stats['struct_fail']} not found."
    )

    # Save lookup
    lookup_path = data_dir / "lookup.json"
    print(f"Saving lookup to {lookup_path}")
    with open(lookup_path, "w") as f:
        json.dump(final_lookup, f, indent=2)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork", force=True)  # Faster on Linux/MacOS
    except RuntimeError:
        pass  # Context already set
    main()
