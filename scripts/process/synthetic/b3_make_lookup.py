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

import lmdb
from tqdm import tqdm

from kfold.data.structure import RefStructure

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
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes.",
    )
    args = parser.parse_args()
    return args


def init_worker(seq_to_id: dict, lmdb_path: pathlib.Path):
    """
    Initialize worker process with read-only shared data.
    This avoids pickling large dictionaries for every task.
    """
    global _GLOBAL_SEQ_TO_ID, _GLOBAL_STRUCT_TO_ID, _GLOBAL_LMDB_PATH
    _GLOBAL_SEQ_TO_ID = seq_to_id
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

    local_lookup: dict[str, dict] = {}
    stats = {
        "seq_success": 0,
        "seq_fail": 0,
        "seq_success_complex": 0,
        "seq_fail_complex": 0,
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
            entry_key = metadata.id
            entry_dict = {}

            # Process Chains
            for chain in ref_structure.chains:
                entity_id = chain.entity_id
                if entity_id in entry_dict:
                    continue
                sequence = chain.get_sequence().replace("X", "A")
                entry_dict[entity_id] = (sequence, chain.ctype.name.lower())

            # free memory explicitly for the object
            del ref_structure

            # Build Lookup Entry
            entry_lookup: dict[str, dict] = {}
            for entity_id, (seq, ctype) in entry_dict.items():
                if (ctype, seq) not in _GLOBAL_SEQ_TO_ID:
                    stats["seq_fail"] += 1
                    entry_lookup[entity_id] = {"type": ctype}
                    continue

                # Match Sequence Embedding
                seq_id, seq_res_map = _GLOBAL_SEQ_TO_ID[(ctype, seq)]
                stats["seq_success"] += 1
                seq_emb = {
                    "path": f"{seq_id}.pt",
                    "residue_map": seq_res_map,
                }
                struct_emb = {
                    "path": f"{seq_id}.pt",
                    "residue_map": seq_res_map,
                }
                apo_info = {
                    "name": f"{seq_id}",
                    "path": f"{seq_id}.pdb.gz",
                    "residue_map": seq_res_map,
                    "source": "esmfold",
                }
                entity_lookup_data = {
                    "type": ctype,
                    "seq_emb": seq_emb,
                    "struct_emb": struct_emb,
                    "apo": [apo_info],
                }
                entity_lookup_data["seq_emb"] = seq_emb
                entry_lookup[entity_id] = entity_lookup_data
                stats["seq_success"] += 1

            local_lookup[entry_key] = entry_lookup

            if all("seq_emb" in v for v in entry_lookup.values()):
                stats["seq_success_complex"] += 1
            else:
                stats["seq_fail_complex"] += 1

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
            res_map = f"1:{len(sequence)}->1:{len(sequence)}"
            seq_to_id[("protein", sequence)] = (key, res_map)
    return seq_to_id


def main():
    """Main function to extract sequences from npz files using multiprocessing."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    print("Loading ID mappings...")
    seq_id_path = data_dir / "unique_proteins.fasta"
    seq_to_id = load_sequence_ids(seq_id_path)

    print("Retrieving keys from LMDB...")
    lmdb_path = data_dir / "structure.lmdb"
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
        "seq_success_complex": 0,
        "seq_fail_complex": 0,
    }

    # Initialize pool with shared read-only data
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(seq_to_id, lmdb_path),
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
        f"Complex entries: {global_stats['seq_success_complex']} complete, "
        f"{global_stats['seq_fail_complex']} incomplete."
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
