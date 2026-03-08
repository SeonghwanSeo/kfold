"""
* apo_lookup.json format
```json
{
  "6oim": {
    "1": [
      {
        "source": "boltz-2",
        "name": "protein_0"
      },
      {
        "source": "esmfold",
        "name": "protein_0"
      },
      ...
    ]
    "2": [...]
  },
  "1a2c": {...}
}
```
"""

import argparse
import io
import json
import multiprocessing
import os
import pathlib

import lmdb
import msgpack
import pandas as pd
from tqdm import tqdm

from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import read_fasta, write_fasta

standard_aa = "ACDEFGHIKLMNPQRSTVWY"
mapping = {aa: aa for aa in standard_aa}  # Add standard amino acids to the mapping
mapping |= {
    "B": "D",  # Aspartic acid or Asparagine -> Aspartic acid
    "Z": "E",  # Glutamic acid or Glutamine -> Glutamic acid
    "J": "L",  # Leucine or Isoleucine -> Leucine
    "X": "A",  # Unknown amino acid remains unchanged
}


def to_standard_aa(seq: str) -> str:
    # Map non-standard amino acids to their standard counterparts
    return "".join(mapping.get(aa, "A") for aa in seq)


def load_protein_from_struct(item: tuple[bytes, bytes]) -> tuple[str, dict[int, str]]:
    k, v = item
    with io.BytesIO(v) as f:
        ref_struct: RefStructure = RefStructure.load_npz(f)
    entity_id_to_seq: dict[int, str] = {}
    for c in ref_struct.chains:
        if c.entity_id in entity_id_to_seq:
            continue
        if c.ctype.is_protein:
            seq = c.get_sequence()
            entity_id_to_seq[c.entity_id] = to_standard_aa(seq)
    return ref_struct.id, entity_id_to_seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Whether to remap the metadata protein orders to match the structure files.",
    )
    args = parser.parse_args()
    return args


def main():
    # TODO: handle RNA too.
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    apo_dir = data_dir / "apo/"
    metadata_path = data_dir / "metadata.csv"
    uniq_protein_fasta = data_dir / "sequences" / "unique_protein_sequences.fasta"

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Load unique protein sequences (= apo ids)
    if not uniq_protein_fasta.exists():
        uniq_protein_fasta.parent.mkdir(parents=True, exist_ok=True)
        # Extract all unique protein sequenccs
        sequences = set()
        for row in df.itertuples():
            # HACK: We only use up to 3 proteins per entry.
            for i in [0, 1, 2]:
                sequence = getattr(row, f"protein_{i}")
                if not pd.isna(sequence):
                    sequences.add(to_standard_aa(sequence))
        uniq_sequences: list[tuple[str, str]] = [
            (f"protein_{i}", seq)
            for i, seq in enumerate(sorted(sequences, key=lambda x: (len(x), x)))
        ]
        write_fasta(uniq_sequences, uniq_protein_fasta)
    else:
        uniq_sequences = read_fasta(uniq_protein_fasta)

    seq_to_apo_id = {seq: id for id, seq in uniq_sequences}
    print(f"Extracted {len(seq_to_apo_id)} unique protein sequences.")

    # Get existing apo files
    existing_apo_files = set()
    for source in ["boltz-2", "esmfold"]:
        source_dir = apo_dir / source
        if source_dir.exists():
            for file in source_dir.iterdir():
                apo_id = file.name.split(".")[0]
                existing_apo_files.add((source, apo_id))

    all_lookup: dict[str, dict[str, list[dict[str, str]]]] = {}
    if args.remap:
        # Remap the protein orders and their entity ids based on the
        # structure files instead of the metadata, since the metadata
        # orders are not guaranteed to be correct.
        lmdb_path = data_dir / "structure.lmdb"
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
        txn = env.begin()
        with multiprocessing.Pool(args.num_workers) as pool:
            for entry_id, entry_info in tqdm(
                pool.imap_unordered(load_protein_from_struct, txn.cursor()),
                total=txn.stat()["entries"],
            ):
                entry_lookup: dict[int, list[dict[str, str]]] = {}
                for entity_id, seq in entry_info.items():
                    if seq not in seq_to_apo_id:
                        print(
                            f"Warning: entity {entity_id} has sequence not found "
                            f"in apo mapping. Skipping:\n{seq}"
                        )
                    else:
                        apo_id = seq_to_apo_id[seq]
                        apo_infos = []
                        for source in ["boltz-2", "esmfold"]:
                            # check is there any protein file with this apo_id
                            if (source, apo_id) not in existing_apo_files:
                                print(
                                    f"Warning: apo '{source}:{apo_id}' from "
                                    f"entity {entity_id} not found. Skipping."
                                )
                                continue
                            apo_infos.append({"source": source, "name": apo_id})
                        entry_lookup[entity_id] = apo_infos
                all_lookup[entry_id] = {str(k): v for k, v in entry_lookup.items()}
        txn.abort()  # Close the LMDB environment
    else:
        for row in tqdm(df.itertuples(), total=len(df)):
            entry_id = f"{row.data_idx}_{row.structure_idx}"
            entry_lookup: dict[int, list[dict[str, str]]] = {}
            for k in ["protein_0", "protein_1", "protein_2"]:
                seq = getattr(row, k)
                if pd.isna(seq):
                    continue
                apo_id = seq_to_apo_id[to_standard_aa(seq)]
                entity_id = int(k.split("_")[1]) + 1
                entry_lookup[entity_id] = [
                    {"source": "boltz-2", "name": apo_id},
                    {"source": "esmfold", "name": apo_id},
                ]
            all_lookup[entry_id] = {str(k): v for k, v in entry_lookup.items()}

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
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Context already set
    main()
