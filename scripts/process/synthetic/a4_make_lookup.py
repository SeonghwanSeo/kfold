"""
* lookup.json format
```json
{
  "6oim": {
    "1": {
      "type": "protein",
      "seq_emb": {
        "path": "protein_1.npy",
      },
      "struct_emb": {
        "path": "apo_1.npy",
      },
      "apo": [
        {
          "source": "boltz-2",
          "path": "apo_1.cif.gz",
        },
      ]
    },
    "2": {...}
  },
  "1a2c": {...}
}
```
"""

import argparse
import json
import multiprocessing
import pathlib

import msgpack
import pandas as pd
from tqdm import tqdm

from kfold.data.types.metadata import Metadata
from kfold.data.utils.io.fasta import write_fasta


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
    args = parser.parse_args()
    return args


def main():
    """Main function to extract sequences from npz files using multiprocessing."""
    # TODO: handle RNA too.
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    metadata_csv_path: pathlib.Path = data_dir / "metadata.csv"
    df: pd.DataFrame = pd.read_csv(metadata_csv_path)

    entry_id_to_row: dict[str, pd.Series] = {}
    protein_seq_to_apo_id: dict[str, str] = {}
    rna_sequences: set[str] = set()

    for row in tqdm(df.itertuples(), total=len(df)):
        # Add entry ID to row mapping
        entry_id = f"{row.data_idx}_{row.structure_idx}"
        entry_id_to_row[entry_id] = row
        # Add sequence to apo ID mapping
        for k in ["protein_0", "protein_1", "protein_2"]:
            if pd.isna(getattr(row, k)):
                continue
            seq = getattr(row, k)
            apo_id = getattr(row, f"{k}_apo_idx")
            if seq in protein_seq_to_apo_id:
                if not protein_seq_to_apo_id[seq] == apo_id:
                    print(
                        f"Sequence {seq} maps to multiple apo IDs: "
                        f"{protein_seq_to_apo_id[seq]} and {apo_id}"
                    )
            else:
                protein_seq_to_apo_id[seq] = apo_id

        # Add rna sequence
        for k in ["rna_0", "rna_1"]:
            if k not in row._fields:
                continue
            if pd.isna(getattr(row, k)):
                continue
            seq = getattr(row, k)
            rna_sequences.add(seq)
    print(f"Extracted {len(protein_seq_to_apo_id)} unique protein sequences.")
    print(f"Extracted {len(rna_sequences)} unique RNA sequences.")

    # save fasta
    seq_dir = data_dir / "sequences"
    seq_dir.mkdir(exist_ok=True)

    fasta_path: pathlib.Path = data_dir / "sequences" / "unique_protein_sequences.fasta"
    uniq_protein_sequences = [
        (f"protein_{k}", seq) for seq, k in protein_seq_to_apo_id.items()
    ]
    uniq_protein_sequences.sort(key=lambda x: x[0])  # Sort by apo ID
    protein_sequence_to_id: dict[str, str] = {
        seq: seq_id for seq_id, seq in uniq_protein_sequences
    }
    if len(uniq_protein_sequences) > 0:
        write_fasta(uniq_protein_sequences, fasta_path)

    uniq_rna_sequences = [
        (f"rna_{i}", seq)
        for i, seq in enumerate(sorted(rna_sequences, key=lambda x: (len(x), x)))
    ]
    fasta_path: pathlib.Path = data_dir / "sequences" / "unique_rna_sequences.fasta"
    if len(uniq_rna_sequences) > 0:
        write_fasta(uniq_rna_sequences, fasta_path)
    rna_sequence_to_id: dict[str, str] = {
        seq: seq_id for seq_id, seq in uniq_rna_sequences
    }

    # Load metadata
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        manifest = msgpack.unpack(f, raw=False)

    all_lookup: dict[str, dict[str, dict]] = {}
    for m_dict in manifest:
        m = Metadata.from_dict(m_dict)
        # Check if entry is in filtered set
        entry_id = m.id
        if entry_id not in entry_id_to_row:
            continue
        row = entry_id_to_row[entry_id]
        prot_idx = 0
        rna_idx = 0
        entry_lookup = {}
        for c_m in m.chains:
            if c_m.ctype.is_protein:
                seq = getattr(row, f"protein_{prot_idx}")
                apo_id = getattr(row, f"protein_{prot_idx}_apo_idx")
                seq_id = protein_sequence_to_id[seq]
                prot_idx += 1
                entry_lookup[str(c_m.entity_id)] = {
                    "type": "protein",
                    "seq_emb": {
                        "path": f"{seq_id}.npy",
                    },
                    "struct_emb": {
                        "path": f"apo_{apo_id}.npy",
                    },
                    "apo": [
                        {
                            "source": "boltz-2",
                            "path": f"apo_{apo_id}.cif.gz",
                        }
                    ],
                }
            elif c_m.ctype.is_rna:
                seq = getattr(row, f"rna_{rna_idx}")
                seq_id = rna_sequence_to_id[seq]
                rna_idx += 1
                entry_lookup[str(c_m.entity_id)] = {
                    "type": "rna",
                    "seq_emb": {
                        "path": f"{seq_id}.npy",
                    },
                }
            else:
                continue

        all_lookup[entry_id] = entry_lookup

    # Save lookup
    lookup_path = data_dir / "lookup.json"
    print(f"Saving lookup to {lookup_path}")
    with open(lookup_path, "w") as f:
        json.dump(all_lookup, f, indent=2)

    lookup_path = data_dir / "lookup.msgpack"
    print(f"Saving lookup to {lookup_path} (msgpack format)")
    with open(lookup_path, "wb") as f:
        msgpack.pack(all_lookup, f)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork", force=True)  # Faster on Linux/MacOS
    except RuntimeError:
        pass  # Context already set
    main()
