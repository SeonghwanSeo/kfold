"""
* apo_lookup.json format
```json
{
  "6oim": {
    "1": [
      {
        "source": "boltz-2",
        "name": "apo_1",
      },
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
import pathlib

import lmdb
import msgpack
import pandas as pd
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import RefStructure

PROTEIN_AA = C.residue.PROTEIN_AMINO_ACIDS_SET | {"B", "Z"}


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

    metadata_csv_path: pathlib.Path = data_dir / "metadata.csv"
    df: pd.DataFrame = pd.read_csv(metadata_csv_path)

    entry_id_to_row: dict[str, pd.Series] = {}
    seq_to_apo_id: dict[str, str] = {}

    for row in tqdm(df.itertuples(), total=len(df)):
        # Add entry ID to row mapping
        entry_id = f"{row.data_idx}_{row.structure_idx}"
        entry_id_to_row[entry_id] = row
        # Add sequence to apo ID mapping
        for k in ["protein_0", "protein_1", "protein_2"]:
            if pd.isna(getattr(row, k)):
                continue
            seq = getattr(row, k)
            seq = "".join(v if v in PROTEIN_AA else "X" for v in seq)
            apo_id = getattr(row, f"{k}_apo_idx")
            if seq in seq_to_apo_id:
                if not seq_to_apo_id[seq] == apo_id:
                    pass
            else:
                seq_to_apo_id[seq] = apo_id

    print(f"Extracted {len(seq_to_apo_id)} unique protein sequences.")

    all_lookup: dict[str, dict[str, list[dict[str, str]]]] = {}
    if args.remap:
        lmdb_path = data_dir / "structure.lmdb"
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
        txn = env.begin()
        for _, v in tqdm(txn.cursor(), total=txn.stat()["entries"]):
            with io.BytesIO(v) as f:
                ref_struct: RefStructure = RefStructure.load_npz(f)
            entry_id = ref_struct.id

            entry_lookup: dict[int, list[dict[str, str]]] = {}
            visited_entities: set[int] = set()
            for c in ref_struct.chains:
                entity_id = c.entity_id
                if entity_id in visited_entities:
                    continue
                visited_entities.add(c.entity_id)
                if not c.ctype.is_protein:
                    continue
                seq = c.get_sequence()
                if seq not in seq_to_apo_id:
                    print(
                        f"Warning: {entry_id} entity {entity_id} has sequence "
                        f"not found in apo mapping. Skipping:\n"
                        f"{seq}"
                    )
                    continue
                apo_id = seq_to_apo_id[seq]
                entry_lookup[entity_id] = [
                    {
                        "source": "boltz-2",
                        "name": f"apo_{apo_id}",
                    }
                ]
            all_lookup[entry_id] = {str(k): v for k, v in entry_lookup.items()}
    else:
        for row in tqdm(df.itertuples(), total=len(df)):
            entry_id = f"{row.data_idx}_{row.structure_idx}"
            entry_lookup: dict[int, list[dict[str, str]]] = {}
            for k in ["protein_0", "protein_1", "protein_2"]:
                if pd.isna(getattr(row, k)):
                    continue
                apo_id = getattr(row, f"{k}_apo_idx")
                entity_id = int(k.split("_")[1]) + 1
                entry_lookup[entity_id] = [
                    {
                        "source": "boltz-2",
                        "name": f"apo_{apo_id}",
                    }
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
        multiprocessing.set_start_method("fork", force=True)  # Faster on Linux/MacOS
    except RuntimeError:
        pass  # Context already set
    main()
