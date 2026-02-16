"""
Create a lookup JSON file mapping AFDB PDB entries

* lookup.json format
```json
{
  "A0A0A0": {
    "1": {
      "type": "protein",
      "seq_emb": {
        "path": "C2/B1/A0B1C2.npy",
      },
      "struct_emb": {
        "path": "C2/B1/A0B1C2.npy",
      }
    }
  },
  "...": { ... },
}
```
"""

import argparse
import json
import multiprocessing
import pathlib
from typing import Any

import msgpack
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
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
        choices=["long", "short"],
        help="Data split to process.",
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to process AFDB entries and create lookup JSON."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"

    # Load metadata
    metadata_path = data_dir / "manifest.msgpack"
    with open(metadata_path, "rb") as f:
        metadata = msgpack.unpack(f)
    print(f"Loaded metadata for {len(metadata)} entries from {metadata_path}")

    all_lookup: dict[str, Any] = {}
    for entry in tqdm(metadata, desc="Processing entries"):
        entry_id = entry["id"]
        # Construct file path based on entry ID (e.g., "A0B1C2" -> "C2/B1/A0B1C2.npy")
        file_path = f"{entry_id[-2:]}/{entry_id[-4:-2]}/{entry_id}.npy"
        # Construct lookup entry
        # For AFDB, we have only one chain per entry, so we can use "1" as the entity ID
        lookup_entry = {
            "1": {
                "type": "protein",
                "seq_emb": {
                    "path": file_path,
                },
                "struct_emb": {
                    "path": file_path,
                },
            }
        }
        # Add to lookup
        all_lookup[entry_id] = lookup_entry

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
    main()
