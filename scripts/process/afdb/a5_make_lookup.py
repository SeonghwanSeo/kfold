"""
Create a lookup JSON file mapping AFDB and ESMFold apo structures.

* apo_lookup.json format
```json
{
  "P01116": {
    "1": [
      {
        "source": "esmfold"
        "name": "P01116",
      },
    ],
  },
}
```
"""

import argparse
import json
import pathlib

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
        "--split",
        required=True,
        type=str,
        choices=["long", "short"],
        help="Data split to process.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"

    # Load manifest
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        metadata_dicts: list[dict] = msgpack.unpack(f, raw=False)
    uniprot_ids = [md["id"] for md in metadata_dicts]

    all_lookup: dict[str, dict[str, list[dict]]] = {}
    entity_id = "1"
    for uniprot_id in tqdm(uniprot_ids, desc="Processing Uniprot IDs"):
        all_lookup[uniprot_id] = {entity_id: [{"source": "esmfold", "name": uniprot_id}]}

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
