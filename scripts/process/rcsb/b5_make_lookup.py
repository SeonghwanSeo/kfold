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
import pathlib
from collections import defaultdict

import lmdb
from tqdm import tqdm

import kfold.constants as C
from kfold.data.structure import RefStructure


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
    args = parser.parse_args()

    return args


def main():
    """Main function to extract sequences from npz files."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir

    # Retrieve
    print("Retrieve items from lmdb")
    lmdb_path = data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=1 * 1024 * 1024 * 1024,  # 1 GB
        readonly=True,
    )
    sequence_dict: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    with env.begin() as txn:
        n_entries = txn.stat()["entries"]
        for _, value in tqdm(txn.cursor(), total=n_entries):
            # Make npz buffer
            with io.BytesIO(value) as byte_stream:
                # Deserialize the value
                ref_structure: RefStructure = RefStructure.load_npz(byte_stream)
            metadata = ref_structure.metadata
            entry_key = metadata.id
            entry_dict = {}
            for chain in ref_structure.chains:
                entity_id = chain.entity_id
                if entity_id in entry_dict:
                    continue
                if chain.ctype.is_nonpolymer:
                    seq = ":".join(chain.get_ccd_sequence())
                    entry_dict[entity_id] = (seq, chain.ctype.name.lower())
                else:
                    sequence = chain.get_sequence()
                    if chain.ctype.is_protein:
                        standard_set = C.residue.PROTEIN_AMINO_ACIDS
                        unk = "X"
                        sequence = (
                            sequence.replace("B", "D").replace("Z", "E").replace("U", "C")
                        )
                    elif chain.ctype.is_rna:
                        standard_set = C.residue.RNA_BASES
                        unk = "N"
                    elif chain.ctype.is_dna:
                        standard_set = C.residue.DNA_BASES
                        unk = "N"
                    sequence = "".join(
                        [v if v in standard_set else unk for v in sequence]
                    )
                    entry_dict[entity_id] = (sequence, chain.ctype.name.lower())
            sequence_dict[entry_key] = entry_dict
            del ref_structure  # free memory
    env.close()
    print(f"Total entries retrieved from lmdb: {len(sequence_dict)}")

    # Load sequence IDs
    seq_to_id: dict[tuple[str, str], tuple[str, str]] = {}
    with open(args.seq_id_path) as f:
        assert pathlib.Path(args.seq_id_path).suffix == ".fasta"
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
            # residue_mapping
            res_map = f"1:{len(sequence)}->1:{len(sequence)}"
            seq_to_id[(ctype, sequence)] = (key, res_map)

    struct_to_id: dict[tuple[str, str], tuple[str, str, str]] = {}
    with open(args.struct_id_path) as f:
        assert pathlib.Path(args.struct_id_path).suffix == ".csv"
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            seq, length, source, key, res, apo_res = parts
            res_map = f"{res}->{apo_res}"
            struct_to_id[("protein", seq)] = (key, source, res_map)

    # Create lookup
    lookup: dict[str, dict] = {}
    seq_success = 0
    seq_fail = 0
    struct_success = 0
    struct_fail = 0
    for entry_key, chains in sequence_dict.items():
        entry_lookup: dict[str, dict] = {}
        for entity_id, (seq, ctype) in chains.items():
            entity_lookup: dict = {
                "type": ctype,
            }

            if (ctype, seq) in seq_to_id:
                seq_id, seq_res_map = seq_to_id[(ctype, seq)]
                seq_success += 1
                seq_emb = (
                    {
                        "path": f"{seq_id}.pt",
                        "residue_map": seq_res_map,
                    },
                )
                entity_lookup["seq_emb"] = seq_emb
            else:
                if ctype in ["protein", "rna", "dna"]:
                    print(
                        f"Sequence embedding not found for entry {entry_key}, "
                        f"entity {entity_id}."
                    )
                    seq_fail += 1

            if (ctype, seq) in struct_to_id:
                struct_success += 1
                struct_id, struct_source, struct_res_map = struct_to_id[(ctype, seq)]
                struct_emb = {
                    "path": f"{seq_id}.pt",
                    "residue_map": struct_res_map,
                }
                entity_lookup["struct_emb"] = struct_emb
                apo_info = {
                    "name": struct_id,
                    "path": f"{struct_id}.pdb.gz",
                    "residue_map": struct_res_map,
                    "source": struct_source,
                }
                entity_lookup["apo"] = [apo_info]
            else:
                if ctype == "protein":
                    print(
                        f"Structure embedding not found for entry {entry_key}, "
                        f"entity {entity_id}."
                    )
                    struct_fail += 1

            entry_lookup[entity_id] = entity_lookup
        lookup[entry_key] = entry_lookup

    print(f"Sequence embedding: {seq_success} found, {seq_fail} not found.")
    print(f"Structure embedding: {struct_success} found, {struct_fail} not found.")

    # Save lookup
    lookup_path = data_dir / "lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"Lookup saved to {lookup_path}")


if __name__ == "__main__":
    main()
