import argparse
import io
import multiprocessing
import pathlib
from collections import OrderedDict
from typing import Any

import lmdb
from tqdm import tqdm

import kfold.constants as C
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.fasta import save_fasta


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
        default=16,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        choices=["rcsb_train", "rcsb_val"],
        help="Prefix for sequence IDs.",
    )
    args = parser.parse_args()
    return args


_GLOBAL_ENV: lmdb.Environment = None


def init_worker(lmdb_path: pathlib.Path):
    """Ignore SIGINT in worker processes to allow graceful termination."""
    global _GLOBAL_ENV
    _GLOBAL_ENV = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=True,  # NOTE: improves read performance
        meminit=False,
    )


def read_lmdb(keys: list[bytes]) -> list[tuple[str, dict[int, tuple[C.ChainType, str]]]]:
    """Read entries from LMDB given a list of keys."""
    global _GLOBAL_ENV
    assert _GLOBAL_ENV is not None, "LMDB environment is not initialized."

    data: list[tuple[str, dict[int, tuple[C.ChainType, str]]]] = []
    with _GLOBAL_ENV.begin() as txn:
        for key in keys:
            value = txn.get(key)
            buffer = io.BytesIO(value)
            struct = RefStructure.load_npz(buffer)
            pdb_id = struct.id
            assert pdb_id == key.decode("utf-8")
            sequences: dict[int, tuple[C.ChainType, str]] = {}
            for chain in struct.chains:
                if chain.ctype.is_polymer:
                    seq = chain.get_sequence(map_to_standard=True)
                    sequences[chain.entity_id] = (chain.ctype, seq)
            data.append((pdb_id, sequences))
    return data


if __name__ == "__main__":
    args = parse_args()
    prefix = args.prefix

    save_dir = args.data_dir / "sequences/"
    save_dir.mkdir(parents=True, exist_ok=True)

    lmdb_path = args.data_dir / "structure.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    keys: list[bytes] = []
    with env.begin() as txn:
        num_entries = txn.stat()["entries"]
        with txn.cursor() as cursor:
            for key, _ in cursor:
                keys.append(key)
    env.close()

    # Chunk keys for multiprocessing
    chunk_size = ((len(keys) + args.num_workers - 1) // args.num_workers) // 10
    chunk_size = max(chunk_size, 1)
    key_chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
    print(f"Using chunk size: {chunk_size}")
    print(f"Total chunks: {len(key_chunks)}")

    # Read sequences in parallel
    print("Extracting sequences from LMDB...")
    data: list[tuple[str, dict[int, tuple[C.ChainType, str]]]] = []
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(lmdb_path,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(read_lmdb, key_chunks), total=len(key_chunks)
        ):
            data.extend(result)

    data.sort(key=lambda x: x[0])  # Sort by structure ID

    # Save sequences to a fasta file
    fasta_path = save_dir / "rcsb_polymer.fasta"
    records: list[tuple[str, str]] = []
    for pdb_id, sequences in data:
        for entity_id in sorted(sequences.keys()):
            ctype, seq = sequences[entity_id]
            ctype_str = ctype.name.lower()
            chain_id = f"{pdb_id}_{entity_id}_{ctype_str}"
            records.append((chain_id, seq))
    save_fasta(records, fasta_path, width=None)

    # Save sequences to separate fasta files by chain type
    for ctype in [C.ChainType.PROTEIN, C.ChainType.DNA, C.ChainType.RNA]:
        ctype_str = ctype.name.lower()
        records = []
        for pdb_id, sequences in data:
            for entity_id in sorted(sequences.keys()):
                chain_ctype, seq = sequences[entity_id]
                if chain_ctype is ctype:
                    chain_id = f"{pdb_id}_{entity_id}_{ctype_str}"
                    records.append((chain_id, seq))

    # Extract unique sequences
    unique_sequences: dict[str, set[str]] = {
        "protein": set(),
        "dna": set(),
        "rna": set(),
    }
    for _, sequences in data:
        for _, (ctype, seq) in sequences.items():
            if ctype is C.ChainType.PROTEIN:
                unique_sequences["protein"].add(seq)
            elif ctype is C.ChainType.DNA:
                unique_sequences["dna"].add(seq)
            elif ctype is C.ChainType.RNA:
                unique_sequences["rna"].add(seq)

    print("Unique sequences found:")
    for ctype_str, seqs in unique_sequences.items():
        print(f"  {ctype_str}: {len(seqs)}")

    # Export unique sequences
    sequence_to_id: dict[str, dict[str, str]] = {
        "protein": OrderedDict(),
        "dna": OrderedDict(),
        "rna": OrderedDict(),
    }
    for ctype_str, seqs in unique_sequences.items():
        for i, seq in enumerate(sorted(seqs, key=lambda x: (len(x), x))):
            seq_id = f"{prefix}_{ctype_str}_{i + 1:06d}"
            sequence_to_id[ctype_str][seq] = seq_id

    for ctype_str in ["protein", "dna", "rna"]:
        fasta_path = save_dir / f"unique_{ctype_str}.fasta"
        records = [(seq_id, seq) for seq, seq_id in sequence_to_id[ctype_str].items()]
        save_fasta(records, fasta_path, width=None)

    def row(*args: Any) -> str:
        return ",".join(str(a) for a in args) + "\n"

    # Replace 'UNK' to 'ALA' in protein sequences for folding
    fasta_path = save_dir / "apo_protein.fasta"
    records: list[tuple[str, str]] = []
    for seq, seq_id in sequence_to_id["protein"].items():
        apo_seq = seq.replace("X", "A")
        records.append((seq_id, apo_seq))
    save_fasta(records, fasta_path, width=None)

    # Save mapping from each rcsb entry to sequence IDs
    mapping_path = save_dir / "sequence_id.csv"
    with open(mapping_path, "w") as f:
        f.write(row("pdb_id", "entity_id", "type", "length", "seq_id"))
        for pdb_id, sequences in data:
            for entity_id in sorted(sequences.keys()):
                ctype, seq = sequences[entity_id]
                ctype_str = ctype.name.lower()
                seq_id = sequence_to_id[ctype_str][seq]
                f.write(row(pdb_id, entity_id, ctype_str, len(seq), seq_id))
