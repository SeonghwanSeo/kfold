import argparse
import io
import logging
from functools import lru_cache
from pathlib import Path

import lmdb
from tqdm import tqdm

import kfold.constants as C
from kfold.data.structure import TokenizedStructure

logger = logging.getLogger(__name__)


# TODO: remove default path before publish
def parse_args():
    parser = argparse.ArgumentParser(
        description="Get protein sequences from processed LMDB dataset."
    )
    parser.add_argument(
        "--processed_lmdb_path",
        type=Path,
        help="Path to the processed LMDB dataset.",
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/structures/kfold_rcsb_processed_v251120.lmdb/",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of directory to save the sequences (fasta).",
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_protein_sequences.fasta",
    )
    return parser.parse_args()


@lru_cache(maxsize=100)
def get_one_letter(name: str) -> str:
    try:
        idx = C.residue.PROTEIN_RESIDUES.index(name)
    except ValueError:
        idx = C.residue.PROTEIN_RESIDUES.index("UNK")
    return C.residue.PROTEIN_AMINO_ACIDS[idx]


def main(args):
    # Load lmdb dataset
    env: lmdb.Environment = lmdb.open(
        str(args.processed_lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False, buffers=True)
    keys = [key.tobytes() for key, _ in txn.cursor()]
    logger.info(f"Total structures in LMDB: {len(keys)}")

    all_sequences: list[tuple[str, int, str]] = []
    for key in tqdm(keys):
        pdb_id = key.decode()
        value_bytes = txn.get(key)
        if value_bytes is None:
            logger.warning(f"Key {pdb_id} not found in LMDB. Skipping.")
            continue

        # Use io.BytesIO to wrap the raw bytes
        with io.BytesIO(value_bytes) as byte_stream:
            struct = TokenizedStructure.load_npz(byte_stream)

        entity_sequences: dict[int, str] = {}
        num_chains = struct.num_chains

        residue_st: int = 0
        for chain_id in range(num_chains):
            residue_end = residue_st + struct.chain.num_residues[chain_id].item()
            if struct.chain.chain_type[chain_id] != C.ChainType.PROTEIN:
                residue_st = residue_end
                continue
            entity_id: int = struct.chain.entity_id[chain_id].item()
            if entity_id in entity_sequences:
                residue_st = residue_end
                continue

            sequence_tokens = struct.residue.name[residue_st:residue_end]
            sequence = "".join(
                [get_one_letter(str(token).strip()) for token in sequence_tokens]
            )
            entity_sequences[entity_id] = sequence
            residue_st = residue_end

        for entity_id in sorted(entity_sequences.keys()):
            sequence = entity_sequences[entity_id]
            all_sequences.append((pdb_id, entity_id, sequence))
    env.close()
    logger.info(f"Total protein sequences extracted: {len(all_sequences)}")

    # Final log (for guidance)
    logger.info(
        "If you excluded the complexes with >300 chains during preprocessing, "
        "The number of sequences extracted here may be ~375k\n"
        "Otherwise, it may be ~401k sequences."
    )

    # Save all sequences to a file
    output_file = args.output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for pdb_id, entity_id, sequence in all_sequences:
            f.write(f">{pdb_id}_{entity_id}_protein\n")
            f.write(f"{sequence}\n")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
