import argparse
import io
import logging
import multiprocessing
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

import kfold.constants as C
from kfold.data.metadata import Metadata
from kfold.data.structure import TokenizedStructure
from kfold.training.folding.dataset.datamodule import load_manifest

logger = logging.getLogger(__name__)

# Global variable for the worker process to hold the LMDB environment
_env = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get polymer sequences from processed LMDB dataset (Multiprocessing)."
    )
    parser.add_argument(
        "--processed_lmdb_path",
        type=Path,
        help="Path to the processed LMDB dataset.",
        default="/cache/wykim_lab/kfold_data/kfold_rcsb_processed_v251120.lmdb/",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        help="Path to the processed LMDB dataset.",
        default="/cache/wykim_lab/kfold_data/manifests/af3_manifest.pkl",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path of directory to save the distance map.",
        default="/cache/wykim_lab/icl_shwan/kfold/contact/",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes. Defaults to CPU count.",
        default=len(os.sched_getaffinity(0)),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of keys to process per batch.",
        default=500,
    )
    return parser.parse_args()


def init_worker(lmdb_path: Path | str):
    """
    Initialize the worker process by opening the LMDB environment.
    This runs once per process.
    """
    global _env
    _env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def get_center_coords(chain_i: int, struct: TokenizedStructure) -> np.ndarray:
    """Returns token center coordinates of shape (num_tokens, 3)."""
    st = int(struct.chain.token_start[chain_i])
    end = st + int(struct.chain.num_tokens[chain_i])

    # Extract token center coordinates
    token_indices = np.arange(st, end)  # (num_tokens,)
    token_center = struct.token.center_index[st:end]  # (n_tokens,)
    atom_coords = struct.atom.coords  # (num_tokens, 24, 3)
    center_coords = atom_coords[token_indices, token_center].reshape(
        -1, 3
    )  # (num_tokens, 3)

    mask = struct.token.resolved_mask[st:end]  # (num_tokens,)
    center_coords[~mask] = np.nan  # Set unresolved tokens to NaN

    # Convert token coords to residue coords
    # HACK: This returns N instead of CA for unstandard amino acids.
    res_st = int(struct.chain.residue_start[chain_i])
    res_end = res_st + int(struct.chain.num_residues[chain_i])
    res_token_st = struct.residue.token_start[res_st:res_end] - st  # (num_res,)
    center_coords = center_coords[res_token_st]  # (num_res, 3)

    return center_coords


def process_batch(
    manifest: list[Metadata],
    root_dir: Path,
) -> None:
    """
    Process a batch of keys. Returns a list of (pdb_id, polymer_sequences).
    """
    global _env

    txn = _env.begin(write=False)

    for record in manifest:
        pdb_id: str = record.id

        assert record.exp is not None
        if record.exp.resolution > 9.0:
            continue

        value_bytes = txn.get(pdb_id.encode("utf-8"))
        if value_bytes is None:
            logger.warning(f"Key {pdb_id} not found in LMDB. Skipping.")
            continue
        try:
            # Use io.BytesIO to wrap the raw bytes
            with io.BytesIO(value_bytes) as byte_stream:
                struct: TokenizedStructure = TokenizedStructure.load_npz(byte_stream)
        except Exception as e:
            logger.error(f"Error processing {pdb_id}: {e}")
            continue

        # Get asym_id -> chain_index mapping
        asym_id_map: dict[int, int] = {}
        for chain_i in range(struct.num_chains):
            asym_id = int(struct.chain.asym_id[chain_i])
            asym_id_map[asym_id] = chain_i

        # Get sequence
        entity_sequences: dict[int, tuple[C.ChainType, str]] = {}
        for chain_i in range(struct.num_chains):
            entity_id: int = int(struct.chain.entity_id[chain_i])
            if entity_id in entity_sequences:
                continue
            ctype = C.ChainType(int(struct.chain.chain_type[chain_i]))
            if ctype is C.ChainType.LIGAND:
                # Skip ligands
                continue
            residue_st = int(struct.chain.residue_start[chain_i])
            residue_end = residue_st + int(struct.chain.num_residues[chain_i])
            sequence_res_names = struct.residue.name[residue_st:residue_end]
            char_list = [
                C.residue.get_one_letter(str(res_name).strip(), ctype)
                for res_name in sequence_res_names
            ]
            seq = "".join(char_list)
            entity_sequences[entity_id] = (ctype, seq)

        interfaces = record.interfaces

        interface_contact: dict[tuple[int, int], dict] = {}
        interface_counter: dict[tuple[int, int], int] = defaultdict(int)

        for iface in interfaces:
            asym_id1, asym_id2 = iface.asym_ids
            chain_i1 = asym_id_map.get(asym_id1, None)
            chain_i2 = asym_id_map.get(asym_id2, None)

            if chain_i1 is None or chain_i2 is None:
                logger.warning(
                    f"Interface asym_ids {asym_id1}, {asym_id2} not found"
                    f" in structure {pdb_id}. Skipping."
                )
                continue

            entity_id1 = int(struct.chain.entity_id[chain_i1])
            entity_id2 = int(struct.chain.entity_id[chain_i2])
            i1, i2 = sorted((entity_id1, entity_id2))
            entity_pair = (i1, i2)

            interface_counter[entity_pair] += 1
            if interface_counter[entity_pair] > 5:
                # Avoid excessive computation for the same entity pair
                continue

            # NOTE: Currently, we skip ligands since it is impossible to
            # distinguish small molecule ligands from metals.
            ctype1 = C.ChainType(int(struct.chain.chain_type[chain_i1]))
            ctype2 = C.ChainType(int(struct.chain.chain_type[chain_i2]))
            if ctype1 is C.ChainType.LIGAND:
                continue
            if ctype2 is C.ChainType.LIGAND:
                continue

            # Get contact map

            coords1 = get_center_coords(chain_i1, struct)  # (N1, 3)
            if np.all(np.isnan(coords1)):
                continue
            coords2 = get_center_coords(chain_i2, struct)  # (N2, 3)
            if np.all(np.isnan(coords2)):
                continue

            dists = cdist(coords1, coords2)  # (N1, N2)
            dists = np.nan_to_num(dists, nan=30.0)
            dists[dists > 30] = 30  # Cap max distance
            n_contacts = np.sum(dists < 8)
            if n_contacts == 0:
                continue

            if (
                entity_pair not in interface_contact
                or n_contacts > interface_contact[entity_pair]["n_contacts"]
            ):
                # convert to distance map with uint8 type
                dist_map_uint8 = np.floor(dists).astype(np.uint8)
                interface_contact[entity_pair] = {
                    "dist_map": dist_map_uint8,
                    "n_contacts": n_contacts,
                }

        if not interface_contact:
            # No valid interfaces found
            continue

        save_dir = root_dir / pdb_id[1:3] / pdb_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save polymer sequences
        sequence_path = save_dir / f"{pdb_id}_polymer_sequences.csv"
        with open(sequence_path, "w") as f:
            f.write("entity_id,chain_type,sequence\n")
            for entity_id, (ctype, sequence) in entity_sequences.items():
                chain_type_str = ctype.name.lower()
                f.write(f"{entity_id},{chain_type_str},{sequence}\n")
        # Save contact maps
        for (entity_id1, entity_id2), contact_data in interface_contact.items():
            ctype1_str = entity_sequences[entity_id1][0].name.lower()
            ctype2_str = entity_sequences[entity_id2][0].name.lower()
            dist_map = contact_data["dist_map"]
            save_path = (
                save_dir
                / f"{pdb_id}_{entity_id1}_{entity_id2}_{ctype1_str}_{ctype2_str}.npz"
            )
            np.savez_compressed(save_path, dist_map=dist_map)
    txn.abort()


def get_all_keys(lmdb_path: str) -> list[bytes]:
    """Quickly retrieve all keys from the LMDB."""
    logger.info("Reading all keys from LMDB...")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        # keys=True, values=False is much faster just to list keys
        with txn.cursor() as cursor:
            keys = [key for key in cursor.iternext(keys=True, values=False)]
    env.close()
    logger.info(f"Found {len(keys)} keys.")
    return keys


def main(args):
    # 1. Load manifest
    manifest: list[Metadata] = load_manifest(args.manifest_path)
    logger.info(f"Loaded manifest with {len(manifest)} records.")

    # 2. Chunk keys for batch processing
    # Processing in batches reduces pickling overhead
    chunk_size = args.batch_size
    manifest_chunk = [
        manifest[i : i + chunk_size] for i in range(0, len(manifest), chunk_size)
    ]

    # 3. Setup Multiprocessing Pool
    logger.info(f"Starting multiprocessing with {args.num_workers} workers...")

    # Prepare output directory
    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    func = partial(process_batch, root_dir=output_dir)

    lmdb_path_str = str(args.processed_lmdb_path)
    if args.num_workers <= 1:
        # Single process (for debugging)
        init_worker(lmdb_path_str)
        for batch in tqdm(
            manifest_chunk,
            total=len(manifest_chunk),
            desc="Extracting contact maps",
        ):
            func(batch)
        return
    else:
        with multiprocessing.Pool(
            processes=args.num_workers, initializer=init_worker, initargs=(lmdb_path_str,)
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(func, manifest_chunk),
                total=len(manifest_chunk),
                desc="Extracting contact maps",
            ):
                pass


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    main(args)
