"""Tokenize the apo structures"""

import argparse
import io
import pathlib

import lmdb
import msgpack
import numpy as np
import torch
from tqdm import tqdm

from kfold.constants.atom import protein_atom37_order as ATOM37_ORDER
from kfold.constants.sequence import encode_protein_sequence
from kfold.data.types.structure import Chain, RefStructure
from kfold.model.modules.structure_encoder.unitok import UniTok

PADDING_SIZES = [32, 64, 128, 200, 256, 384, 512, 640, 768, 1024, 1280]
BATCH_THRESHOLD = 1280
SHORT_BATCH_SIZE = 64  # short: seqlen <= 200
LONG_BATCH_SIZE = 12  # long: seqlen > 200


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
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=pathlib.Path,
        help="Path to UniTok checkpoint.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_chunk",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.Dataset):
    """Dataset for processing RCSB PDB entries."""

    def __init__(self, keys: list[str], lmdb_path: str | pathlib.Path):
        self.keys: list[str] = keys
        self._lmdb_path = str(lmdb_path)
        self._lmdb_env = None

    @property
    def lmdb_env(self):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_path, readonly=True, lock=False, readahead=False
            )
        return self._lmdb_env

    def __del__(self):
        if self._lmdb_env is not None:
            self._lmdb_env.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        k = self.keys[index]
        try:
            with self.lmdb_env.begin() as txn:
                v = txn.get(k.encode("utf-8"))
            with io.BytesIO(v) as f:
                struct: RefStructure = RefStructure.load_npz(f)
        except Exception as e:
            print(f"Error loading structure for key {k}: {e}")
            return k, None, None

        chain: Chain = struct.chains[0]
        del struct

        seq = chain.get_sequence(map_to_standard=True)
        atom_names: list[str] = chain.atom.name.tolist()  # [Natom,]
        coords: np.ndarray = chain.atom.coords  # [Natom, 3]

        atom37_coords = np.full((len(seq), 37, 3), np.nan, dtype=np.float32)
        for res_i in range(len(seq)):
            atom_slice = chain.residue.get_atom_slice(res_i + 1)
            atom_names_i = atom_names[atom_slice]
            coords_i = coords[atom_slice]
            # Pad or truncate to 37 atoms per residue
            for an, c in zip(atom_names_i, coords_i, strict=True):
                if an in ATOM37_ORDER:
                    atom37_coords[res_i, ATOM37_ORDER[an]] = c
        seq_tok = encode_protein_sequence(seq)

        seq_t = torch.tensor(seq_tok, dtype=torch.long)
        coords_t = torch.tensor(atom37_coords, dtype=torch.float32)
        return k, seq_t, coords_t


def collate_fn(batch):
    """Collate function to filter out failed samples."""
    val_batches = [b for b in batch if b[1] is not None and b[2] is not None]
    if len(val_batches) == 0:
        return None, None, None, None

    keys, seq_token_ids, coords_list = zip(*val_batches, strict=True)
    lengths = [len(s) for s in seq_token_ids]
    max_len = max(lengths)
    # Choose the smallest padding size that can fit the longest sequence
    padding_size = next((s for s in PADDING_SIZES if s >= max_len), None)
    if padding_size is None:
        padding_size = max_len

    seq_tok_padded = torch.zeros((len(seq_token_ids), padding_size), dtype=torch.long)
    coords_padded = torch.full(
        (len(coords_list), padding_size, 37, 3), torch.nan, dtype=torch.float32
    )
    for i, length in enumerate(lengths):
        seq_tok_padded[i, :length] = seq_token_ids[i]
        coords_padded[i, :length] = coords_list[i]
    return keys, seq_tok_padded, coords_padded, lengths


@torch.inference_mode()
def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"
    chunk_i = args.chunk
    num_chunk = args.num_chunk

    # === Initialize UniTok ===
    tok = UniTok(UniTok.Config(path=args.ckpt_path, return_attn=False))
    bb_tok = tok.bb_tok.cuda()
    fa_tok = tok.fa_tok.cuda()
    del tok

    # === Set up dataset and dataloader ===
    # Load metadata
    metadata_csv_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(metadata_csv_path, "rb") as f:
        metadata_dicts = msgpack.load(f)

    # Chunk and sort by sequence length
    metadata_dicts = metadata_dicts[chunk_i::num_chunk]
    get_resnum = lambda d: d["chains"][0]["num_residues"]  # noqa
    metadata_dicts = sorted(metadata_dicts, key=get_resnum)
    keys = [d["id"] for d in metadata_dicts]
    print(f"Processing {len(keys)} entries in chunk {chunk_i} of {num_chunk}...")

    struct_lmdb_path = data_dir / "structure.lmdb"
    dataset = Dataset(keys, struct_lmdb_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=SHORT_BATCH_SIZE if args.split == "short" else LONG_BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    out_dir = data_dir / "apo_unitok_chunk/"
    out_dir.mkdir(exist_ok=True)
    out_lmdb_path = out_dir / f"{args.chunk}_{args.num_chunk}.lmdb"
    out_env = lmdb.open(
        str(out_lmdb_path),
        map_size=100 * 1024 * 1024 * 1024,  # 100 GB
        meminit=False,
        map_async=True,
        sync=False,
    )
    txn = out_env.begin(write=True)
    for b_i, (keys, seq_token_id, coords, lengths) in enumerate(
        tqdm(dataloader, desc="Tokenize")
    ):
        if keys is None:
            continue
        seq_token_id = seq_token_id.cuda(non_blocking=True)
        coords = coords.cuda(non_blocking=True)
        if max(lengths) <= BATCH_THRESHOLD:
            bb_tokens = bb_tok.tokenize_batch(coords)
            fa_tokens = fa_tok.tokenize_batch(seq_token_id, coords)
            bb_tokens = bb_tokens.cpu().numpy().astype(np.uint16)
            fa_tokens = fa_tokens.cpu().numpy().astype(np.uint16)
            combined = np.stack([bb_tokens, fa_tokens], axis=-2)
            for k, tokens, length in zip(keys, combined, lengths, strict=True):
                tokens = tokens[:, :length]
                txn.put(k.encode("utf-8"), tokens.tobytes())
        else:
            # To prevent OOM, we tokenize each sample in the batch sequentially
            for i in range(len(keys)):
                length = lengths[i]
                seq_tok_i = seq_token_id[i]
                coords_i = coords[i]
                # keep padded tokens for CUDA efficiency.
                bb_tokens_i = bb_tok.tokenize(coords_i)[:length]
                fa_tokens_i = fa_tok.tokenize(seq_tok_i, coords_i)[:length]
                bb_tokens_i = bb_tokens_i.cpu().numpy().astype(np.uint16)
                fa_tokens_i = fa_tokens_i.cpu().numpy().astype(np.uint16)
                combined_i = np.stack([bb_tokens_i, fa_tokens_i], axis=-2)
                txn.put(keys[i].encode("utf-8"), combined_i.tobytes())

        if (b_i + 1) % 1000 == 0:
            print(f"Processed {b_i + 1} batches, committing transaction...")
            txn.commit()
            txn = out_env.begin(write=True)
    txn.commit()
    out_env.close()


if __name__ == "__main__":
    main()
