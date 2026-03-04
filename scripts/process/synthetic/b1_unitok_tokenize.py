"""Tokenize the apo structures"""

import argparse
import pathlib

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from kfold.constants.sequence import encode_protein_sequence
from kfold.data.utils.io.structure import read_protein_structure
from kfold.model.modules.structure_encoder.unitok import UniTok

PADDING_SIZES = [32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280]
BATCH_THRESHOLD = 1280


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
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

    def __init__(self, files: list[pathlib.Path]):
        self.files: list[pathlib.Path] = files
        # sort by the file size (small to large) to minimize OOM risk
        self.files.sort(key=lambda x: x.stat().st_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        path = self.files[index]
        key = path.name.split(".")[0]
        try:
            seq, coords = read_protein_structure(path)
            seq_tok = encode_protein_sequence(seq)
            return (
                key,
                torch.tensor(seq_tok, dtype=torch.long),
                torch.tensor(coords, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return (key, None, None)


def collate_fn(batch):
    """Collate function to filter out failed samples."""
    keys, seq_token_ids, coords_list = zip(*batch, strict=True)
    seq_token_ids = [s for s in seq_token_ids if s is not None]
    coords_list = [c for c in coords_list if c is not None]
    if len(seq_token_ids) == 0 or len(coords_list) == 0:
        return keys, None, None, None
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
    data_dir: pathlib.Path = args.data_dir / args.name
    chunk_i = args.chunk
    num_chunk = args.num_chunk

    apo_dir = data_dir / "apo/"
    out_dir = data_dir / "apo_unitok_chunk/"
    out_dir.mkdir(exist_ok=True)

    # Initialize UniTok
    tok = UniTok(UniTok.Config(path=args.ckpt_path, return_attn=False))
    bb_tok = tok.bb_tok.cuda()
    fa_tok = tok.fa_tok.cuda()
    del tok

    for apo_subdir in sorted(apo_dir.iterdir()):
        torch.cuda.empty_cache()
        apo_type = apo_subdir.name
        print(f"Processing {apo_subdir} ({apo_type})...")
        files = []
        for suffix in ("*.pdb", "*.pdb.gz", "*.cif", "*.cif.gz"):
            files += sorted(apo_subdir.rglob(suffix))
        if args.num_chunk > 1:
            files = files[chunk_i::num_chunk]  # Shard files for parallel processing

        dataset = Dataset(files)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=12,
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        # Create lmdb
        out_subdir = out_dir / apo_type
        out_subdir.mkdir(exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
        env = lmdb.open(
            str(out_lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,  # 10 GB
            meminit=False,
            map_async=True,
            sync=False,
        )

        with env.begin(write=True) as txn:
            for keys, seq_token_id, coords, lengths in tqdm(
                dataloader, desc=f"Tokenizing {apo_type}"
            ):
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
                        bb_tokens_i = bb_tok.tokenize(coords_i)[:length]
                        fa_tokens_i = fa_tok.tokenize(seq_tok_i, coords_i)[:length]
                        bb_tokens_i = bb_tokens_i.cpu().numpy().astype(np.uint16)
                        fa_tokens_i = fa_tokens_i.cpu().numpy().astype(np.uint16)
                        combined_i = np.stack([bb_tokens_i, fa_tokens_i], axis=-2)
                        txn.put(keys[i].encode("utf-8"), combined_i.tobytes())
        env.close()


if __name__ == "__main__":
    main()
