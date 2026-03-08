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

PADDING_SIZES = [32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048]
BATCH_THRESHOLD = 2560
MAX_TOKENS_PER_BATCH = 2560 * 6


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
        "--num_chunks",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        files: list[tuple[str, pathlib.Path]],
        max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
    ):
        self.files: list[tuple[str, pathlib.Path]] = files
        # sort by the file size (small to large) to minimize OOM risk
        self.files.sort(key=lambda x: x[1].stat().st_size)
        self.max_tokens_per_batch = max_tokens_per_batch

    def __iter__(self):
        batch: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for index in range(len(self.files)):
            apo_type, path = self.files[index]
            key = path.name.split(".")[0]
            key = f"{apo_type}:{key}"
            try:
                seq, coords = read_protein_structure(path)
                seq_tok = encode_protein_sequence(seq)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            coords_t = torch.tensor(coords, dtype=torch.float32)
            seq_tok_t = torch.tensor(seq_tok, dtype=torch.long)
            next_sample = (key, seq_tok_t, coords_t)

            if (
                len(batch) > 0
                and self.calc_batch_tokens(batch, next_sample) > self.max_tokens_per_batch
            ):
                yield self.to_batch(batch)
                batch = []

            batch.append(next_sample)

        if batch:
            yield self.to_batch(batch)

    def calc_batch_tokens(
        self,
        samples: list[tuple[str, torch.Tensor, torch.Tensor]],
        next_sample: tuple[str, torch.Tensor, torch.Tensor] | None = None,
    ) -> int:
        """Compute the total number of tokens in the batch."""
        max_seq_len = max((s.shape[0] for _, s, _ in samples), default=0)
        n_samples = len(samples)
        if next_sample is not None:
            max_seq_len = max(max_seq_len, next_sample[1].shape[0])
            n_samples += 1
        # consider padding
        padding_size = next((s for s in PADDING_SIZES if s >= max_seq_len), None)
        if padding_size is not None:
            max_seq_len = padding_size
        return n_samples * max_seq_len

    def to_batch(self, samples: list[tuple[str, torch.Tensor, torch.Tensor]]):
        """Convert a single sample to a batch format."""
        keys, seq_token_ids, coords_list = zip(*samples, strict=True)
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
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    chunk_i = args.chunk
    num_chunks = args.num_chunks

    apo_dir = data_dir / "apo/"
    out_dir = data_dir / "apo_unitok_chunk/"
    out_dir.mkdir(exist_ok=True)

    # Initialize UniTok
    tok = UniTok(UniTok.Config(path=args.ckpt_path, return_attn=False))
    bb_tok = tok.bb_tok.cuda()
    fa_tok = tok.fa_tok.cuda()
    del tok

    files: list[tuple[str, pathlib.Path]] = []
    for apo_subdir in sorted(apo_dir.iterdir()):
        torch.cuda.empty_cache()
        apo_type = apo_subdir.name
        print(f"Processing {apo_subdir} ({apo_type})...")
        apo_files: list[pathlib.Path] = []
        for suffix in ("*.pdb", "*.pdb.gz", "*.cif", "*.cif.gz"):
            apo_files += apo_subdir.rglob(suffix)
        print(f"Found {len(apo_files)} files for {apo_type}.")
        files += [(apo_type, f) for f in apo_files]
    files.sort()
    print(f"Total {len(files)} files found across all apo types.")
    if num_chunks > 1:
        files = files[chunk_i::num_chunks]  # Shard files for parallel processing
        print(f"Processing {len(files)} files for chunk {chunk_i}/{num_chunks}.")

    # Create dataset and dataloader
    dataset = Dataset(files)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Create lmdb
    out_lmdb_path = out_dir / f"{args.chunk}_{num_chunks}.lmdb"
    env = lmdb.open(
        str(out_lmdb_path),
        map_size=1 * 1024 * 1024 * 1024,  # 0 GB
        meminit=False,
        map_async=True,
        sync=False,
    )

    pbar = tqdm(total=len(files), desc="Tokenizing")
    with env.begin(write=True) as txn:
        for keys, seq_token_id, coords, lengths in dataloader:
            if seq_token_id is None or coords is None:
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
                    k_i = keys[i]
                    length_i = lengths[i]
                    seq_tok_i = seq_token_id[i][:length_i]
                    coords_i = coords[i][:length_i]
                    bb_tokens_i = bb_tok.tokenize(coords_i)
                    fa_tokens_i = fa_tok.tokenize(seq_tok_i, coords_i)
                    bb_tokens_i = bb_tokens_i.cpu().numpy().astype(np.uint16)
                    fa_tokens_i = fa_tokens_i.cpu().numpy().astype(np.uint16)
                    combined_i = np.stack([bb_tokens_i, fa_tokens_i], axis=-2)
                    txn.put(k_i.encode("utf-8"), combined_i.tobytes())
            pbar.update(len(keys))
    env.close()


if __name__ == "__main__":
    main()
