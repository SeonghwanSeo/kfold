"""Tokenize the apo structures"""

import argparse
import io
import pathlib

import lmdb
import msgpack
import numpy as np
import torch
from tqdm import tqdm

from kfold.constants.sequence import encode_protein_sequence
from kfold.model.modules.structure_encoder.unitok import UniTok

PADDING_SIZES = [32, 64, 128, 200, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560]
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
        "--num_chunks",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


class LMDBDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        keys: list[str],
        lmdb_path: str | pathlib.Path,
        max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
    ):
        self.keys: list[str] = keys
        self._lmdb_path = str(lmdb_path)
        self._lmdb_env = None
        self.max_tokens_per_batch = max_tokens_per_batch

    def __iter__(self):
        env = lmdb.open(self._lmdb_path, readonly=True, lock=False, readahead=False)
        txn = env.begin()
        batch: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for k in self.keys:
            # fetch esmfold structure first
            has_apo = False
            for prefix in ["esmfold", "afdb"]:
                _k = f"{prefix}:{k}"
                bytes_value = txn.get(_k.encode("utf-8"))
                if bytes_value is None:
                    continue

                try:
                    with io.BytesIO(bytes_value) as buffer:
                        with np.load(buffer) as npz:
                            seq = npz["seq"].tobytes().decode("ascii")  # S1 -> string
                            coords = npz["coords"]
                except Exception as e:
                    print(f"Error loading structure for key {_k}: {e}")
                    continue

                has_apo = True

                seq_tok = encode_protein_sequence(seq)
                seq_t = torch.tensor(seq_tok, dtype=torch.long)
                coords_t = torch.tensor(coords, dtype=torch.float32)
                next_sample = (_k, seq_t, coords_t)
                if (
                    len(batch) > 0
                    and self.calc_batch_tokens(batch, next_sample)
                    > self.max_tokens_per_batch
                ):
                    yield self.to_batch(batch)
                    batch = []
                batch.append(next_sample)
            if not has_apo:
                print(
                    f"Warning: No structure found for key {k} "
                    f"with prefixes 'esmfold' or 'afdb'."
                )

        if batch:
            yield self.to_batch(batch)

        txn.abort()  # close the transaction when done
        env.close()  # close the environment when done

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
    data_dir: pathlib.Path = args.data_dir / f"afdb-{args.split}"
    chunk_i = args.chunk
    num_chunks = args.num_chunks

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
    get_resnum = lambda d: (d["chains"][0]["num_residues"], d["id"])  # noqa
    metadata_dicts = sorted(metadata_dicts, key=get_resnum)
    metadata_dicts = metadata_dicts[chunk_i::num_chunks]
    entries = [d["id"] for d in metadata_dicts]
    print(f"Processing {len(entries)} entries in chunk {chunk_i} of {num_chunks}...")

    apo_lmdb_path = data_dir / "apo.lmdb"
    dataset = LMDBDataset(entries, apo_lmdb_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    out_dir = data_dir / "apo_unitok_chunk/"
    out_dir.mkdir(exist_ok=True)
    out_lmdb_path = out_dir / f"{args.chunk}_{args.num_chunks}.lmdb"
    out_env = lmdb.open(
        str(out_lmdb_path),
        map_size=100 * 1024 * 1024 * 1024,  # 100 GB
        meminit=False,
        map_async=True,
        sync=False,
    )

    pbar = tqdm(
        total=len(entries),
        desc=f"Tokenizing chunk {chunk_i}/{num_chunks}",
    )
    txn = out_env.begin(write=True)
    for b_i, (keys, seq_token_id, coords, lengths) in enumerate(dataloader):
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
        pbar.update(len(keys))

        if (b_i + 1) % 1000 == 0:
            print(f"Processed {b_i + 1} batches, committing transaction...")
            txn.commit()
            txn = out_env.begin(write=True)
    txn.commit()
    out_env.close()


if __name__ == "__main__":
    main()
