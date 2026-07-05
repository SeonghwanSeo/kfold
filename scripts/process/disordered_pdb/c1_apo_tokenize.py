"""Tokenize the apo structures directly from LMDB."""

import argparse
import io
import pathlib

import lmdb
import numpy as np
import torch
from kfold.model.modules.structure_encoder.triprorep import TriProRep, restype_order
from tqdm import tqdm

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
        "--ckpt_path",
        required=True,
        type=pathlib.Path,
        help="Path to structure tokenizer checkpoint",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_chunk",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


class LmdbDataset(torch.utils.data.Dataset):
    """Dataset for processing apo structures directly from LMDB."""

    def __init__(self, lmdb_path: pathlib.Path, keys: list[str]):
        self.lmdb_path = str(lmdb_path)
        self.keys = keys
        self.env = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        # Initialize LMDB environment locally per worker process to avoid pickling issues
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        full_key = self.keys[index]
        # Key format is "apo_type:raw_id"
        raw_id = full_key.split(":", 1)[1]

        try:
            with self.env.begin() as txn:
                value = txn.get(full_key.encode("utf-8"))

            with io.BytesIO(value) as buffer:
                data = np.load(buffer)
                seq_arr = data["seq"]
                coords = data["coords"]

            # Convert numpy 'S1' byte array back to python string
            seq = b"".join(seq_arr).decode("utf-8")
            aatypes = [restype_order.get(res, 0) for res in seq]

            return (
                raw_id,
                torch.tensor(aatypes, dtype=torch.long),
                torch.tensor(coords, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error processing {full_key}: {e}")
            return (raw_id, None, None)


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

    aatypes_padded = torch.zeros((len(seq_token_ids), padding_size), dtype=torch.long)
    coords_padded = torch.full(
        (len(coords_list), padding_size, 37, 3), torch.nan, dtype=torch.float32
    )
    for i, length in enumerate(lengths):
        aatypes_padded[i, :length] = seq_token_ids[i]
        coords_padded[i, :length] = coords_list[i]
    return keys, aatypes_padded, coords_padded, lengths


@torch.inference_mode()
def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb"
    chunk_i = args.chunk
    num_chunk = args.num_chunk

    in_lmdb_path = data_dir / "apo.lmdb"
    assert in_lmdb_path.exists(), f"Input LMDB not found at {in_lmdb_path}"

    out_dir = data_dir / "apo_tok_chunk/"
    out_dir.mkdir(exist_ok=True)

    # Initialize tokenizer
    tok = TriProRep(TriProRep.Config(path=args.ckpt_path))
    bb_tok = tok.bb_tok.cuda()
    fa_tok = tok.fa_tok.cuda()
    del tok

    # Read all keys from the input LMDB and group them by apo_type
    apo_type_to_items = {}

    env_in = lmdb.open(str(in_lmdb_path), readonly=True, lock=False)
    with env_in.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            full_key = k.decode("utf-8")
            apo_type = full_key.split(":")[0]

            if apo_type not in apo_type_to_items:
                apo_type_to_items[apo_type] = []

            # Store (full_key, byte_size) to mimic the previous file size sorting
            apo_type_to_items[apo_type].append((full_key, len(v)))
    env_in.close()
    print(f"Found {len(apo_type_to_items)} apo types: {list(apo_type_to_items.keys())}")

    for apo_type, items in apo_type_to_items.items():
        torch.cuda.empty_cache()
        print(f"Processing {apo_type}...")

        # Sort by value size (small to large) to minimize OOM risk
        items.sort(key=lambda x: x[1])
        keys = [x[0] for x in items]

        # Shard keys for parallel processing
        if args.num_chunk > 1:
            keys = keys[chunk_i::num_chunk]

        if len(keys) == 0:
            print(f"No samples for chunk {chunk_i}/{num_chunk} in {apo_type}. Skipping.")
            continue

        print(
            f"Total samples for {apo_type}: {len(keys)}. "
            f"Processing chunk {chunk_i}/{num_chunk} with {len(keys)} samples."
        )

        dataset = LmdbDataset(in_lmdb_path, keys)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=12,
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Create output lmdb
        out_subdir = out_dir / apo_type
        out_subdir.mkdir(exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
        env_out = lmdb.open(
            str(out_lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,  # 10 GB
            meminit=False,
            map_async=True,
            sync=False,
        )

        with env_out.begin(write=True) as txn:
            for batch in (pbar := tqdm(dataloader, desc=f"Tokenizing {apo_type}")):
                keys, aatypes, coords, lengths = batch

                aatypes = aatypes.to("cuda", non_blocking=True)
                coords = coords.to("cuda", non_blocking=True)

                if max(lengths) <= BATCH_THRESHOLD:
                    # Slice the first 3 atoms for the backbone tokenizer
                    bb_tokens = bb_tok.tokenize_batch(coords[..., :3, :])
                    fa_tokens = fa_tok.tokenize_batch(aatypes, coords)

                    # Save
                    bb_tokens = bb_tokens.cpu().numpy().astype(np.int16)
                    fa_tokens = fa_tokens.cpu().numpy().astype(np.int16)
                    combined = np.stack([bb_tokens, fa_tokens], axis=-2)

                    for k, tokens, length in zip(keys, combined, lengths, strict=True):
                        tokens = tokens[:, :length]
                        txn.put(k.encode("utf-8"), tokens.tobytes())
                else:
                    # To prevent OOM, we tokenize each sample in the batch sequentially
                    for i in range(len(keys)):
                        length = lengths[i]  # Extract length for the current sequence
                        aatypes_i = aatypes[i]
                        coords_i = coords[i]

                        # Slice the first 3 atoms for the backbone tokenizer
                        bb_tokens_i = bb_tok.tokenize(coords_i[..., :3, :])[:length]
                        fa_tokens_i = fa_tok.tokenize(aatypes_i, coords_i)[:length]

                        # Save
                        bb_tokens_i = bb_tokens_i.cpu().numpy().astype(np.int16)
                        fa_tokens_i = fa_tokens_i.cpu().numpy().astype(np.int16)
                        combined_i = np.stack([bb_tokens_i, fa_tokens_i], axis=-2)

                        txn.put(keys[i].encode("utf-8"), combined_i.tobytes())
                pbar.set_postfix({"Last batch max length": max(lengths)})
        env_out.close()


if __name__ == "__main__":
    main()
