"""Tokenize the apo structures directly from LMDB."""

import argparse
import pathlib

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from kfold.data.utils.io.apo import unpack_apo_record
from kfold.model.layers.struct_enc import BackboneTokenizer, FullAtomTokenizer
from kfold.model.modules.structure_encoder import restype_order

PADDING_SIZES = [32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280]
BATCH_THRESHOLD = 1280


def load_tokenizers(
    ckpt_path: pathlib.Path,
    device: torch.device | str = "cuda",
) -> tuple[BackboneTokenizer, FullAtomTokenizer]:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    bb_state_dict = {
        k.removeprefix("bb_tok."): v
        for k, v in state_dict.items()
        if k.startswith("bb_tok.")
    }
    fa_state_dict = {
        k.removeprefix("fa_tok."): v
        for k, v in state_dict.items()
        if k.startswith("fa_tok.")
    }
    if not bb_state_dict or not fa_state_dict:
        raise KeyError(
            "Expected a standalone StructureEncoder checkpoint containing "
            "'bb_tok.' and 'fa_tok.' weights."
        )

    bb_tok = BackboneTokenizer().to(torch.bfloat16)
    fa_tok = FullAtomTokenizer().to(torch.bfloat16)
    bb_tok.load_state_dict(bb_state_dict, strict=True)
    fa_tok.load_state_dict(fa_state_dict, strict=True)
    return bb_tok.eval().to(device), fa_tok.eval().to(device)


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
        choices=["train", "val", "test"],
        help="Data split to process (train/val/test).",
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
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional apo source names to tokenize, e.g. esmfold afdb.",
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

        raw_id = self.keys[index]

        try:
            with self.env.begin() as txn:
                value = txn.get(raw_id.encode("utf-8"))

            record = unpack_apo_record(value)
            seq = record["seq"]
            coords = record["coords"]
            chain_type = record.get("chain_type", "protein")
            if chain_type != "protein" or coords.shape[1:] != (37, 3):
                raise ValueError(
                    f"Expected protein atom37 record, got chain_type={chain_type}, "
                    f"coords_shape={coords.shape}"
                )

            aatypes = [restype_order.get(res, 0) for res in seq]

            return (
                raw_id,
                torch.tensor(aatypes, dtype=torch.long),
                torch.tensor(coords, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error processing {raw_id}: {e}")
            return (raw_id, None, None)


def collate_fn(batch):
    """Collate function to filter out failed samples."""
    batch = [item for item in batch if item[1] is not None and item[2] is not None]
    if len(batch) == 0:
        return [], None, None, None
    keys, seq_token_ids, coords_list = zip(*batch, strict=True)
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
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"
    chunk_i = args.chunk
    num_chunk = args.num_chunk

    in_root = data_dir / "apo_lmdb" / "protein"
    assert in_root.exists(), f"Input protein apo LMDB directory not found at {in_root}"

    out_dir = data_dir / "apo_tok_chunk" / "protein"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    bb_tok, fa_tok = load_tokenizers(args.ckpt_path, device="cuda")

    lmdb_paths = sorted(in_root.glob("*.lmdb"))
    if args.sources is not None:
        requested_sources = set(args.sources)
        lmdb_paths = [path for path in lmdb_paths if path.stem in requested_sources]
        found_sources = {path.stem for path in lmdb_paths}
        missing_sources = sorted(requested_sources - found_sources)
        if missing_sources:
            raise FileNotFoundError(
                f"Requested apo source LMDBs not found under {in_root}: {missing_sources}"
            )
    print(f"Found {len(lmdb_paths)} protein apo sources: {[p.stem for p in lmdb_paths]}")

    for in_lmdb_path in lmdb_paths:
        source = in_lmdb_path.stem
        torch.cuda.empty_cache()
        print(f"Processing {source}...")

        items = []
        env_in = lmdb.open(str(in_lmdb_path), readonly=True, lock=False)
        with env_in.begin() as txn:
            for k, v in txn.cursor():
                items.append((k.decode("utf-8"), len(v)))
        env_in.close()
        # Sort by value size (small to large) to minimize OOM risk
        items.sort(key=lambda x: x[1])
        keys = [x[0] for x in items]

        # Shard keys for parallel processing
        if args.num_chunk > 1:
            keys = keys[chunk_i::num_chunk]

        if len(keys) == 0:
            print(f"No samples for chunk {chunk_i}/{num_chunk} in {source}. Skipping.")
            continue

        print(
            f"Total samples for {source}: {len(keys)}. "
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
        out_subdir = out_dir / source
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
        env_out = lmdb.open(
            str(out_lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,  # 10 GB
            meminit=False,
            map_async=True,
            sync=False,
        )

        with env_out.begin(write=True) as txn:
            for batch in (pbar := tqdm(dataloader, desc=f"Tokenizing {source}")):
                keys, aatypes, coords, lengths = batch
                if aatypes is None or coords is None:
                    continue

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
