"""Tokenize JASPAR protein apo structures from source-specific apo LMDBs."""

import argparse
import pathlib
import sys

import lmdb
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(".")

from scripts.process.rcsb.h1_tokenize_apo_monomer import (  # noqa: E402
    BATCH_THRESHOLD,
    LmdbDataset,
    collate_fn,
    load_tokenizers,
)

DATASET_NAME = "JASPAR"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The JASPAR/ folder is appended.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=pathlib.Path,
        help="Path to structure tokenizer checkpoint.",
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunk", type=int, default=0)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional apo source names to tokenize.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / DATASET_NAME

    in_root = data_dir / "apo_lmdb" / "protein"
    if not in_root.exists():
        raise FileNotFoundError(f"Input protein apo LMDB directory not found: {in_root}")

    out_dir = data_dir / "apo_tok_chunk" / "protein"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        items.sort(key=lambda x: x[1])
        keys = [x[0] for x in items]

        if args.num_chunk > 1:
            keys = keys[args.chunk :: args.num_chunk]
        if len(keys) == 0:
            print(f"No samples for chunk {args.chunk}/{args.num_chunk} in {source}.")
            continue

        print(
            f"Total samples for {source}: {len(items)}. "
            f"Processing chunk {args.chunk}/{args.num_chunk} with {len(keys)} samples."
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

        out_subdir = out_dir / source
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
        env_out = lmdb.open(
            str(out_lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,
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
                    bb_tokens = bb_tok.tokenize_batch(coords[..., :3, :])
                    fa_tokens = fa_tok.tokenize_batch(aatypes, coords)
                    bb_tokens = bb_tokens.cpu().numpy().astype(np.int16)
                    fa_tokens = fa_tokens.cpu().numpy().astype(np.int16)
                    combined = np.stack([bb_tokens, fa_tokens], axis=-2)
                    for k, tokens, length in zip(keys, combined, lengths, strict=True):
                        txn.put(k.encode("utf-8"), tokens[:, :length].tobytes())
                else:
                    for i in range(len(keys)):
                        length = lengths[i]
                        coords_i = coords[i]
                        aatypes_i = aatypes[i]
                        bb_tokens_i = bb_tok.tokenize(coords_i[..., :3, :])[:length]
                        fa_tokens_i = fa_tok.tokenize(aatypes_i, coords_i)[:length]
                        combined_i = np.stack(
                            [
                                bb_tokens_i.cpu().numpy().astype(np.int16),
                                fa_tokens_i.cpu().numpy().astype(np.int16),
                            ],
                            axis=-2,
                        )
                        txn.put(keys[i].encode("utf-8"), combined_i.tobytes())
                pbar.set_postfix({"Last batch max length": max(lengths)})
        env_out.close()


if __name__ == "__main__":
    main()
