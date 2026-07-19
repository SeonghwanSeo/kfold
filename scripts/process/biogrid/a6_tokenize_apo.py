"""Tokenize BioGRID protein apo LMDBs into chunk-level LMDBs."""

import argparse
import pathlib
import sys

import lmdb
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.process.rcsb.h1_tokenize_apo_monomer import (  # noqa: E402
    BATCH_THRESHOLD,
    LmdbDataset,
    collate_fn,
    load_tokenizers,
)

DATASET_NAME = "Biogrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--ckpt_path", required=True, type=pathlib.Path)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunk", type=int, default=1)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=12)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_chunk < 1 or not 0 <= args.chunk < args.num_chunk:
        raise ValueError("Require num_chunk >= 1 and 0 <= chunk < num_chunk.")
    data_dir = args.data_dir / DATASET_NAME
    in_root = data_dir / "apo_lmdb" / "protein"
    if not in_root.exists():
        raise FileNotFoundError(in_root)
    out_root = data_dir / "apo_tok_chunk" / "protein"
    out_root.mkdir(parents=True, exist_ok=True)

    bb_tok, fa_tok = load_tokenizers(args.ckpt_path, device="cuda")
    lmdb_paths = sorted(in_root.glob("*.lmdb"))
    if args.sources is not None:
        requested = set(args.sources)
        lmdb_paths = [path for path in lmdb_paths if path.stem in requested]
        missing = requested - {path.stem for path in lmdb_paths}
        if missing:
            raise FileNotFoundError(f"Missing apo LMDB sources: {sorted(missing)}")

    for in_lmdb_path in lmdb_paths:
        source = in_lmdb_path.stem
        torch.cuda.empty_cache()
        items = []
        env_in = lmdb.open(str(in_lmdb_path), readonly=True, lock=False)
        with env_in.begin() as txn:
            for key, value in txn.cursor():
                items.append((key.decode("utf-8"), len(value)))
        env_in.close()
        items.sort(key=lambda item: item[1])
        keys = [item[0] for item in items][args.chunk :: args.num_chunk]
        if not keys:
            print(f"No samples for {source} chunk {args.chunk}/{args.num_chunk}")
            continue

        dataset = LmdbDataset(in_lmdb_path, keys)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        out_dir = out_root / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.chunk}_{args.num_chunk}.lmdb"
        env_out = lmdb.open(
            str(out_path),
            map_size=10 * 1024**3,
            meminit=False,
            map_async=True,
            sync=False,
        )
        with env_out.begin(write=True) as txn:
            for batch in (pbar := tqdm(dataloader, desc=f"Tokenizing {source}")):
                batch_keys, aatypes, coords, lengths = batch
                if aatypes is None or coords is None:
                    continue
                aatypes = aatypes.to("cuda", non_blocking=True)
                coords = coords.to("cuda", non_blocking=True)
                if max(lengths) <= BATCH_THRESHOLD:
                    bb_tokens = bb_tok.tokenize_batch(coords[..., :3, :])
                    fa_tokens = fa_tok.tokenize_batch(aatypes, coords)
                    combined = np.stack(
                        [
                            bb_tokens.cpu().numpy().astype(np.int16),
                            fa_tokens.cpu().numpy().astype(np.int16),
                        ],
                        axis=-2,
                    )
                    for key, tokens, length in zip(
                        batch_keys, combined, lengths, strict=True
                    ):
                        txn.put(key.encode(), tokens[:, :length].tobytes())
                else:
                    for i, key in enumerate(batch_keys):
                        length = lengths[i]
                        bb_i = bb_tok.tokenize(coords[i, ..., :3, :])[:length]
                        fa_i = fa_tok.tokenize(aatypes[i], coords[i])[:length]
                        combined_i = np.stack(
                            [
                                bb_i.cpu().numpy().astype(np.int16),
                                fa_i.cpu().numpy().astype(np.int16),
                            ],
                            axis=-2,
                        )
                        txn.put(key.encode(), combined_i.tobytes())
                pbar.set_postfix({"max_length": max(lengths)})
        env_out.close()


if __name__ == "__main__":
    main()
