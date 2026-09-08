"""Tokenize multimer apo structures directly from apo_lmdb/protein-multimer."""

from __future__ import annotations

import argparse
import pathlib
import shutil

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from kfold.model.layers.struct_enc import BackboneTokenizer, FullAtomTokenizer
from kfold.model.layers.struct_enc.fa_vqvae.utils.residue_constants import (
    restype_order_with_x as restype_order,
)
from kfold.training.dataset.utils.apo_io import (
    pack_apo_multimer_token_record,
    unpack_apo_multimer_record,
)
from kfold.training.preprocess.structure_tokenizers import load_tokenizers

PADDING_SIZES = [32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280]
BATCH_THRESHOLD = 1280


def parse_args() -> argparse.Namespace:
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
        choices=["train", "val", "test"],
        help="Data split to process.",
    )
    parser.add_argument(
        "--cache_dir",
        type=pathlib.Path,
        help="Hugging Face download cache directory.",
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunk", type=int, default=1)
    parser.add_argument("--map_size_gb", type=int, default=10)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.chunk < args.num_chunk:
        parser.error("Require 0 <= chunk < num_chunk")
    return args


def next_padding_size(length: int) -> int:
    return next((size for size in PADDING_SIZES if size >= length), length)


def tokenize_record(
    chains: dict[int, dict],
    bb_tok: BackboneTokenizer,
    fa_tok: FullAtomTokenizer,
) -> dict[int, np.ndarray]:
    chain_ids = sorted(chains)
    lengths = [len(chains[asym_id]["seq"]) for asym_id in chain_ids]
    max_len = max(lengths)
    pad_len = next_padding_size(max_len)

    aatypes = torch.zeros((len(chain_ids), pad_len), dtype=torch.long, device="cuda")
    coords = torch.full(
        (len(chain_ids), pad_len, 37, 3),
        torch.nan,
        dtype=torch.float32,
        device="cuda",
    )
    for i, asym_id in enumerate(chain_ids):
        seq = chains[asym_id]["seq"]
        chain_coords = chains[asym_id]["coords"]
        if chain_coords.shape != (len(seq), 37, 3):
            raise ValueError(
                f"Expected protein atom37 coords for asym_id {asym_id}, got "
                f"{chain_coords.shape} for len={len(seq)}."
            )
        aatypes[i, : len(seq)] = torch.tensor(
            [restype_order[res] for res in seq],
            dtype=torch.long,
            device="cuda",
        )
        coords[i, : len(seq)] = torch.tensor(
            chain_coords,
            dtype=torch.float32,
            device="cuda",
        )

    if max_len <= BATCH_THRESHOLD:
        bb_tokens = bb_tok.tokenize_batch(coords[..., :3, :])
        fa_tokens = fa_tok.tokenize_batch(aatypes, coords)
    else:
        bb_tokens = []
        fa_tokens = []
        for i in range(len(chain_ids)):
            bb_tokens.append(bb_tok.tokenize(coords[i, ..., :3, :]))
            fa_tokens.append(fa_tok.tokenize(aatypes[i], coords[i]))
        bb_tokens = torch.stack(bb_tokens, dim=0)
        fa_tokens = torch.stack(fa_tokens, dim=0)

    bb_tokens = bb_tokens.cpu().numpy().astype(np.int16)
    fa_tokens = fa_tokens.cpu().numpy().astype(np.int16)

    out: dict[int, np.ndarray] = {}
    for i, asym_id in enumerate(chain_ids):
        length = lengths[i]
        out[asym_id] = np.stack([bb_tokens[i, :length], fa_tokens[i, :length]], axis=0)
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    in_root = data_dir / "apo_lmdb" / "protein-multimer"
    if not in_root.exists():
        raise FileNotFoundError(in_root)

    out_dir = data_dir / "apo_tok_chunk" / "protein-multimer"
    out_dir.mkdir(parents=True, exist_ok=True)
    bb_tok, fa_tok = load_tokenizers(cache_dir=args.cache_dir, device="cuda")

    lmdb_paths = sorted(in_root.glob("*.lmdb"))
    if args.sources is not None:
        missing = set(args.sources) - {path.stem for path in lmdb_paths}
        if missing:
            raise FileNotFoundError(f"Requested sources not found: {sorted(missing)}")
        lmdb_paths = [path for path in lmdb_paths if path.stem in args.sources]
    print(
        f"Found {len(lmdb_paths)} protein multimer apo sources: "
        f"{[path.stem for path in lmdb_paths]}"
    )

    for in_lmdb_path in lmdb_paths:
        source = in_lmdb_path.stem
        torch.cuda.empty_cache()
        print(f"Processing {source}...")

        env_in = lmdb.open(str(in_lmdb_path), readonly=True, lock=False)
        with env_in.begin() as txn:
            keys = [key.decode("utf-8") for key, _ in txn.cursor()]
        env_in.close()

        keys.sort()
        if args.num_chunk > 1:
            keys = keys[args.chunk :: args.num_chunk]
        out_subdir = out_dir / source
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
        if out_lmdb_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{out_lmdb_path}; use --overwrite to rebuild this shard"
                )
            shutil.rmtree(out_lmdb_path)
        env_out = lmdb.open(
            str(out_lmdb_path),
            map_size=args.map_size_gb * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        env_in = lmdb.open(str(in_lmdb_path), readonly=True, lock=False)
        with env_out.begin(write=True) as txn_out:
            with env_in.begin() as txn_in:
                for key in tqdm(keys, desc=f"Tokenizing multimer {source}"):
                    value = txn_in.get(key.encode("utf-8"))
                    if value is None:
                        print(f"Missing apo multimer key {key}; skipping.")
                        continue
                    try:
                        chains = unpack_apo_multimer_record(value)
                        tokens = tokenize_record(chains, bb_tok, fa_tok)
                        txn_out.put(
                            key.encode("utf-8"),
                            pack_apo_multimer_token_record(tokens),
                        )
                    except Exception as e:
                        print(f"Error processing {source}:{key}: {e}")
            if txn_out.stat()["entries"] != len(keys):
                raise RuntimeError(
                    f"Incomplete token shard for {source}; see parsing errors above"
                )
        env_in.close()
        env_out.close()


if __name__ == "__main__":
    main()
