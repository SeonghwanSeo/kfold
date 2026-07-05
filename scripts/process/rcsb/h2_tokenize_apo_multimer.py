"""Tokenize multimer apo structures directly from apo_multimer_lmdb."""

from __future__ import annotations

import argparse
import pathlib

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from kfold.data.utils.io.apo import (
    pack_apo_multimer_token_record,
    unpack_apo_multimer_record,
)
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
        "--ckpt_path",
        required=True,
        type=pathlib.Path,
        help="Path to structure tokenizer checkpoint.",
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunk", type=int, default=0)
    parser.add_argument("--map_size_gb", type=int, default=10)
    return parser.parse_args()


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
            [restype_order.get(res, 0) for res in seq],
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
    in_root = data_dir / "apo_multimer_lmdb" / "protein"
    if not in_root.exists():
        raise FileNotFoundError(in_root)

    out_dir = data_dir / "apo_tok_chunk" / "protein_multimer"
    out_dir.mkdir(parents=True, exist_ok=True)
    bb_tok, fa_tok = load_tokenizers(args.ckpt_path, device="cuda")

    lmdb_paths = sorted(in_root.glob("*.lmdb"))
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
        if len(keys) == 0:
            print(f"No samples for chunk {args.chunk}/{args.num_chunk} in {source}.")
            continue

        out_subdir = out_dir / source
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_lmdb_path = out_subdir / f"{args.chunk}_{args.num_chunk}.lmdb"
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
        env_in.close()
        env_out.close()


if __name__ == "__main__":
    main()
