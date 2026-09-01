"""Export a Lightning affinity checkpoint as a structure-style inference weight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from kfold.inference.affinity import (
    affinity_head_state_dict,
    load_affinity_head,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Lightning .ckpt or direct state dict")
    parser.add_argument("output", type=Path, help="Output inference .pth")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.input,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state_dict = affinity_head_state_dict(checkpoint)
    stage = args.output.with_name(f".{args.output.stem}.incomplete.pth")
    torch.save(state_dict, stage)
    load_affinity_head(stage, device="cpu")
    os.replace(stage, args.output)
    print(f"Exported {len(state_dict)} tensors to {args.output}")
    print(f"sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
