"""Inspect interaction pseudo-label density and basic statistics per batch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import kfold.constants as C
from kfold.config import load_config
from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.input_embedder.pretrained_embedder_with_interaction import (
    compute_pair_interactions,
)
from kfold.training.dataset.datamodule import TrainingDataModule

PAIR_TYPE_NAMES = [pair_type.name.lower() for pair_type in C.PairInteractionType]


def _get_nested(config: Any, path: str, default: Any) -> Any:
    """Safely traverse nested config objects/dicts using dot paths."""
    current = config
    for key in path.split("."):
        if current is None:
            return default
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def _build_pair_mask(f_input: FoldingInput, inter_chain_only: bool) -> torch.Tensor:
    """Build a valid-pair mask aligned with interaction pseudo-labels."""
    token_mask = f_input.token.pad_mask & f_input.token.disto_mask
    if token_mask.ndim == 1:
        pair_mask = token_mask[:, None] & token_mask[None, :]
        if inter_chain_only:
            asym_id = f_input.token.asym_id
            pair_mask = pair_mask & (asym_id[:, None] != asym_id[None, :])
    else:
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        if inter_chain_only:
            asym_id = f_input.token.asym_id
            pair_mask = pair_mask & (asym_id[:, :, None] != asym_id[:, None, :])
    pair_mask.diagonal(dim1=-2, dim2=-1).zero_()
    return pair_mask


def _compute_target(
    f_input: FoldingInput, distance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute distance-gated interaction pseudo-labels and auxiliaries."""
    disto_coords = f_input.token.disto_coords.float()
    pdist = torch.cdist(disto_coords, disto_coords)
    within_threshold = pdist <= distance_threshold
    pair_interactions = compute_pair_interactions(
        f_input.token.interaction_type, f_input.token.chain_type
    )
    target = pair_interactions * within_threshold.unsqueeze(-1)
    return target, within_threshold, pair_interactions


def _summarize_batch(
    f_input: FoldingInput,
    meta_infos: list[dict[str, Any]],
    distance_threshold: float,
    inter_chain_only: bool,
    batch_idx: int,
) -> None:
    """Print compact, per-sample interaction pseudo-label statistics."""
    if not f_input.is_batched:
        f_input = FoldingInput.from_list([f_input], pad_to_max=False)
        meta_infos = [meta_infos[0]]

    with torch.no_grad():
        pair_mask = _build_pair_mask(f_input, inter_chain_only)
        target, within_threshold, pair_interactions = _compute_target(
            f_input, distance_threshold
        )
        positive_target = (target > 0.5) & pair_mask.unsqueeze(-1)
        positive_any = (positive_target.sum(-1) > 0) & pair_mask

        valid_tokens = f_input.token.pad_mask.sum(dim=-1)
        tokens_with_type = (
            f_input.token.interaction_type.float().clamp(0.0, 1.0).sum(-1) > 0
        ) & f_input.token.pad_mask
        tokens_with_type = tokens_with_type.sum(dim=-1)

        valid_pairs = pair_mask.sum(dim=(-2, -1))
        pairs_with_type = (pair_interactions.sum(-1) > 0) & pair_mask
        pairs_with_type = pairs_with_type.sum(dim=(-2, -1))
        pairs_within = (within_threshold & pair_mask).sum(dim=(-2, -1))
        pairs_target = positive_any.sum(dim=(-2, -1))

        per_type_counts = positive_target.sum(dim=(-3, -2)).to(torch.long)
        invalid_positive = (target > 0.5) & ~pair_mask.unsqueeze(-1)
        invalid_positive = invalid_positive.sum(dim=(-3, -2, -1)).to(torch.long)

        raw_min = int(f_input.token.interaction_type.min().item())
        raw_max = int(f_input.token.interaction_type.max().item())

    print(f"\n[batch {batch_idx}]")
    print(
        f"- interaction_type min/max: {raw_min}/{raw_max}, "
        f"distance_threshold: {distance_threshold}, "
        f"inter_chain_only: {inter_chain_only}"
    )

    batch_size = int(f_input.batch_size) if f_input.is_batched else 1
    for i in range(batch_size):
        sample_id = meta_infos[i].get("id", f"sample_{i}")
        counts = per_type_counts[i].tolist()
        counts_str = ", ".join(
            f"{name}={count}" for name, count in zip(PAIR_TYPE_NAMES, counts, strict=True)
        )
        print(
            f"- {sample_id}: tokens={int(valid_tokens[i])}, "
            f"tokens_with_type={int(tokens_with_type[i])}, "
            f"valid_pairs={int(valid_pairs[i])}, "
            f"pairs_with_type={int(pairs_with_type[i])}, "
            f"pairs_within_threshold={int(pairs_within[i])}, "
            f"pairs_target_any={int(pairs_target[i])}"
        )
        print(f"  target_per_type: {counts_str}")
        if int(invalid_positive[i]) > 0:
            print(f"  WARNING: positives on invalid pairs = {int(invalid_positive[i])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect interaction pseudo labels from a dataloader batch."
    )
    parser.add_argument("--config", type=str, default="configs/train-af3-tiny.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument(
        "--inter-chain-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf dotlist overrides (e.g., train.data.ccd_path=/path).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config), override_args=args.override)
    data_cfg = config.train.data

    if args.batch_size is not None:
        if args.split == "train":
            data_cfg.train_batch_size = args.batch_size
        else:
            data_cfg.val_batch_size = args.batch_size

    data_cfg.num_workers = 0
    data_cfg.persistent_workers = False
    data_cfg.pin_memory = False

    distance_threshold = args.distance_threshold
    if distance_threshold is None:
        distance_threshold = float(
            _get_nested(config, "train.loss.interaction_loss.distance_threshold", 7.5)
        )

    inter_chain_only = args.inter_chain_only
    if inter_chain_only is None:
        inter_chain_only = bool(
            _get_nested(config, "train.loss.interaction_loss.inter_chain_only", False)
        )

    dm = TrainingDataModule(data_cfg)
    if args.split == "train":
        dm.setup("fit")
        dataloader = dm.train_dataloader()
    else:
        dm.setup("validate")
        dataloader = dm.val_dataloader()

    for batch_idx, (f_input, meta_infos) in enumerate(dataloader):
        _summarize_batch(
            f_input,
            meta_infos,
            distance_threshold=distance_threshold,
            inter_chain_only=inter_chain_only,
            batch_idx=batch_idx,
        )
        if batch_idx + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    main()
