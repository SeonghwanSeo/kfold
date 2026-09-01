#!/usr/bin/env python3
"""Compare submitted cache-backed CASP crops with on-the-fly inference inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

import kfold.constants as C
from kfold.inference.affinity import (
    AFFINITY_QUERY_WINDOW_CONTRACT_V1,
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
    build_per_query_affinity_inputs,
    load_affinity_head,
    sha256_file,
)
from kfold.training.affinity.cache import FeatureCacheReader
from kfold.training.affinity.crop import select_pocket_annotation_crop
from kfold.training.affinity.pair_storage import (
    crop_full_cross_payload,
    full_cross_pl_distogram_profile,
    unpack_full_cross_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--benchmark-prefix", default="casp16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-protein-tokens", type=int, default=200)
    parser.add_argument("--neighborhood-size", type=int, default=10)
    return parser.parse_args()


def _head_inputs(crop: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    token_mask = crop["token_mask"].bool()
    chain_type = crop["chain_type"].long()
    return {
        "s_inputs": crop["s_inputs"].float()[None],
        "s_lm": crop["s_lm"].float()[None],
        "z": crop["z"].float()[None],
        "distogram_features": crop["distogram_features"].float()[None],
        "token_mask": token_mask[None],
        "protein_mask": (token_mask & (chain_type == C.ChainType.PROTEIN.value))[None],
        "ligand_mask": (token_mask & (chain_type == C.ChainType.LIGAND.value))[None],
    }


def _to_device(
    values: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in values.items()}


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.dtype == torch.bool or right.dtype == torch.bool:
        return 0.0 if torch.equal(left, right) else float("inf")
    return float((left.float() - right.float()).abs().max().item())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite equivalence report: {args.output}")
    if args.limit <= 0 or args.limit > 10:
        raise ValueError("--limit must be in [1, 10].")
    config = PerQueryAffinityConfig(
        max_tokens=args.max_tokens,
        max_protein_tokens=args.max_protein_tokens,
        neighborhood_size=args.neighborhood_size,
        cache_compatible_bfloat16=True,
    )
    rows = pq.read_table(args.manifest).to_pylist()
    if args.benchmark_prefix:
        prefix = args.benchmark_prefix.lower()
        rows = [
            row
            for row in rows
            if str(row.get("benchmark", "")).lower().startswith(prefix)
        ]
    rows = sorted(rows, key=lambda row: str(row["record_id"]))[: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"Requested {args.limit} records but selected only {len(rows)}.")

    device = torch.device(args.device)
    head = load_affinity_head(args.head_checkpoint, device=device)
    predictor = PerQueryAffinityPredictor(
        head,
        checkpoint_sha256=sha256_file(args.head_checkpoint),
        config=config,
    ).eval()
    reader = FeatureCacheReader(args.cache_root)
    results: list[dict[str, object]] = []
    tensor_names = (
        "s_inputs",
        "s_lm",
        "z",
        "distogram_features",
        "token_mask",
        "protein_mask",
        "ligand_mask",
    )
    with torch.inference_mode():
        for row in rows:
            arrays = reader.get(
                {
                    "shard": str(row["cache_shard"]),
                    "key": str(row["cache_key"]),
                    "cache_encoding": row.get("cache_encoding"),
                }
            )
            profile = full_cross_pl_distogram_profile(arrays)
            cached_indices = select_pocket_annotation_crop(
                token_mask=torch.from_numpy(arrays["token_mask"]).bool(),
                chain_type=torch.from_numpy(arrays["chain_type"]).long(),
                protein_min_distance=profile.protein_min_expected_distance,
                max_tokens=config.max_tokens,
                max_protein_tokens=config.max_protein_tokens,
                neighborhood_size=config.neighborhood_size,
                require_contiguous_monomer=True,
            )
            cached_crop = crop_full_cross_payload(
                arrays,
                max_tokens=config.max_tokens,
                max_protein_tokens=config.max_protein_tokens,
                crop_indices=cached_indices,
            )
            cached_inputs = _head_inputs(cached_crop)
            full = unpack_full_cross_payload(arrays, include_logits=True)
            full_device = {
                name: full[name].to(device)
                for name in (
                    "s_inputs",
                    "s_lm",
                    "z",
                    "distogram_logits",
                    "token_mask",
                    "chain_type",
                )
            }
            online = predictor(**full_device)
            built = build_per_query_affinity_inputs(**full_device, config=config)
            built_inputs = built.head_kwargs()
            cached_device = _to_device(cached_inputs, device)
            cached_prediction = head(**cached_device).float()
            differences = {
                name: _maximum_difference(cached_device[name], built_inputs[name])
                for name in tensor_names
            }
            prediction_difference = abs(
                float(cached_prediction.item()) - float(online["p_activity"].item())
            )
            passed = (
                torch.equal(cached_indices.to(device), online["crop_indices"])
                and max(differences.values()) == 0.0
                and prediction_difference == 0.0
            )
            results.append(
                {
                    "record_id": str(row["record_id"]),
                    "benchmark": str(row.get("benchmark")),
                    "system_id": str(row["system_id"]),
                    "crop_indices_equal": torch.equal(
                        cached_indices.to(device), online["crop_indices"]
                    ),
                    "crop_token_count": len(cached_indices),
                    "tensor_max_abs_difference": differences,
                    "cached_prediction_p_activity": float(cached_prediction.item()),
                    "online_prediction_p_activity": float(online["p_activity"].item()),
                    "prediction_abs_difference": prediction_difference,
                    "passed": passed,
                }
            )

    report = {
        "schema_version": "affinity_online_equivalence_report_v1",
        "state": "complete" if all(row["passed"] for row in results) else "failed",
        "comparison": "submitted_cache_crop_vs_on_the_fly_adapter",
        "crop_contract": AFFINITY_QUERY_WINDOW_CONTRACT_V1,
        "manifest": str(args.manifest.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "head_checkpoint": str(args.head_checkpoint.resolve()),
        "head_checkpoint_sha256": predictor.checkpoint_sha256,
        "config": {
            "max_tokens": config.max_tokens,
            "max_protein_tokens": config.max_protein_tokens,
            "neighborhood_size": config.neighborhood_size,
            "cache_compatible_bfloat16": config.cache_compatible_bfloat16,
        },
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["state"] != "complete":
        raise SystemExit("Online affinity equivalence failed.")


if __name__ == "__main__":
    main()
