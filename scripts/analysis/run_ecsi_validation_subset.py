#!/usr/bin/env python3
"""Run validation on a subset of the validation manifest with config overrides."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import msgpack
import torch
from omegaconf import OmegaConf

from kfold.config import load_config
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.training_module import KFoldTrainingModule

DEFAULT_CASES = [
    "baseline:ode_time_duration=0.6,sampling_schedule_ode_fraction=0.40",
    "ode050_frac030:ode_time_duration=0.5,sampling_schedule_ode_fraction=0.30",
    "ode040_frac025:ode_time_duration=0.4,sampling_schedule_ode_fraction=0.25",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ECSI validation on a manifest subset with override cases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base training config path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path used for validation.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Root directory for subset manifests, per-case outputs, and summaries.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=200,
        help="Number of validation entries to keep from the manifest head.",
    )
    parser.add_argument(
        "--source_manifest",
        type=Path,
        default=None,
        help=(
            "Optional source manifest path. Defaults to the validation dataset manifest."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use inside the visible CUDA device set.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Validation dataloader workers per rank.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of diffusion steps for validation.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=3,
        help="Number of recycles for validation.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case specification in the form "
            "'name:key=value,key2=value2'. Bare keys are applied under "
            "'model.structure_module'."
        ),
    )
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_case_spec(spec: str) -> tuple[str, dict[str, Any]]:
    try:
        name, payload = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid case spec {spec!r}. Expected 'name:key=value,...'."
        ) from exc

    overrides: dict[str, Any] = {}
    for item in payload.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, raw_value = item.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid override {item!r} in case {name!r}. Expected key=value."
            ) from exc
        full_key = key.strip()
        if "." not in full_key:
            full_key = f"model.structure_module.{full_key}"
        overrides[full_key] = parse_scalar(raw_value.strip())

    if not overrides:
        raise ValueError(f"Case {name!r} does not contain any overrides.")

    return name.strip(), overrides


def load_manifest_entries(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".msgpack":
        with open(path, "rb") as file:
            return msgpack.unpack(file)
    with open(path) as file:
        return json.load(file)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        tmp_file.write(content)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def atomic_write_msgpack(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp_file:
        msgpack.pack(payload, tmp_file)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def resolve_source_manifest(args: argparse.Namespace, cfg) -> Path:
    if args.source_manifest is not None:
        return args.source_manifest.resolve()

    val_dataset = cfg.train.data.val_datasets[0]
    manifest_path = getattr(val_dataset, "manifest_path", None)
    if manifest_path is not None:
        return Path(manifest_path).resolve()

    return (Path(val_dataset.data_path) / "manifest.msgpack").resolve()


def prepare_subset_manifest(
    source_manifest: Path,
    subset_size: int,
    out_root: Path,
) -> tuple[Path, Path]:
    entries = load_manifest_entries(source_manifest)
    subset = entries[:subset_size]
    manifest_dir = out_root / "manifests"
    msgpack_path = manifest_dir / f"first_{subset_size}.msgpack"
    json_path = manifest_dir / f"first_{subset_size}.json"
    atomic_write_msgpack(msgpack_path, subset)
    atomic_write_text(json_path, json.dumps(subset, indent=2))
    return msgpack_path.resolve(), json_path.resolve()


def save_case_config(case_dir: Path, cfg) -> None:
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    atomic_write_text(case_dir / "resolved_config.yaml", config_text)


def load_checkpoint_state_dict(
    checkpoint_path: Path,
    *,
    use_ema: bool = True,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" not in checkpoint:
        return checkpoint
    if use_ema and "ema" in checkpoint:
        return checkpoint["ema"]["shadow_params"]
    return checkpoint["state_dict"]


def run_case(
    *,
    base_cfg_path: Path,
    checkpoint_path: Path,
    subset_manifest_path: Path,
    case_dir: Path,
    case_name: str,
    overrides: dict[str, Any],
    num_gpus: int,
    num_workers: int,
    num_steps: int,
    num_recycles: int,
) -> dict[str, Any]:
    cfg = load_config(base_cfg_path)

    cfg.train.data.val_datasets[0].manifest_path = str(subset_manifest_path)
    cfg.train.data.num_workers = num_workers
    cfg.train.validation.num_steps = num_steps
    cfg.train.validation.num_recycles = num_recycles
    cfg.train.validation.save_predictions = False
    cfg.train.validation.return_traj = False

    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)

    save_case_config(case_dir, cfg)

    pl.seed_everything(cfg.train.seed, workers=False)
    torch.set_float32_matmul_precision("high")

    model_module = KFoldTrainingModule(cfg)
    state_dict = load_checkpoint_state_dict(
        checkpoint_path,
        use_ema=cfg.train.optimizer.use_ema,
    )
    model_module.load_state_dict(state_dict, strict=False)
    data_module = TrainingDataModule(cfg.train.data)

    trainer_strategy: str | Any
    trainer_strategy = "ddp" if num_gpus > 1 else cfg.train.trainer.strategy
    trainer = pl.Trainer(
        default_root_dir=str(case_dir),
        logger=False,
        devices=num_gpus,
        accelerator=cfg.train.trainer.accelerator,
        strategy=trainer_strategy,
        precision=cfg.train.trainer.precision,
        deterministic=True,
        enable_checkpointing=False,
    )

    results = trainer.validate(
        model_module,
        datamodule=data_module,
        ckpt_path=None,
    )
    if len(results) != 1:
        raise RuntimeError(
            f"Expected a single validation result for case {case_name!r}, got {results}."
        )
    return dict(results[0])


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    case_specs = args.case or DEFAULT_CASES
    parsed_cases = [parse_case_spec(spec) for spec in case_specs]

    base_cfg = load_config(args.config)
    source_manifest = resolve_source_manifest(args, base_cfg)
    subset_msgpack, subset_json = prepare_subset_manifest(
        source_manifest=source_manifest,
        subset_size=args.subset_size,
        out_root=args.out_root,
    )

    run_summary: dict[str, Any] = {
        "base_config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "source_manifest": str(source_manifest),
        "subset_manifest_msgpack": str(subset_msgpack),
        "subset_manifest_json": str(subset_json),
        "subset_size": args.subset_size,
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_steps": args.num_steps,
        "num_recycles": args.num_recycles,
        "cases": [],
    }

    for case_name, overrides in parsed_cases:
        case_dir = args.out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_case(
            base_cfg_path=args.config.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            subset_manifest_path=subset_msgpack,
            case_dir=case_dir,
            case_name=case_name,
            overrides=overrides,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            num_steps=args.num_steps,
            num_recycles=args.num_recycles,
        )

        case_result = {
            "name": case_name,
            "overrides": overrides,
            "metrics": metrics,
        }
        atomic_write_text(
            case_dir / "metrics.json",
            json.dumps(case_result, indent=2, sort_keys=True),
        )
        run_summary["cases"].append(case_result)

    atomic_write_text(
        args.out_root / "summary.json",
        json.dumps(run_summary, indent=2, sort_keys=True),
    )


if __name__ == "__main__":
    main()
