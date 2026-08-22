#!/usr/bin/env python3
"""Train the cached frozen-Stage-2 affinity Pairformer readout."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import torch
from omegaconf import OmegaConf

from kfold.model.modules.affinity_pairformer import AffinityPairformer
from kfold.training.affinity.datamodule import AffinityDataModule
from kfold.training.affinity.module import AffinityRankingConfig, AffinityRankingModule
from kfold.training.affinity.telemetry import (
    AffinityPerformanceSmokeCallback,
    AffinityRunTelemetryCallback,
    validate_direct_activity_cliff_batch,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AtomicMilestoneCheckpoint(callbacks.Callback):
    """Publish exact-step checkpoints only after their checksum is durable."""

    def __init__(
        self,
        *,
        dirpath: str | Path,
        milestones: tuple[int, ...] | list[int],
        run_metadata_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        normalized = tuple(sorted(int(step) for step in milestones))
        if any(step <= 0 for step in normalized):
            raise ValueError("Checkpoint milestones must be positive.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Checkpoint milestones must be unique.")
        self.dirpath = Path(dirpath)
        self.milestones = normalized
        self.run_metadata_path = (
            Path(run_metadata_path) if run_metadata_path is not None else None
        )
        self._completed: set[int] = set()

    def state_dict(self) -> dict[str, list[int]]:
        return {"completed": sorted(self._completed)}

    def load_state_dict(self, state_dict: dict[str, list[int]]) -> None:
        self._completed = {int(step) for step in state_dict.get("completed", [])}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            if (
                self.run_metadata_path is not None
                and not self.run_metadata_path.is_file()
            ):
                raise FileNotFoundError(
                    "Run metadata is absent before milestone checkpointing: "
                    f"{self.run_metadata_path}"
                )
        trainer.strategy.barrier("affinity_milestone_checkpoint_dir")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step not in self.milestones or step in self._completed:
            return

        basename = f"milestone-step={step:08d}.ckpt"
        checkpoint_path = self.dirpath / basename
        incomplete_path = self.dirpath / f".{basename}.incomplete"
        marker_path = self.dirpath / f"{basename}.complete.json"
        marker_incomplete_path = self.dirpath / f".{basename}.complete.json.incomplete"
        if trainer.is_global_zero:
            for path in (checkpoint_path, incomplete_path, marker_path):
                if path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite an existing milestone artifact: {path}"
                    )

        trainer.strategy.barrier(f"affinity_milestone_{step}_pre_save")
        # Lightning requires every rank to participate in checkpoint creation,
        # even though only global zero writes for ordinary DDP.
        trainer.save_checkpoint(str(incomplete_path), weights_only=False)
        if trainer.is_global_zero:
            if not incomplete_path.is_file():
                raise RuntimeError(
                    f"Lightning did not create milestone checkpoint {incomplete_path}."
                )
            checkpoint_sha256 = sha256_file(incomplete_path)
            checkpoint_bytes = incomplete_path.stat().st_size
            os.replace(incomplete_path, checkpoint_path)
            marker = {
                "schema_version": "affinity_milestone_checkpoint_v1",
                "global_step": step,
                "checkpoint": checkpoint_path.name,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_sha256": checkpoint_sha256,
                "world_size": int(trainer.world_size),
                "published_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            if self.run_metadata_path is not None:
                marker["run_metadata"] = os.path.relpath(
                    self.run_metadata_path, marker_path.parent
                )
                marker["run_metadata_sha256"] = sha256_file(self.run_metadata_path)
            with marker_incomplete_path.open("w", encoding="utf-8") as handle:
                json.dump(marker, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(marker_incomplete_path, marker_path)
        trainer.strategy.barrier(f"affinity_milestone_{step}_published")
        self._completed.add(step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def launch_rank() -> int:
    """Return scheduler-owned DDP identity before Lightning initializes.

    Rank discovery is infrastructure metadata rather than experiment behavior;
    all affinity behavior is loaded from the resolved YAML configuration.
    """
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def configure_torch_runtime(
    *,
    multiprocessing_sharing_strategy: str,
    float32_matmul_precision: str,
) -> None:
    """Apply the explicit, resolved runtime section of the affinity config."""
    available_strategies = torch.multiprocessing.get_all_sharing_strategies()
    if multiprocessing_sharing_strategy not in available_strategies:
        raise ValueError(
            "Unsupported Torch multiprocessing sharing strategy "
            f"{multiprocessing_sharing_strategy!r}; expected one of "
            f"{sorted(available_strategies)}."
        )
    if float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "float32_matmul_precision must be one of 'highest', 'high', or 'medium'."
        )
    torch.multiprocessing.set_sharing_strategy(multiprocessing_sharing_strategy)
    torch.set_float32_matmul_precision(float32_matmul_precision)


def run_loader_preflight(
    *,
    datamodule: AffinityDataModule,
    batches: int,
    batch_size: int,
    report_path: Path,
) -> None:
    """Measure actual raw-LMDB sparse batches before model compilation."""
    if batches <= 0:
        raise ValueError("Loader preflight requires a positive batch count.")
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    if len(loader) < batches:
        raise ValueError("Configured epoch is shorter than the loader preflight.")
    iterator = iter(loader)
    durations: list[float] = []
    payload_bytes = 0
    for _ in range(batches):
        started = time.perf_counter()
        batch = next(iterator)
        durations.append(time.perf_counter() - started)
        validate_direct_activity_cliff_batch(batch)
        if "z" in batch or "distogram_features" in batch:
            raise ValueError("Raw direct loader materialized a dense CPU pair tensor.")
        payload_bytes += sum(
            value.numel() * value.element_size()
            for value in batch.values()
            if isinstance(value, torch.Tensor)
        )
    ordered = sorted(durations)
    report = {
        "schema_version": "affinity_direct_loader_preflight_v1",
        "state": "passed",
        "batches": batches,
        "batch_size": batch_size,
        "seconds_p50": ordered[len(ordered) // 2],
        "seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "mean_sparse_batch_bytes": payload_bytes / batches,
        "dense_cpu_pair_allocations": 0,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del iterator, loader, batch
    datamodule.teardown("fit")
    gc.collect()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    train = config.train
    configure_torch_runtime(
        multiprocessing_sharing_strategy=str(
            train.runtime.multiprocessing_sharing_strategy
        ),
        float32_matmul_precision=str(train.runtime.float32_matmul_precision),
    )
    if args.smoke:
        train.trainer.devices = 1
        train.trainer.num_nodes = 1
        train.trainer.max_steps = 1
        train.data.train_batches_per_epoch = 1
        train.trainer.limit_val_batches = 1
        train.trainer.num_sanity_val_steps = 0
        train.data.num_workers = 0
    output_dir = Path(train.out_dir) / str(train.name)
    rank = launch_rank()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "affinity_train_config.yaml"
    if rank == 0 and resolved_config.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {output_dir}"
        )
    if rank == 0:
        OmegaConf.save(config, resolved_config)
    pl.seed_everything(int(train.seed), workers=True, verbose=False)
    datamodule = AffinityDataModule(
        manifest_path=str(train.data.manifest_path),
        cache_root=str(train.data.cache_root),
        train_batches_per_epoch=int(train.data.train_batches_per_epoch),
        seed=int(train.seed),
        num_workers=int(train.data.num_workers),
        pin_memory=bool(train.data.pin_memory),
        train_batch_size=int(train.data.train_batch_size),
        val_batch_size=int(train.data.val_batch_size),
        cache_schema=str(train.data.cache_schema),
        cache_encoding=(
            str(train.data.cache_encoding)
            if train.data.get("cache_encoding") is not None
            else None
        ),
        max_crop_tokens=int(train.data.max_crop_tokens),
        max_protein_crop_tokens=int(train.data.max_protein_crop_tokens),
        shape_buckets=tuple(int(bucket) for bucket in train.data.shape_buckets),
        sampling_mode=str(train.data.sampling_mode),
        crop_mode=str(train.data.crop_mode),
        pocket_manifest_path=(
            str(train.data.pocket_manifest_path)
            if train.data.pocket_manifest_path is not None
            else None
        ),
        pocket_neighborhood_size=int(train.data.pocket_neighborhood_size),
        crop_contract_version=str(train.data.crop_contract_version),
        distogram_pocket_distance_cutoff=float(
            train.data.distogram_pocket_distance_cutoff
        ),
        distogram_use_entropy_tiebreak=bool(train.data.distogram_use_entropy_tiebreak),
        distogram_entropy_gate=float(train.data.distogram_entropy_gate),
        distogram_strong_binder_p_activity=float(
            train.data.distogram_strong_binder_p_activity
        ),
        activity_group_size=int(train.data.activity_group_size),
        activity_groups_per_batch=int(train.data.activity_groups_per_batch),
        singleton_regression_slots=int(train.data.singleton_regression_slots),
        labels_per_source=int(train.data.labels_per_source),
        rankable_assays_per_batch=int(train.data.rankable_assays_per_batch),
        records_per_rankable_assay=int(train.data.records_per_rankable_assay),
    )
    performance_config = train.get("performance_smoke", {})
    performance_enabled = bool(performance_config.get("enabled", False))
    if performance_enabled:
        if (
            rank != 0
            or int(train.trainer.devices) != 1
            or int(train.trainer.num_nodes) != 1
        ):
            raise ValueError("The direct performance smoke requires one process/GPU.")
        run_loader_preflight(
            datamodule=datamodule,
            batches=int(performance_config.loader_batches),
            batch_size=int(train.data.train_batch_size),
            report_path=output_dir / "loader-preflight.json",
        )
    model_config = OmegaConf.to_container(train.model, resolve=True)
    if not isinstance(model_config, dict):
        raise TypeError("Resolved affinity model config must be a mapping.")
    use_kernels = bool(model_config.pop("use_kernels", False))
    model = AffinityRankingModule(
        model_config=AffinityPairformer.Config(**model_config),
        task_config=AffinityRankingConfig(
            **OmegaConf.to_container(train.task, resolve=True)
        ),
        use_kernels=use_kernels,
    )
    if bool(train.compile.enabled):
        model.model = torch.compile(
            model.model,
            mode=str(train.compile.mode),
            dynamic=bool(train.compile.dynamic),
        )
    logger: bool | pl.loggers.Logger = False
    wandb_run_id: str | None = None
    if bool(train.wandb.use):
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            name=str(train.name),
            project=str(train.wandb.project),
            entity=(str(train.wandb.entity) if train.wandb.entity is not None else None),
            group=(str(train.wandb.group) if train.wandb.group is not None else None),
            tags=list(train.wandb.tags),
            config=OmegaConf.to_container(config, resolve=True),
            save_dir=str(output_dir),
            log_model=bool(train.wandb.log_model),
        )
        logger = wandb_logger
        if rank == 0:
            wandb_run_id = str(wandb_logger.experiment.id)
    telemetry = AffinityRunTelemetryCallback(
        summary_path=output_dir / "training-summary.json",
        log_every_n_steps=int(train.trainer.log_every_n_steps),
    )
    active_callbacks: list[pl.Callback] = [telemetry]
    if bool(train.checkpoint.get("enabled", True)):
        active_callbacks.extend(
            [
                callbacks.ModelCheckpoint(
                    dirpath=output_dir / "checkpoints",
                    monitor=str(train.checkpoint.monitor),
                    mode=str(train.checkpoint.monitor_mode),
                    save_top_k=int(train.checkpoint.save_top_k),
                    every_n_epochs=int(train.checkpoint.every_n_epochs),
                    save_on_train_epoch_end=False,
                    save_last=True,
                    filename=(
                        "epoch={epoch:04d}-step={step:08d}-"
                        "pearson={val/mean_assay_pearson:.4f}"
                    ),
                    auto_insert_metric_name=False,
                ),
                AtomicMilestoneCheckpoint(
                    dirpath=output_dir / "checkpoints",
                    milestones=[int(step) for step in train.checkpoint.milestones],
                    run_metadata_path=output_dir / "run_metadata.json",
                ),
            ]
        )
    if performance_enabled:
        active_callbacks.append(
            AffinityPerformanceSmokeCallback(
                report_path=output_dir / "performance-smoke.json",
                warmup_steps=int(performance_config.warmup_steps),
                measured_steps=int(performance_config.measured_steps),
                max_p50_seconds=float(performance_config.max_p50_seconds),
                max_p95_seconds=float(performance_config.max_p95_seconds),
                max_data_wait_fraction=float(performance_config.max_data_wait_fraction),
            )
        )
    if logger is not False and torch.cuda.is_available():
        active_callbacks.append(callbacks.DeviceStatsMonitor(cpu_stats=None))
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=train.trainer.accelerator,
        strategy=train.trainer.strategy,
        devices=train.trainer.devices,
        num_nodes=int(train.trainer.num_nodes),
        precision=train.trainer.precision,
        max_epochs=int(train.trainer.max_epochs),
        max_steps=int(train.trainer.max_steps),
        limit_val_batches=train.trainer.limit_val_batches,
        log_every_n_steps=int(train.trainer.log_every_n_steps),
        gradient_clip_val=float(train.trainer.gradient_clip_val),
        num_sanity_val_steps=int(train.trainer.num_sanity_val_steps),
        use_distributed_sampler=bool(train.trainer.use_distributed_sampler),
        logger=logger,
        callbacks=active_callbacks,
    )
    if rank == 0:
        run_metadata = {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip(),
            "config_sha256": sha256_file(resolved_config),
            "manifest_sha256": sha256_file(Path(train.data.manifest_path)),
            "pocket_manifest_sha256": (
                sha256_file(Path(train.data.pocket_manifest_path))
                if train.data.pocket_manifest_path is not None
                else None
            ),
            "cache_root": str(train.data.cache_root),
            "snapshot_digest": (
                str(train.lineage.snapshot_digest)
                if train.lineage.snapshot_digest is not None
                else None
            ),
            "crop_contract_version": str(train.data.crop_contract_version),
            "backbone_in_graph": False,
            "use_kernels": use_kernels,
            "compile": OmegaConf.to_container(train.compile, resolve=True),
            "wandb": {
                "enabled": bool(train.wandb.use),
                "project": str(train.wandb.project),
                "run_id": wandb_run_id,
            },
            "checkpoint_metric": str(train.checkpoint.monitor),
            "milestone_steps": [int(step) for step in train.checkpoint.milestones],
            "performance_smoke": (
                OmegaConf.to_container(performance_config, resolve=True)
                if OmegaConf.is_config(performance_config)
                else dict(performance_config)
            ),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
