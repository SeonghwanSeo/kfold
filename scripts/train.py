import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from kfold.config import load_config, print_config, save_config, to_dict
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.training_module import KFoldTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Co-Folding model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment for logging purposes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output root directory for saving logs and checkpoints.",
    )
    # Easy overrides
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        help="Number of nodes to use for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="Global batch size for training across all GPUs.",
    )
    parser.add_argument(
        "--num_steps_per_epoch",
        type=int,
        help="Number of training steps per epoch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation steps",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        help="Override configuration options using 'key=value' format.",
    )
    return parser.parse_args()


def parse_config(args) -> DictConfig:
    cfg = load_config(args.config, override_args=args.override)

    # Override some config options with command line args
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.experiment_name is not None:
        cfg.train.name = args.experiment_name
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.num_nodes is not None:
        cfg.train.trainer.num_nodes = args.num_nodes
    if args.batch_size is not None:
        cfg.train.global_hparams.batch_size = args.batch_size
    if args.global_batch_size is not None:
        cfg.train.global_hparams.global_batch_size = args.global_batch_size
    if args.num_steps_per_epoch is not None:
        cfg.train.global_hparams.num_global_steps_per_epoch = args.num_steps_per_epoch
    if args.num_workers is not None:
        cfg.train.data.num_workers = args.num_workers
    if args.wandb:
        cfg.train.wandb.use = True

    # Apply global_hparams overrides
    apply_global_hparams_overrides(cfg)

    if args.debug:
        # Enable debug mode settings
        print("Debug mode is enabled: Single GPU, 0 workers, no wandb.")
        cfg.train.trainer.devices = 1
        cfg.train.trainer.num_nodes = 1
        cfg.train.trainer.accumulate_grad_batches = 1
        cfg.train.trainer.log_every_n_steps = 1
        cfg.train.trainer.limit_train_batches = 100
        cfg.train.trainer.limit_val_batches = 10
        cfg.train.trainer.enable_checkpointing = False
        cfg.train.data.train_batch_size = 1
        cfg.train.data.num_workers = 0
        cfg.train.data.safe_load = False
        cfg.train.wandb.use = False

    if args.skip_val:
        # Skip validation steps, use when validation process is not yet ready
        print("Skipping validation steps.")
        cfg.train.trainer.num_sanity_val_steps = 0
        cfg.train.trainer.limit_val_batches = 0

    return cfg


def apply_global_hparams_overrides(cfg: DictConfig) -> None:
    # Estimate number of GPUs
    train_cfg = cfg.train
    if train_cfg.trainer.devices == "auto":
        num_gpus: int = torch.cuda.device_count()
    else:
        num_gpus = cfg.train.trainer.devices
    assert isinstance(num_gpus, int), "num_gpus should be an integer or 'auto'."
    assert num_gpus > 0, "No GPUs available for training."
    world_size: int = train_cfg.trainer.num_nodes * num_gpus

    # Compute parameters dependent on global_hparams
    global_hparams = train_cfg.global_hparams
    max_chains: int = global_hparams.max_chains
    max_tokens: int = global_hparams.max_tokens
    batch_size: int = global_hparams.batch_size
    diffusion_batch_size: int = global_hparams.diffusion_batch_size
    global_batch_size: int = global_hparams.global_batch_size

    if global_batch_size % (batch_size * world_size) != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not "
            f"divisible by (batch_size {batch_size} * world_size {world_size})"
        )
    # compute accumulate_grad_batches
    accumulate_grad_batches: int = global_batch_size // (batch_size * world_size)
    # compute limit_train_batches
    limit_train_batches: int = (
        global_hparams.num_global_steps_per_epoch * accumulate_grad_batches
    )

    train_cfg.data.max_chains = max_chains
    train_cfg.data.max_tokens = max_tokens
    train_cfg.data.train_batch_size = batch_size
    train_cfg.training.diffusion_batch_size = diffusion_batch_size
    train_cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
    train_cfg.trainer.limit_train_batches = limit_train_batches

    # Remove global_hparams from cfg after overrides
    del cfg.train.global_hparams


def build_trainer(cfg, debug: bool = False, skip_val: bool = False) -> pl.Trainer:
    train_cfg = cfg.train
    pl_trainer_cfg = train_cfg.trainer
    save_dir = Path(train_cfg.out_dir) / train_cfg.name

    callbacks = []
    if train_cfg.wandb.use:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            name=train_cfg.name,
            project=train_cfg.wandb.project,
            group=train_cfg.wandb.group,
            entity=train_cfg.wandb.entity,
            tags=train_cfg.wandb.tags,
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        loggers = [wandb_logger]

        @rank_zero_only
        def _save_config() -> None:
            config_out = Path(wandb_logger.experiment.dir) / "train_config.yaml"
            save_config(cfg, config_out)
            wandb_logger.experiment.save("train_config.yaml")

        _save_config()

    else:
        loggers = None  # use default logger

    # Learning rate monitor
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl_callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # TQDM
    tqdm_refresh_rate = 1 if debug else 10
    tqdm_callback = pl_callbacks.TQDMProgressBar(refresh_rate=tqdm_refresh_rate)
    callbacks.append(tqdm_callback)

    if not debug:
        if skip_val:
            # Save checkpoint only based on training loss
            checkpoint_callback = pl_callbacks.ModelCheckpoint(
                monitor="train/loss",
                save_top_k=-1,
                filename="epoch{epoch:04d}_step{step:08d}_loss{train/loss:.4f}",
                mode="min",
                auto_insert_metric_name=False,
            )
        else:
            checkpoint_callback = pl_callbacks.ModelCheckpoint(
                monitor="val/weighted_lddt",
                save_top_k=-1,
                filename="epoch{epoch:04d}_step{step:08d}_lddt{val/weighted_lddt:.4f}",
                mode="max",
                auto_insert_metric_name=False,
            )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=loggers,
        callbacks=callbacks,
        accelerator=pl_trainer_cfg.accelerator,
        strategy=pl_trainer_cfg.strategy,
        devices=pl_trainer_cfg.devices,
        num_nodes=pl_trainer_cfg.num_nodes,
        precision=pl_trainer_cfg.precision,
        max_epochs=pl_trainer_cfg.max_epochs,
        limit_train_batches=pl_trainer_cfg.limit_train_batches,
        limit_val_batches=pl_trainer_cfg.limit_val_batches,
        log_every_n_steps=pl_trainer_cfg.log_every_n_steps,
        enable_checkpointing=pl_trainer_cfg.enable_checkpointing,
        accumulate_grad_batches=pl_trainer_cfg.accumulate_grad_batches,
        gradient_clip_val=pl_trainer_cfg.gradient_clip_val,
        use_distributed_sampler=False,
        # reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def train(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = parse_config(args)

    trainer = build_trainer(cfg, args.debug)

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=True, verbose=False)

    model_module = KFoldTrainingModule(cfg)
    data_module = TrainingDataModule(cfg.train.data)

    # Print config
    if trainer.is_global_zero:
        print_config(cfg)

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    train(args)
