import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch

from kfold.config import load_config, print_config, to_dict
from kfold.training.folding.dataset.datamodule import TrainingDataModule
from kfold.training.folding.training_module import KFoldTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Boltzmann Generator model.")
    parser.add_argument(
        "config",
        type=str,
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
        help="Enable debug mode.",
    )
    return parser.parse_args()


def build_trainer(cfg) -> pl.Trainer:
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
            config=to_dict(cfg),
            save_dir=save_dir,
        )
        loggers = [wandb_logger]
    else:
        loggers = None  # use default logger

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    if not train_cfg.wandb.use:
        # Progress bar (use when not using wandb)
        progress_bar = pl.callbacks.TQDMProgressBar(
            refresh_rate=pl_trainer_cfg.log_every_n_steps
        )
        callbacks.append(progress_bar)

    # FIXME: currently, validation is not implemented yet.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=None,
        save_top_k=-1,
        filename="best-train-loss-{epoch}-{step}",
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
        limit_val_batches=0.0,  # FIXME: validation is not implemented yet
        log_every_n_steps=pl_trainer_cfg.log_every_n_steps,
        enable_checkpointing=pl_trainer_cfg.enable_checkpointing,
        accumulate_grad_batches=pl_trainer_cfg.accumulate_grad_batches,
        gradient_clip_val=pl_trainer_cfg.gradient_clip_val,
        reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def train(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)

    # Override some config options with command line args
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.experiment_name is not None:
        cfg.train.name = args.experiment_name
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.num_nodes is not None:
        cfg.train.trainer.num_nodes = args.num_nodes
    if args.num_workers is not None:
        cfg.train.data.num_workers = args.num_workers
    if args.wandb:
        cfg.train.wandb.use = True

    if args.debug:
        print("Debug mode is enabled: Single GPU, 0 workers, no wandb.")
        cfg.train.trainer.devices = 1
        cfg.train.data.num_workers = 0
        cfg.train.wandb.use = False
        cfg.train.data.safe_load = False
        cfg.train.trainer.accumulate_grad_batches = 1
        cfg.train.trainer.log_every_n_steps = 1

    # Set random seed
    pl.seed_everything(cfg.train.seed)

    trainer = build_trainer(cfg, debug=args.debug)
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
    args = parse_args()
    train(args)
