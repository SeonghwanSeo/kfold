import argparse

import lightning.pytorch as pl
import torch

from kfold.config import load_config
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.training_module import KFoldTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Co-Folding model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint file for validation.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument("--save_dir", type=str, help="Directory path to save structure")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of diffusion steps for validation",
    )
    parser.add_argument(
        "--num_recycles", type=int, default=3, help="Number of cycling for validation"
    )
    parser.add_argument(
        "--return_traj",
        action="store_true",
        help="Return and save diffusion trajectories during validation.",
    )
    parser.add_argument(
        "--traj_format",
        type=str,
        default="pdb",
        choices=["cif", "pdb"],
        help="Trajectory output format when --return_traj is set.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def validate(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=False)

    if args.checkpoint is None:
        print("No checkpoint path provided for validation.")
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    else:
        cfg.train.trainer.devices = "auto"
    cfg.train.validation.num_steps = args.num_steps
    cfg.train.validation.num_recycles = args.num_recycles
    cfg.train.validation.save_predictions = True
    cfg.train.validation.return_traj = args.return_traj
    cfg.train.validation.traj_format = args.traj_format

    if args.debug:
        cfg.train.data.safe_load = False
        cfg.train.data.num_workers = 0

    model_module = KFoldTrainingModule(cfg)
    data_module = TrainingDataModule(cfg.train.data)

    trainer = pl.Trainer(
        default_root_dir=args.save_dir,
        logger=False,
        devices=cfg.train.trainer.devices,
        accelerator=cfg.train.trainer.accelerator,
        precision=cfg.train.trainer.precision,
        deterministic=True,
        limit_val_batches=5 if args.debug else None,
        enable_checkpointing=False,
    )

    trainer.validate(
        model_module,
        datamodule=data_module,
        ckpt_path=args.checkpoint,
    )


if __name__ == "__main__":
    args = parse_args()
    validate(args)
