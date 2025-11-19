import argparse

import lightning.pytorch as pl
import torch

from kfold.config import load_config
from kfold.training.folding.dataset.datamodule import TrainingDataModule
from kfold.training.folding.training_module import KFoldTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Boltzmann Generator model.")
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
    return parser.parse_args()


def validate(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)

    # Set random seed
    pl.seed_everything(cfg.train.seed)

    if args.checkpoint is None:
        print("No checkpoint path provided for validation.")
    if args.num_gpus is not None:
        cfg.train.trainer.devices = args.num_gpus
    if args.save_dir is not None:
        cfg.train.validation.save_structure_path = args.save_dir

    model_module = KFoldTrainingModule(cfg)
    data_module = TrainingDataModule(cfg.train.data)

    trainer = pl.Trainer(
        devices=cfg.train.trainer.devices,
        accelerator=cfg.train.trainer.accelerator,
        precision=cfg.train.trainer.precision,
    )

    trainer.validate(
        model_module,
        datamodule=data_module,
        ckpt_path=args.checkpoint,
    )


if __name__ == "__main__":
    args = parse_args()
    validate(args)
