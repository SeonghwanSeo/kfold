import argparse
import random
from pathlib import Path

import lightning.pytorch as pl
import torch

from kfold.config import load_config, print_config
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.training.training_module import KFoldTrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Co-Folding model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the yaml configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a checkpoint file for validation.",
        required=True,
    )
    return parser.parse_args()


def validate() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Create wandb logger for logging validation metrics
    logger = pl.loggers.WandbLogger(
        entity="K-Fold",
        project="val-sweep-v260408-6",
    )
    trainer = pl.Trainer(
        logger=logger,
        devices="auto",
        accelerator="auto",
        precision="bf16-mixed",
        log_every_n_steps=1,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    if trainer.global_rank == 0:
        print_config(cfg)

    # Set random seed
    pl.seed_everything(cfg.train.seed)

    # Construct the model and data modules
    model_module = KFoldTrainingModule.load_from_checkpoint(
        args.checkpoint, config=cfg, weights_only=True
    )
    data_module = TrainingDataModule(cfg.train.data)

    # Randomly select 192 entries from the validation dataset for evaluation
    data_module.setup(stage="validate")
    rng = random.Random(42)
    num_all_entries = len(data_module._val_ds)
    selected_indices = rng.sample(range(num_all_entries), 160)
    selected_indices.sort()
    data_module._val_ds.metadatas = [
        data_module._val_ds.metadatas[i] for i in selected_indices
    ]

    trainer.validate(model_module, datamodule=data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    validate()
