import argparse
import json
import random
from pathlib import Path

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
        required=True,
        help="Path to a checkpoint file for validation.",
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
        "--save_traj",
        action="store_true",
        help="Return and save diffusion trajectories during validation.",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Do not save predicted structures even when --save_dir is provided.",
    )
    parser.add_argument(
        "--traj_format",
        type=str,
        default="pdb",
        choices=["cif", "pdb"],
        help="Trajectory output format when --save_traj is set.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_val_entries",
        type=int,
        default=None,
        help="Number of validation samples to use",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Config overrides in OmegaConf dotlist format.",
    )
    return parser.parse_args()


def validate(args) -> None:
    # To ignore warning
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config, override_args=args.override)

    # Set random seed
    pl.seed_everything(cfg.train.seed, workers=False)

    cfg.train.validation.num_steps = args.num_steps
    cfg.train.validation.num_recycles = args.num_recycles
    cfg.train.validation.save_predictions = (
        args.save_dir is not None and not args.no_save_predictions
    )
    cfg.train.validation.return_traj = args.save_traj
    cfg.train.validation.traj_format = args.traj_format

    if args.debug:
        cfg.train.data.safe_load = False
        cfg.train.data.num_workers = 0

    model_module = KFoldTrainingModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", config=cfg, weights_only=True
    )
    data_module = TrainingDataModule(cfg.train.data)

    if args.num_val_entries is not None:
        # Construct the validation dataset
        data_module.setup(stage="validate")

        # Use a fixed random seed to ensure the same subset of validation data
        # is selected across different runs
        rng = random.Random(42)

        num_all_entries = len(data_module._val_ds)
        num_entries = args.num_val_entries
        if num_entries > num_all_entries:
            raise ValueError(
                f"Requested number of validation entries ({num_entries}) exceeds the "
                f"total available ({num_all_entries})."
            )
        # select indices
        selected_indices = rng.sample(range(num_all_entries), num_entries)
        selected_indices.sort()
        data_module._val_ds.metadatas = [
            data_module._val_ds.metadatas[i] for i in selected_indices
        ]

    trainer = pl.Trainer(
        default_root_dir=args.save_dir,
        logger=False,
        devices=cfg.train.trainer.devices,
        num_nodes=cfg.train.trainer.num_nodes,
        accelerator=cfg.train.trainer.accelerator,
        precision=cfg.train.trainer.precision,
        strategy=cfg.train.trainer.strategy,
        deterministic=True,
        limit_val_batches=5 if args.debug else None,
        enable_checkpointing=False,
    )

    results = trainer.validate(model_module, datamodule=data_module)
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def to_jsonable(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    return value.item()
                return value.tolist()
            return value

        with open(save_dir / "validation_metrics.json", "w") as f:
            json.dump(
                [{k: to_jsonable(v) for k, v in result.items()} for result in results],
                f,
                indent=2,
            )


if __name__ == "__main__":
    args = parse_args()
    validate(args)
