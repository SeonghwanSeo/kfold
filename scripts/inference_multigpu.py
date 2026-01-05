import pathlib
import random

import numpy as np
import torch
from lightning import pytorch as pl

from kfold.config import load_config
from kfold.data.types.ccd import CCD
from kfold.inference.dataset import prepare_inference_dataloader
from kfold.inference.pl_client import InferenceConfig, KFoldInferenceClient
from kfold.inference.query import InputFile, parse_input_files
from kfold.model.models import KFold


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="KFold Inference Script")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to the input(s) for inference (file or directory).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("./inference_results/"),
        help="Root directory to save inference results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for inference reproducibility.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="Number of trunk cycles to run during inference.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of diffusion steps to run during inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate per input.",
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd.pkl",
        help="Path to the CCD data file.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )

    return parser.parse_args()


def main():
    # Setup environment
    torch.set_float32_matmul_precision("highest")

    args = parse_args()

    # Determine the number of devices
    devices: str | int = "auto"
    if args.num_gpus is not None:
        devices = args.num_gpus

    # Load model and setup inference client
    config = load_config(args.config)
    if "model" in config:
        # Get model config if wrapped in a higher-level config
        config = config.model
    model: KFold = KFold.from_checkpoint(config, args.checkpoint)

    # Inference configuration
    inference_config = InferenceConfig(
        num_recycles=args.num_recycles,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    inference_client = KFoldInferenceClient(
        model, inference_config, save_dir=args.out_dir
    )

    # Load CCD data
    ccd: CCD = CCD.load(args.ccd)

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    input_queries: list[InputFile] = parse_input_files(
        args.input,
        ccd=ccd,
        skip_invalid=True,
    )

    # Create data loader
    dataloader = prepare_inference_dataloader(
        input_files=input_queries,
        ccd=ccd,
        seq_embedding_dim=model.channel_seq_encoder,
        struct_embedding_dim=model.channel_struct_encoder,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    trainer = pl.Trainer(
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        precision="bf16-mixed",
        benchmark=False,
        deterministic=True,
    )

    # Run inference
    trainer.predict(inference_client, dataloaders=dataloader)


if __name__ == "__main__":
    main()
