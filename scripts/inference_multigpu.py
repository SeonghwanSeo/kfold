import logging
import pathlib

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from kfold.data.types.ccd import CCD
from kfold.inference.dataset import prepare_inference_dataloader
from kfold.inference.pl_client import (
    InferenceConfig,
    KFoldInferenceClient,
    KFoldPredictionWriter,
)
from kfold.inference.query import Query, parse_input_files
from kfold.model.models import KFold

logger = logging.getLogger("kfold.inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@rank_zero_only
def log_info(message: str):
    logger.info(message)


@rank_zero_only
def log_warning(message: str):
    logger.warning(message)


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
        "--use_sequence_masking",
        action="store_true",
        help=(
            "Whether to mask sequence to increase sampling diversity."
            "This is only meaningful when using multiple seeds"
        ),
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd-test.pkl",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous inference results if available.",
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
    log_info(f"Loading model from checkpoint: {args.checkpoint}")
    model: KFold = KFold.from_checkpoint(args.config, args.checkpoint)
    model = model.cast_to_bf16().eval()
    log_info("Model loaded successfully.")

    # Inference configuration
    inference_config = InferenceConfig(
        num_recycles=args.num_recycles,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    inference_client = KFoldInferenceClient(model, inference_config)
    inference_writer = KFoldPredictionWriter(args.out_dir)

    # Construct PyTorch Lightning trainer
    trainer = pl.Trainer(
        devices=devices,
        logger=False,
        callbacks=[inference_writer],
        enable_checkpointing=False,
        precision="bf16-mixed",
        benchmark=False,
        deterministic=True,
    )

    # Load CCD data
    ccd: CCD = CCD.load(args.ccd)

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    input_queries: list[Query] = parse_input_files(
        args.input,
        ccd=ccd,
        skip_invalid=True,
    )
    if trainer.is_global_zero:
        print(f"Parsed {len(input_queries)} valid input queries from {args.input}")

    if args.resume:
        # Filter out queries that already have results saved
        input_queries = [q for q in input_queries if not (args.out_dir / q.name).exists()]
        if trainer.is_global_zero:
            print(
                f"{len(input_queries)} queries remaining after filtering existing results"
            )

    # Create data loader
    dataloader = prepare_inference_dataloader(
        queries=input_queries,
        ccd=ccd,
        num_samples=args.num_samples,
        use_sequence_masking=args.use_sequence_masking,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Run inference
    trainer.predict(inference_client, dataloaders=dataloader)


if __name__ == "__main__":
    main()
