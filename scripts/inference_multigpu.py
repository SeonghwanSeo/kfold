import logging
import pathlib
import time

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from kfold.data.types.ccd import CCD
from kfold.inference.dataset import InferenceDataset
from kfold.inference.pl_client import (
    InferenceConfig,
    KFoldInferenceClient,
    KFoldPredictionWriter,
)
from kfold.inference.query import Query, parse_input_files
from kfold.model import KFold

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


@rank_zero_only
def log_error(message: str):
    logger.error(message)


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
        nargs="+",
        type=int,
        default=[42],
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
        default=pathlib.Path(
            "/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd-test.pkl"
        ),
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
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing inference results.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "OmegaConf dotlist override applied before model construction, "
            "e.g. model.structure_module.sampling_schedule_type=phase_power"
        ),
    )

    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("highest")

    args = parse_args()
    # Check output directory
    log_info(f"Output directory: {args.out_dir}")
    if (not args.overwrite) and args.out_dir.exists():
        raise FileExistsError(
            f"Output directory {args.out_dir} already exists. "
            f"Use --overwrite to overwrite existing results."
        )

    # === Input preparation ===
    # Load CCD data
    ccd: CCD = CCD.load(args.ccd)

    # Parse input query(s) and create dataloader
    input_queries: list[Query] = parse_input_files(args.input, ccd, args.seed)
    if len(input_queries) == 0:
        log_error(f"No valid input queries found in {args.input}. Exiting.")
        return
    nsample = len(input_queries)
    nseed = len(args.seed)
    nquery = nsample // nseed
    log_info(f"Predict total {nsample} samples: {nquery} inputs x {nseed} seeds.")

    # Create dataloader
    dataset = InferenceDataset(input_queries, ccd, args.num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.num_workers
    )

    # === Trainer setup ===
    ngpu = args.num_gpus or torch.cuda.device_count()
    if nsample < ngpu:
        log_warning(
            f"Number of inputs({nsample}) is less than the number of GPUs({ngpu}). "
            f"Reducing number of GPUs to {nsample}."
        )
        ngpu = nsample
    log_info(f"Using {ngpu} GPU(s) for inference.")

    # Construct PyTorch Lightning trainer
    inference_writer = KFoldPredictionWriter(args.out_dir, input_queries)
    trainer = pl.Trainer(
        devices=ngpu,
        logger=False,
        callbacks=[inference_writer],
        enable_checkpointing=False,
        precision="bf16-mixed",
        benchmark=False,
        deterministic=True,
    )

    # === Model loading ===
    # Load model and setup inference client
    log_info(f"Loading model from checkpoint: {args.checkpoint}")
    model: KFold = KFold.from_checkpoint(
        args.config, args.checkpoint, override_args=args.override
    )
    log_info("Model loaded successfully.")

    # === Run inference ===
    # Inference configuration
    inference_config = InferenceConfig(
        num_recycles=args.num_recycles,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
    )
    inference_client = KFoldInferenceClient(model, inference_config)

    st = time.time()
    trainer.predict(inference_client, dataloader)
    et = time.time()
    logger.info(
        f"[Rank {trainer.global_rank}] Inference completed. ({et - st:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
