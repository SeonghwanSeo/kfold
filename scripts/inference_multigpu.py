import logging
import pathlib
import time

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from kfold.config import load_config
from kfold.data.types.ccd import CCD
from kfold.inference.dataset import InferenceDataset
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


@rank_zero_only
def log_error(message: str):
    logger.error(message)


@rank_zero_only
def check_out_dir(out_dir: pathlib.Path, overwrite: bool) -> None:
    if (not overwrite) and out_dir.exists():
        raise FileExistsError(
            f"Output directory {out_dir} already exists. "
            f"Use --overwrite to overwrite existing results."
        )


@rank_zero_only
def create_out_dir(queries: list[Query], out_dir: pathlib.Path):
    for query in queries:
        query_dir = out_dir / query.name
        query_dir.mkdir(parents=True, exist_ok=True)
        query.save(query_dir / "query.yaml")


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
        default=None,
        help=(
            "Path to the CCD data file. If omitted, use train.data.ccd_path "
            "from the resolved config."
        ),
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


def resolve_ccd_path(
    config_path: pathlib.Path,
    override_args: list[str],
    ccd_path: pathlib.Path | None,
) -> pathlib.Path:
    if ccd_path is not None:
        return ccd_path

    config = load_config(config_path, override_args=override_args or None)
    resolved_path = getattr(getattr(config.train, "data", None), "ccd_path", None)
    if resolved_path is None:
        raise ValueError(
            "CCD path was not provided and train.data.ccd_path was not found "
            f"in config: {config_path}"
        )
    return pathlib.Path(str(resolved_path))


def main():
    torch.set_float32_matmul_precision("highest")

    args = parse_args()

    # Check output directory
    check_out_dir(args.out_dir, args.overwrite)

    # === Input preparation ===
    # Load CCD data
    ccd_path = resolve_ccd_path(args.config, args.override, args.ccd)
    log_info(f"Loading CCD data from: {ccd_path}")
    ccd: CCD = CCD.load(ccd_path)

    # Parse input query(s) and create dataloader
    input_queries: list[Query] = parse_input_files(args.input, ccd, args.seed)
    if len(input_queries) == 0:
        log_error(f"No valid input queries found in {args.input}. Exiting.")
        return
    nsample = len(input_queries)
    nseed = len(args.seed)
    nquery = nsample // nseed
    log_info(f"Predict total {nsample} samples: {nquery} inputs x {nseed} seeds.")

    # Create output directories for each query
    create_out_dir(input_queries, args.out_dir)

    # Create dataloader
    dataset = InferenceDataset(
        input_queries, ccd, args.num_samples, args.use_sequence_masking
    )
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

    inference_writer = KFoldPredictionWriter(args.out_dir)
    # Construct PyTorch Lightning trainer
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
        args.config,
        args.checkpoint,
        override_args=args.override or None,
    )
    model = model.cast_to_bf16().eval()
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
