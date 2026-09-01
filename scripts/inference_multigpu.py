import logging
import pathlib
import time

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from kfold.data.types.ccd import CCD
from kfold.inference.affinity import (
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
)
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
        "--weight",
        type=pathlib.Path,
        required=True,
        help="Path to the model weight file.",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/model/kfold-ecsi.yaml"),
        help="Path to the model configuration file.",
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
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("./inference_results/"),
        help="Root directory to save inference results.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[1],
        help="Random seed for inference reproducibility.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Number of trunk cycles to run during inference.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps to run during inference.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate per input.",
    )
    parser.add_argument(
        "--num-apo",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of apo structures to use per input (default: all).",
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        required=True,
        help="Path to the CCD data file.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing inference results.",
    )
    parser.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save distogram logits, bin edges, and token indices in NPZ format.",
    )
    parser.add_argument(
        "--affinity-head-checkpoint",
        type=pathlib.Path,
        help=(
            "Optional trained affinity head. When supplied, predict p_activity "
            "from the same trunk/distogram pass with the submitted CASP16 "
            "per-query crop contract."
        ),
    )
    parser.add_argument(
        "--affinity-crop-max-tokens",
        type=int,
        default=256,
        help="Maximum ligand-preserving affinity crop size.",
    )
    parser.add_argument(
        "--affinity-crop-max-protein-tokens",
        type=int,
        default=200,
        help="Maximum number of protein tokens in the affinity crop.",
    )
    parser.add_argument(
        "--affinity-pocket-neighborhood-size",
        type=int,
        default=10,
        help="Same-chain protein window size for per-query affinity cropping.",
    )
    parser.add_argument(
        "--affinity-full-precision-inputs",
        action="store_true",
        help=(
            "Do not emulate the BF16 cache boundary used by the submitted "
            "CASP16 scorer. This changes the numerical inference contract."
        ),
    )
    parser.add_argument(
        "--allow-missing-backbone-keys",
        action="store_true",
        help=(
            "Load a legacy backbone with strict=False. Use only for a known "
            "checkpoint/config compatibility gap."
        ),
    )

    return parser.parse_args()


def main():
    # torch.set_float32_matmul_precision("highest")
    torch.set_float32_matmul_precision("high")

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
    dataset = InferenceDataset(input_queries, ccd, args.num_samples, args.num_apo)
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
    inference_writer = KFoldPredictionWriter(
        args.out_dir,
        input_queries,
        save_confidence_scores=False,
        save_distogram=args.save_distogram,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
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
    log_info(f"Loading model from checkpoint: {args.weight}")
    model: KFold = KFold.from_checkpoint(
        args.config,
        args.weight,
        strict=not args.allow_missing_backbone_keys,
    )
    log_info("Model loaded successfully.")

    # === Run inference ===
    # Inference configuration
    inference_config = InferenceConfig(
        num_recycles=args.num_recycles,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
    )
    affinity_predictor = None
    if args.affinity_head_checkpoint is not None:
        affinity_config = PerQueryAffinityConfig(
            max_tokens=args.affinity_crop_max_tokens,
            max_protein_tokens=args.affinity_crop_max_protein_tokens,
            neighborhood_size=args.affinity_pocket_neighborhood_size,
            cache_compatible_bfloat16=not args.affinity_full_precision_inputs,
        )
        affinity_predictor = PerQueryAffinityPredictor.from_checkpoint(
            args.affinity_head_checkpoint,
            config=affinity_config,
        )
        log_info(
            "Affinity head loaded successfully "
            f"(sha256={affinity_predictor.checkpoint_sha256})."
        )
    inference_client = KFoldInferenceClient(
        model,
        inference_config,
        affinity_predictor=affinity_predictor,
    )

    st = time.time()
    trainer.predict(inference_client, dataloader)
    et = time.time()
    logger.info(
        f"[Rank {trainer.global_rank}] Inference completed. ({et - st:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
