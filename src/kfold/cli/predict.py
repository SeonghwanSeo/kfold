"""Run predictions for each query and seed."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("kfold")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # Stage selection, query paths, and generation seeds.
    parser.add_argument(
        "--stage",
        choices=("all", "apo", "complex"),
        default="all",
        help="Run both stages (all), prepare apos (apo), or predict complexes "
        "from prepared queries in --out-dir (complex).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Query JSON/YAML file or directory selecting the queries to predict.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Generation seeds (default: [1]).",
    )
    # Apo generation settings.
    parser.add_argument(
        "--apo-config",
        type=Path,
        help="YAML settings for apo batching and AtlasFold prediction.",
    )
    parser.add_argument(
        "--num-apos",
        type=int,
        default=1,
        help=(
            "Apo structures per seed per protein entry during automatic generation "
            "(1–5; default: 1). Provided structures are used directly."
        ),
    )
    # Complex prediction settings.
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Predictions per query/seed (default: 5).",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Number of recycling iterations (default: 10).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100).",
    )
    # Devices, model components, and model cache.
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Visible CUDA indices for independent query/seed jobs (default: [0]).",
    )
    parser.add_argument(
        "--disable-struct-encoder",
        action="store_true",
        help="Disable pretrained structure encoder for apo structures. "
        "Use this to reduce memory usage (~6 GB)",
    )
    parser.add_argument(
        "--disable-rna-encoder",
        action="store_true",
        help="Disable pretrained RNA encoder for RNA targets. "
        "Only use this when RNA targets are not present in the queries",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep the protein structure backbone encoder and RNA LM on CPU "
        "between feature extraction calls. Trades transfer time for GPU memory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory for models and CCD.",
    )
    # Execution control and optional output files.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check query files and paths; count pending jobs without loading models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed query/seed jobs.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Save per-atom and token-pair confidence arrays.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save internal trunk embeddings.",
    )
    parser.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save distogram.",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save diffusion trajectories.",
    )


def run(args: argparse.Namespace) -> None:
    """Validate arguments, load queries, and run the selected prediction stages."""
    from kfold.cli.predict_complex import run as predict_complex
    from kfold.cli.prepare_apo import run as prepare_apo
    from kfold.inference.query import Query

    # Validate all runtime arguments, regardless of the selected stages.
    # GPU selection and generation seeds.
    if not args.gpu_ids or any(gpu_id < 0 for gpu_id in args.gpu_ids):
        raise ValueError("--gpu-ids must be a non-empty list of non-negative indices.")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("--gpu-ids values must be unique.")
    if any(seed < 1 for seed in args.seed) or len(set(args.seed)) != len(args.seed):
        raise ValueError("--seed values must be unique and positive.")

    # Apo and complex sampling counts.
    if not 1 <= args.num_apos <= 5:
        raise ValueError("--num-apos must be between 1 and 5.")
    for name in ("num_samples", "num_recycles", "num_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    # Discover query files from a single file or directory.
    if args.input.is_dir():
        paths = sorted(
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
    else:
        paths = [args.input]
    if not paths:
        raise ValueError(f"No JSON/YAML query files found in {args.input}.")

    # Load queries and reject names that would share an output directory.
    queries = [Query.load(path) for path in paths]
    names = set()
    for query in queries:
        if query.name in names:
            raise ValueError(f"Duplicate query name: {query.name!r}.")
        names.add(query.name)

    # Validate CCD membership without loading prediction models.
    if args.dry_run:
        from huggingface_hub import hf_hub_download

        from kfold.data.types.ccd import CCD
        from kfold.inference.runner import ASSETS_REPO_ID

        logger.info("Checking query CCD codes.")
        ccd_path = hf_hub_download(
            ASSETS_REPO_ID, "assets/ccd.pkl", cache_dir=args.cache_dir
        )
        ccd = CCD.load(ccd_path)
        for query in queries:
            query.validate_ccd_codes(ccd.keys())

    # Schedule smaller queries first, with one output directory per query/seed.
    queries.sort(key=lambda query: query.priority)
    jobs = [
        (query, seed, args.out_dir / query.name / f"{query.name}_seed-{seed}")
        for query in queries
        for seed in args.seed
    ]

    logger.info("Input: %s; output: %s.", args.input, args.out_dir)

    # Prepare provided and generated apo structures.
    if args.stage in ("all", "apo"):
        prepare_apo(args, jobs)

    # Predict complexes from prepared apo structures.
    if args.stage in ("all", "complex"):
        predict_complex(args, jobs)

    # Report completion after all selected stages finish.
    if args.dry_run:
        logger.info("Dry run complete; no models loaded or outputs written.")
    else:
        logger.info("Outputs: %s", args.out_dir.resolve())
