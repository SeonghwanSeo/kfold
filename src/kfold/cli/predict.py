"""Run predictions for each query and seed."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kfold.inference.query import Query

logger = logging.getLogger("kfold")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # Stage selection, query paths, and generation seeds.
    parser.add_argument(
        "--stage",
        choices=("all", "apo", "complex"),
        default="all",
        help="Run all stages, prepare apos, or predict complexes from --out-dir.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Query JSON/YAML file or directory.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Complex inference seeds (default: [1]).",
    )
    # Apo generation settings.
    parser.add_argument(
        "--apo-config",
        type=Path,
        help="YAML settings for apo batching and AtlasFold prediction.",
    )
    apo_generation = parser.add_mutually_exclusive_group()
    apo_generation.add_argument(
        "--num-apos",
        type=int,
        help="Generated apos per protein entry per inference seed (1–5; default: 1).",
    )
    apo_generation.add_argument(
        "--share-apo-seeds",
        type=int,
        nargs="+",
        help="Generate shared apos with these seeds "
        "(unique, positive; excludes --num-apos).",
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
    parser.add_argument(
        "--conditioning",
        choices=("prior_only", "prior_and_trunk"),
        default="prior_only",
        help=(
            "For assembly queries, reuse predicted intermediates in the ECSI prior "
            "only, or also re-encode them in the trunk apo module."
        ),
    )
    parser.add_argument(
        "--provided-intermediates",
        type=Path,
        help=(
            "For assembly queries, read QUERY/STAGE.npz structures instead of "
            "predicting non-final stages."
        ),
    )
    # Devices, model components, and model cache.
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Visible CUDA device IDs (default: [0]).",
    )
    parser.add_argument(
        "--disable-struct-encoder",
        action="store_true",
        help="Disable the apo structure encoder to save ~6 GB of GPU memory.",
    )
    parser.add_argument(
        "--disable-rna-encoder",
        action="store_true",
        help="Disable the RNA encoder; only for queries without RNA.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload structure and RNA encoders to CPU to save GPU memory (slower).",
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


def _load_queries(args: argparse.Namespace) -> list["Query"]:
    """Validate runtime arguments and load queries in prediction order."""
    from kfold.inference.query import Query

    # Validate all runtime arguments, regardless of the selected stages.
    # GPU selection and generation seeds.
    if not args.gpu_ids or any(gpu_id < 0 for gpu_id in args.gpu_ids):
        raise ValueError("--gpu-ids must be a non-empty list of non-negative indices.")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("--gpu-ids values must be unique.")
    for name in ("seeds", "share_apo_seeds"):
        seeds = getattr(args, name)
        if seeds is not None and (
            not seeds or any(seed < 1 for seed in seeds) or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} requires one or more unique positive seeds."
            )

    # Apo and complex sampling counts.
    if args.num_apos is None and args.share_apo_seeds is None:
        args.num_apos = 1
    if args.num_apos is not None and not 1 <= args.num_apos <= 5:
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

    queries.sort(key=lambda query: query.priority)
    sequential = [query for query in queries if query.assembly is not None]
    if args.provided_intermediates is not None and not sequential:
        raise ValueError("--provided-intermediates requires an assembly query.")
    if (
        args.stage in ("all", "complex")
        and sequential
        and any((args.save_embeddings, args.save_distogram, args.save_trajectory))
    ):
        raise ValueError(
            "Assembly queries do not yet support --save-embeddings, "
            "--save-distogram, or --save-trajectory."
        )
    return queries


def _find_obsolete_queries(
    args: argparse.Namespace, queries: list["Query"]
) -> dict[Path, list[Path]]:
    """Validate preparation mode changes and locate superseded query files."""
    # Switching preparation layouts requires rebuilding the selected apo inputs.
    obsolete_queries: dict[Path, list[Path]] = {}
    for query in queries:
        target_dir = args.out_dir / query.name
        if args.share_apo_seeds is not None:
            paths = list(target_dir.glob(f"{query.name}_seed-*/query.json"))
        else:
            path = target_dir / "query.json"
            paths = [path] if path.is_file() else []
        if not paths:
            continue
        if not args.overwrite or args.stage == "complex":
            raise ValueError(
                f"Changing apo sharing mode for {query.name} requires preparing "
                "apos again with --overwrite and --stage apo or --stage all. "
                "Use the same --share-apo-seeds setting for both stages."
            )
        obsolete_queries[target_dir] = paths

    return obsolete_queries


def _build_jobs(
    args: argparse.Namespace, queries: list["Query"]
) -> tuple[list[tuple["Query", int | None, Path]], list[tuple["Query", int, Path]]]:
    """Build preparation and inference jobs using their respective output layouts."""
    jobs = [
        (query, seed, args.out_dir / query.name / f"{query.name}_seed-{seed}")
        for query in queries
        for seed in args.seeds
    ]
    apo_jobs = (
        [(query, None, args.out_dir / query.name) for query in queries]
        if args.share_apo_seeds is not None
        else jobs
    )
    return apo_jobs, jobs


def dry_run(args: argparse.Namespace) -> None:
    """Validate selected queries and report pending work without writing outputs."""
    from huggingface_hub import hf_hub_download

    from kfold.cli.predict_complex import dry_run as predict_complex
    from kfold.cli.predict_sequential import dry_run as predict_sequential
    from kfold.cli.prepare_apo import dry_run as prepare_apo
    from kfold.data.types.ccd import CCD
    from kfold.inference.runner import ASSETS_REPO_ID

    queries = _load_queries(args)
    _find_obsolete_queries(args, queries)

    logger.info("Checking query CCD codes.")
    ccd_path = hf_hub_download(ASSETS_REPO_ID, "assets/ccd.pkl", cache_dir=args.cache_dir)
    ccd = CCD.load(ccd_path)
    for query in queries:
        query.validate_ccd_codes(ccd.keys())

    apo_jobs, jobs = _build_jobs(args, queries)
    logger.info("Input: %s; output: %s.", args.input, args.out_dir)
    if args.stage in ("all", "apo"):
        prepare_apo(args, apo_jobs)
    if args.stage in ("all", "complex"):
        direct_jobs = [job for job in jobs if job[0].assembly is None]
        sequential_queries = [query for query in queries if query.assembly is not None]
        if direct_jobs:
            predict_complex(args, direct_jobs)
        if sequential_queries:
            predict_sequential(args, sequential_queries)
    logger.info("Dry run complete; no models loaded or outputs written.")


def run(args: argparse.Namespace) -> None:
    """Validate arguments, load queries, and run the selected prediction stages."""
    from kfold.cli.predict_complex import run as predict_complex
    from kfold.cli.predict_sequential import run as predict_sequential
    from kfold.cli.prepare_apo import run as prepare_apo

    queries = _load_queries(args)
    obsolete_queries = _find_obsolete_queries(args, queries)
    apo_jobs, jobs = _build_jobs(args, queries)

    logger.info("Input: %s; output: %s.", args.input, args.out_dir)

    # Prepare provided and generated apo structures.
    if args.stage in ("all", "apo"):
        # Retire the previous layout before writing the replacement queries.
        for target_dir, paths in obsolete_queries.items():
            for done in target_dir.glob(f"{target_dir.name}_seed-*/done.txt"):
                done.unlink()
            for path in paths:
                path.unlink()
                (path.parent / "apo_setting.json").unlink(missing_ok=True)
        prepare_apo(args, apo_jobs)

    # Predict complexes from prepared apo structures.
    if args.stage in ("all", "complex"):
        direct_jobs = [job for job in jobs if job[0].assembly is None]
        sequential_queries = [query for query in queries if query.assembly is not None]
        if direct_jobs:
            predict_complex(args, direct_jobs)
        if sequential_queries:
            predict_sequential(args, sequential_queries)

    # Report completion after all selected stages finish.
    logger.info("Outputs: %s", args.out_dir.resolve())
