"""Run predictions for each query and seed."""

import argparse
import csv
import json
import logging
import math
import shutil
from pathlib import Path
from time import perf_counter


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Query JSON/YAML file, or directory of query files.",
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
    parser.add_argument(
        "--num-apos",
        type=int,
        default=1,
        help=(
            "Apo structures generated per seed per protein entry without apos "
            "(1–5; default: 1)."
        ),
    )
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
        "--gpu-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Visible CUDA indices for independent query/seed jobs (default: [0]).",
    )
    parser.add_argument(
        "--disable-struct-encoder",
        action="store_true",
        help="Disable pretrained structure encoder for apo structures."
        "Use this to reduce memory usage (~6 GB)",
    )
    parser.add_argument(
        "--disable-rna-encoder",
        action="store_true",
        help="Disable pretrained RNA encoder for RNA targets."
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
    from kfold.inference.query import Query

    if not args.gpu_ids or any(gpu_id < 0 for gpu_id in args.gpu_ids):
        raise ValueError("--gpu-ids must be a non-empty list of non-negative indices.")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("--gpu-ids values must be unique.")
    for name in ("num_samples", "num_recycles", "num_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 1 <= args.num_apos <= 5:
        raise ValueError("--num-apos must be between 1 and 5.")
    if any(seed < 1 for seed in args.seed) or len(set(args.seed)) != len(args.seed):
        raise ValueError("--seed values must be unique and positive.")

    paths = (
        sorted(
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
        if args.input.is_dir()
        else [args.input]
    )
    if not paths:
        raise ValueError(f"No JSON/YAML query files found in {args.input}.")
    queries = [Query.load(path) for path in paths]
    names = set()
    for query in queries:
        if query.name in names:
            raise ValueError(f"Duplicate query name: {query.name!r}.")
        names.add(query.name)
    queries.sort(key=lambda query: query.priority)
    jobs = [
        (query, seed, args.out_dir / query.name / f"{query.name}_seed-{seed}")
        for query in queries
        for seed in args.seed
    ]
    pending_jobs = []
    num_overwrites = 0
    for query, seed, save_dir in jobs:
        if (save_dir / "done.txt").is_file() and not args.overwrite:
            continue
        if save_dir.exists() and any(save_dir.iterdir()):
            num_overwrites += 1
        pending_jobs.append((query, seed, save_dir))
    logging.info(
        "Loaded %d input files, %d queries; seeds=%s, %d query/seed jobs.",
        len(paths),
        len(queries),
        args.seed,
        len(jobs),
    )
    logging.info(
        "Remaining: %d queries, %d query/seed jobs "
        "(%d skipped completed jobs, %d outputs to overwrite).",
        len({query.name for query, _, _ in pending_jobs}),
        len(pending_jobs),
        len(jobs) - len(pending_jobs),
        num_overwrites,
    )
    jobs = pending_jobs
    if args.dry_run:
        logging.info(
            "Dry run complete. CCD checks, structure parsing, and inference were not run."
        )
        return

    start = perf_counter()
    if jobs:
        from kfold.cli.multigpu import launch

        logging.info(
            "Starting runners: num_apos=%d, num_samples=%d, "
            "num_recycles=%d, num_steps=%d.",
            args.num_apos,
            args.num_samples,
            args.num_recycles,
            args.num_steps,
        )
        launch(_worker, args, jobs)
    else:
        logging.info("No pending jobs. Summarizing saved predictions.")

    # All GPU workers have finished writing before reading their outputs.
    for query in queries:
        _summarize_predictions(
            args.out_dir / query.name, query.name, args.seed, args.num_samples
        )
    logging.info(
        "Finished %d query/seed jobs in %.1f s (including model loading and summaries).",
        len(jobs),
        perf_counter() - start,
    )


def _summarize_predictions(
    out_dir: Path, name: str, seeds: list[int], num_samples: int
) -> None:
    """Rank saved samples from the requested seeds and copy the best prediction."""
    records = []
    for seed in seeds:
        directory = out_dir / f"{name}_seed-{seed}"
        if not (directory / "done.txt").is_file():
            raise ValueError(f"Prediction job is incomplete: {directory}.")
        prefix = f"{directory.name}_sample-"
        for sample in range(num_samples):
            path = directory / f"{prefix}{sample}_confidence.json"
            if not path.is_file():
                continue
            with path.open() as f:
                scores = json.load(f)["complex"]
            if not math.isfinite(scores["ranking_score"]):
                raise ValueError(f"Non-finite ranking score in {path}.")
            records.append({"seed": seed, "sample": sample, **scores})

    if not records:
        raise ValueError(f"No saved confidence summaries for {name} in {out_dir}.")

    records.sort(key=lambda row: (-row["ranking_score"], row["seed"], row["sample"]))
    best = records[0]
    directory = out_dir / f"{name}_seed-{best['seed']}"
    prefix = f"{name}_seed-{best['seed']}_sample-{best['sample']}"
    for suffix in ("model.cif", "confidence.json"):
        shutil.copyfile(directory / f"{prefix}_{suffix}", out_dir / f"{name}_{suffix}")

    confidence = directory / f"{prefix}_confidence.npz"
    best_confidence = out_dir / f"{name}_confidence.npz"
    if confidence.is_file():
        shutil.copyfile(confidence, best_confidence)
    else:
        best_confidence.unlink(missing_ok=True)

    with (out_dir / f"{name}_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "sample",
                "ranking_score",
                "plddt",
                "ptm",
                "iptm",
                "pde",
                "has_clash",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    logging.info(
        "%s: best prediction is seed %s sample %s (ranking score %.4f) among %s samples.",
        name,
        best["seed"],
        best["sample"],
        best["ranking_score"],
        len(records),
    )


def _worker(gpu_id, args, jobs, num_workers):
    import torch
    from huggingface_hub.utils import disable_progress_bars

    from kfold.inference.runner import KFoldRunner
    from kfold.model import KFold

    log_format = "%(levelname)s: "
    if num_workers > 1:
        log_format += f"[GPU {gpu_id}] "
    logging.basicConfig(level=logging.INFO, format=log_format + "%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.cuda.set_device(gpu_id)
    torch.set_float32_matmul_precision("highest")
    logging.info(
        "Loading models for %d queries, %d query/seed jobs.",
        len({query.name for query, _, _ in jobs}),
        len(jobs),
    )
    with disable_progress_bars():
        model = KFold.from_pretrained(
            device=torch.device("cuda", gpu_id),
            cache_dir=args.cache_dir,
            use_struct_encoder=not args.disable_struct_encoder,
            use_rna_encoder=not args.disable_rna_encoder,
            cpu_offload=args.cpu_offload,
        )
        runner = KFoldRunner(model, cache_dir=args.cache_dir, lazy_load=True)
    start = perf_counter()
    for index, (query, seed, directory) in enumerate(jobs, start=1):
        job_start = perf_counter()
        logging.info(
            "[%d/%d] starting %s (seed %d).",
            index,
            len(jobs),
            query.name,
            seed,
        )
        result = runner.fold(
            query,
            seed=seed,
            num_apos=args.num_apos,
            num_samples=args.num_samples,
            num_recycles=args.num_recycles,
            num_steps=args.num_steps,
            return_trajectory=args.save_trajectory,
            return_embeddings=args.save_embeddings,
            return_distogram=args.save_distogram,
        )
        (directory / "done.txt").unlink(missing_ok=True)
        result.save(
            directory,
            save_confidence=args.save_confidence,
            save_embeddings=args.save_embeddings,
            save_distogram=args.save_distogram,
            save_trajectory=args.save_trajectory,
        )
        (directory / "done.txt").touch()
        del result
        job_end = perf_counter()
        elapsed = job_end - start
        logging.info(
            "[%d/%d] completed; time=%.1f s, elapsed=%.1f s, ETA=%.1f s.",
            index,
            len(jobs),
            job_end - job_start,
            elapsed,
            elapsed / index * (len(jobs) - index),
        )
