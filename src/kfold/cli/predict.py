"""Run single-query inference for each requested target and seed."""

import argparse
import logging
import shutil
from pathlib import Path


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
        help="Number of AtlasFold samples per entry without supplied apos (default: 1).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples (default: 5).",
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
        "--num-gpus",
        type=int,
        default=1,
        help="GPU processes distributing target/seed jobs (default: 1).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory for models and CCD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check query files and paths, then list jobs without loading models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target/seed directories after successful inference.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Save confidence scores",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save single/pair embeddings",
    )
    parser.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save distogram logits",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save generative trajectory",
    )


def run(args: argparse.Namespace) -> None:
    from kfold.inference.query import ProteinMultimerSequence, ProteinSequence, Query

    for name in ("num_gpus", "num_apos", "num_samples", "num_recycles", "num_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
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
    sources = {path.resolve() for path in paths}
    for query in queries:
        if query.name in {".", ".."} or "/" in query.name or "\\" in query.name:
            raise ValueError(f"Query name must be a directory name: {query.name!r}.")
        if query.name in names:
            raise ValueError(f"Duplicate query name: {query.name!r}.")
        names.add(query.name)
        for entry in [*query.sequences, *query.multimer_sequences]:
            if not isinstance(entry, (ProteinSequence, ProteinMultimerSequence)):
                continue
            if entry.prior is not None and not entry.apo:
                raise ValueError(
                    f"Query {query.name!r}: 'prior' requires supplied 'apo'."
                )
            for field in ("apo", "prior"):
                sources.update(
                    Path(path).resolve() for path in getattr(entry, field) or []
                )
            if not entry.apo:
                logging.info(
                    "%s chains %s: will generate %s apo group(s) per KFold seed.",
                    query.name,
                    entry.ids,
                    args.num_apos,
                )
    queries.sort(key=lambda query: query.priority)
    jobs = [
        (query, seed, args.out_dir / query.name / f"{query.name}_seed-{seed}")
        for query in queries
        for seed in args.seed
    ]
    for query, seed, directory in jobs:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"Output must be a regular directory: {directory}.")
        if directory.exists() and any(directory.iterdir()):
            if not args.overwrite:
                raise ValueError(
                    f"Output directory is not empty: {directory}. "
                    "Use --overwrite or another output directory."
                )
            if any(source.is_relative_to(directory.resolve()) for source in sources):
                raise ValueError(
                    f"Cannot overwrite {directory}: it contains an input query or "
                    "structure file. Use another output directory."
                )
        logging.info("%s seed %s -> %s", query.name, seed, directory)
    if args.dry_run:
        logging.info(
            "Validated %s query file(s); planned %s job(s). "
            "CCD checks, structure parsing, and inference were not run.",
            len(queries),
            len(jobs),
        )
        return

    from kfold.cli.multigpu import launch

    launch(_worker, args, jobs)


def _worker(rank, args, jobs):
    import torch
    from tqdm import tqdm

    from kfold.inference.runner import KFoldRunner
    from kfold.model import KFold

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("highest")
    model = KFold.from_pretrained(
        device=torch.device("cuda", rank), cache_dir=args.cache_dir
    )
    runner = KFoldRunner(model, cache_dir=args.cache_dir)
    for query, seed, directory in tqdm(jobs, desc=f"Predict GPU {rank}"):
        logging.info("Predicting %s seed %s on GPU %s", query.name, seed, rank)
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
        if args.overwrite and directory.exists():
            shutil.rmtree(directory)
        result.save(
            directory,
            save_confidence=args.save_confidence,
            save_embeddings=args.save_embeddings,
            save_distogram=args.save_distogram,
            save_trajectory=args.save_trajectory,
        )
        del result
