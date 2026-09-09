"""Predict structures from prepared apo inputs."""

import argparse
import logging
from pathlib import Path


def create_parser(prog="kfold predict"):
    parser = argparse.ArgumentParser(
        prog=prog, description="Predict structures from prepared apo inputs"
    )
    required = parser.add_argument_group("input and output")
    required.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Query YAML/JSON file or directory.",
    )
    required.add_argument(
        "-o", "--out-dir", type=Path, required=True, help="Output directory."
    )
    inference = parser.add_argument_group("inference options")
    inference.add_argument("--seed", type=int, nargs="+", default=[1])
    inference.add_argument("--num-samples", type=int, default=5)
    inference.add_argument("--num-recycles", type=int, default=10)
    inference.add_argument("--num-steps", type=int, default=100)
    inference.add_argument(
        "--num-apo",
        type=int,
        default=1,
        help="Apo candidates per protein entry (default: 1).",
    )
    runtime = parser.add_argument_group("runtime options")
    runtime.add_argument(
        "--cache-dir", type=Path, help="Hugging Face cache for models and CCD."
    )
    runtime.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPU workers distributing (query, seed) jobs.",
    )
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prepared inputs on CPU without model inference.",
    )
    output = parser.add_argument_group("output options")
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--save-confidence", action="store_true")
    output.add_argument("--save-distogram", action="store_true")
    output.add_argument("--save-trajectory", action="store_true")
    local = parser.add_argument_group("local model overrides")
    local.add_argument(
        "--weight", type=Path, help="Optional local weights; defaults to the release."
    )
    local.add_argument(
        "--config", type=Path, help="Optional local config; defaults to the release."
    )
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_gpus < 1 or args.num_samples < 1:
        parser.error("GPU and sample counts must be positive")
    if any(s < 0 for s in args.seed) or len(set(args.seed)) != len(args.seed):
        parser.error("--seed values must be unique and nonnegative")
    if args.num_recycles < 1 or args.num_steps < 1 or args.num_apo < 1:
        parser.error("--num-recycles, --num-steps and --num-apo must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


def _worker(rank, args, jobs):
    from collections import defaultdict

    import torch
    from tqdm import tqdm

    from kfold.inference.dataset import InferenceDataset
    from kfold.runner import KFoldRunner

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.dry_run:
        torch.cuda.set_device(rank)
        torch.set_float32_matmul_precision("highest")
    runner = KFoldRunner(
        device="cpu" if args.dry_run else f"cuda:{rank}",
        cache_dir=args.cache_dir,
        weight=args.weight,
        config=args.config,
    )
    seeds_by_file = defaultdict(list)
    for path, seed in jobs:
        seeds_by_file[path].append(seed)
    queries = [
        q
        for path, seeds in seeds_by_file.items()
        for q in runner.read_queries(path, seeds=seeds)
    ]
    queries.sort(key=lambda q: q.priority)
    if args.dry_run:
        for _ in tqdm(
            InferenceDataset(queries, runner.ccd, args.num_samples, args.num_apo),
            desc="Validate inputs",
        ):
            pass
        return
    for query in tqdm(queries, desc=f"Predict GPU {rank}"):
        result = runner.fold(
            query,
            num_samples=args.num_samples,
            num_recycles=args.num_recycles,
            num_steps=args.num_steps,
            num_apo=args.num_apo,
            return_trajectory=args.save_trajectory,
            return_distogram=args.save_distogram,
        )
        result.save(args.out_dir, save_confidence=args.save_confidence)


def run(args):
    import yaml

    from kfold.cli.multigpu import launch
    from kfold.inference.preparation import input_files, needs_apo

    files = input_files(args.input)
    documents = [yaml.safe_load(path.read_text()) for path in files]
    names = [
        data.get("name", path.stem) for path, data in zip(files, documents, strict=True)
    ]
    if len(set(names)) != len(names):
        raise ValueError("Query names must be unique across input files.")
    missing = any(needs_apo(data) for data in documents)
    if missing:
        raise ValueError(
            "predict requires apo for every protein; run kfold prepare first."
        )
    jobs = [
        (path, seed)
        for path, name in zip(files, names, strict=True)
        for seed in args.seed
        if args.dry_run
        or args.overwrite
        or not (args.out_dir / name / f"{name}_seed-{seed}" / "done.txt").exists()
    ]
    if not jobs:
        logging.info("All query/seed jobs are complete.")
        return
    if args.dry_run:
        _worker(0, args, jobs)
    else:
        launch(_worker, args, jobs)
