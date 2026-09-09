"""Generate apo/prior structures and prepared query files."""

import argparse
import logging
from pathlib import Path


def create_parser(prog="kfold prepare"):
    parser = argparse.ArgumentParser(
        prog=prog, description="Generate apo/prior structures and prepared query files"
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
    runtime = parser.add_argument_group("runtime options")
    runtime.add_argument(
        "--cache-dir", type=Path, help="Hugging Face cache for models and CCD."
    )
    runtime.add_argument(
        "--num-gpus", type=int, default=1, help="GPU workers distributing query files."
    )
    output = parser.add_argument_group("output options")
    output.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_gpus < 1 or args.num_samples < 1:
        parser.error("GPU and sample counts must be positive")
    if any(s < 0 for s in args.seed) or len(set(args.seed)) != len(args.seed):
        parser.error("--seed values must be unique and nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


def _worker(rank, args, files):
    import torch
    import yaml

    from kfold.inference.preparation import needs_apo
    from kfold.runner import KFoldRunner

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    missing = any(needs_apo(yaml.safe_load(p.read_text())) for p in files)
    if missing:
        torch.cuda.set_device(rank)
        torch.set_float32_matmul_precision("highest")
    runner = KFoldRunner(
        device=f"cuda:{rank}" if missing else "cpu", cache_dir=args.cache_dir
    )
    try:
        for path in files:
            runner.prepare(
                path,
                args.out_dir,
                seeds=args.seed,
                num_samples=args.num_samples,
                overwrite=args.overwrite,
            )
    finally:
        runner.release_apo_models()


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
    if {p.resolve() for p in files} & {
        (args.out_dir / f"{name}.yaml").resolve() for name in names
    }:
        raise ValueError("Output queries must not overwrite source queries.")
    if not missing:
        _worker(0, args, files)
    else:
        launch(_worker, args, files)
