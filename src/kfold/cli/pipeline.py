"""Prepare apo/prior inputs, then predict structures."""

import argparse
import logging

from kfold.cli import predict, prepare


def create_parser(prog="kfold pipeline"):
    parser = predict.create_parser(prog=prog)
    parser.description = "Prepare apo/prior inputs, then predict structures"
    parser.add_argument(
        "--apo-seed",
        type=int,
        nargs="+",
        help="Preparation seeds; defaults to --seed when omitted.",
    )
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.apo_seed is None:
        args.apo_seed = args.seed
    for option, seeds in (("--seed", args.seed), ("--apo-seed", args.apo_seed)):
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            parser.error(f"{option} values must be unique and nonnegative")
    if args.num_gpus < 1 or args.num_samples < 1:
        parser.error("GPU and sample counts must be positive")
    if args.num_recycles < 1 or args.num_steps < 1 or args.num_apo < 1:
        parser.error("--num-recycles, --num-steps and --num-apo must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


def run(args):
    prepare_args = argparse.Namespace(**vars(args))
    prepare_args.seed = args.apo_seed
    prepare_args.out_dir = args.out_dir / "prepared"
    files = prepare.run(prepare_args)
    predict.run_files(args, files)
