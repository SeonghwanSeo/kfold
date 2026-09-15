"""Run structure prediction from the installed kfold command."""

import argparse
import logging

from kfold.cli.predict import add_arguments, dry_run, run


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kfold", description="K-Fold structure prediction", allow_abbrev=False
    )
    add_arguments(parser)
    return parser


def main(argv=None) -> None:
    # Parse the command and configure logging before running prediction stages.
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.dry_run:
        dry_run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
