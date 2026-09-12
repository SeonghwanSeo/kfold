"""Dispatch the installed kfold command."""

import argparse
import logging

from kfold.cli.predict import add_arguments, run


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="K-Fold", description="K-Fold structure prediction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Generate apos and predict structures")
    add_arguments(predict)
    return parser


def main(argv=None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
