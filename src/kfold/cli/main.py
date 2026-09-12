"""Dispatch the installed kfold command."""

import argparse
import logging

from kfold.cli.predict import add_arguments, run


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kfold", description="K-Fold structure prediction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict", help="Predict biomolecular complex structures"
    )
    add_arguments(predict)
    return parser


def main(argv=None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(args)


if __name__ == "__main__":
    main()
