"""Dispatch the installed ``kfold`` command."""

import argparse
import sys


def create_parser():
    parser = argparse.ArgumentParser(
        prog="kfold", description="KFold structure prediction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", add_help=False, help="Generate apo/prior inputs")
    commands.add_parser("predict", add_help=False, help="Predict from prepared inputs")
    commands.add_parser("pipeline", add_help=False, help="Prepare inputs, then predict")
    return parser


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    parser = create_parser()
    if not values or values[0] not in {"prepare", "predict", "pipeline"}:
        parser.parse_args(values)
        return
    if values[0] == "prepare":
        from kfold.cli import prepare as command
    elif values[0] == "predict":
        from kfold.cli import predict as command
    else:
        from kfold.cli import pipeline as command
    command.main(values[1:])
