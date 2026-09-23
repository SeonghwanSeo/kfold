# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.dry_run:
        dry_run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
