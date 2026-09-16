"""Predict partial assemblies, then the full complex with optional re-encoding."""

import argparse
import fcntl
from pathlib import Path

from kfold.data.types.ccd import CCD
from kfold.inference.sequential import (
    ModelBackend,
    execution_plan,
    provided_stage_paths,
    run_manifest,
    run_query,
    write_json,
)
from kfold.inference.sequential_pipeline import InputDataPipeline
from kfold.inference.sequential_query import parse_input_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "ccd", "config", "weight", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-apo", type=int)
    parser.add_argument(
        "--provided-intermediates",
        type=Path,
        help=(
            "Directory QUERY/STAGE.npz of atom-mapped structures with optional "
            "observed_mask; skip intermediate predictions"
        ),
    )
    parser.add_argument(
        "--conditioning",
        choices=["prior_only", "prior_and_trunk"],
        default="prior_only",
        help="Re-encode intermediate objects for the next trunk as well as ECSI",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Final-only control with the same backend; input must have no assembly",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate chemistry, all stage features and source files; no model/GPU run",
    )
    args = parser.parse_args()
    if args.direct and args.conditioning != "prior_only":
        parser.error("--direct requires --conditioning prior_only")
    if args.direct and args.provided_intermediates:
        parser.error("--direct cannot use --provided-intermediates")
    if (
        not args.seed
        or len(set(args.seed)) != len(args.seed)
        or min(args.seed) < 0
        or min(args.num_samples, args.num_recycles, args.num_steps) < 1
    ):
        parser.error(
            "Seeds must be unique nonnegative integers; sampling counts must be positive"
        )
    ccd = CCD.load(args.ccd)
    queries = parse_input_files(args.input, ccd, [0], skip_invalid=False)
    if not queries:
        parser.error("No queries found")
    for q in queries:
        if Path(q.name).name != q.name or q.name in {".", ".."}:
            parser.error("Query names must be safe directory names")
        execution_plan(q, args.direct)
        if args.provided_intermediates:
            provided_stage_paths(q, args.provided_intermediates)
    pipeline = InputDataPipeline(ccd, args.num_samples, args.num_apo)
    manifest = run_manifest(args, queries)
    if args.dry_run:
        from kfold.inference.assembly import subset_query

        for query in queries:
            for stage in execution_plan(query, args.direct):
                struct, _, _, _ = pipeline.run(subset_query(query, stage["chains"]))
                print(
                    query.name, stage["id"], stage["chains"], "atoms=", struct.num_atoms
                )
            if args.provided_intermediates:
                from kfold.inference.assembly import PriorObject, atom_keys

                stages = execution_plan(query)
                groups = []
                supplied = provided_stage_paths(query, args.provided_intermediates)
                for stage in stages[:-1]:
                    obj = PriorObject.load(supplied[stage["id"]])
                    sub = pipeline.read_query(subset_query(query, stage["chains"]))
                    if set(obj.keys) != set(atom_keys(sub)):
                        raise ValueError(
                            "Provided intermediate atom identities do not match query"
                        )
                    groups.append(obj)
                plain = query.copy(assembly=None)
                extra = (
                    {"trunk_groups": groups}
                    if args.conditioning == "prior_and_trunk"
                    else {}
                )
                pipeline.run(
                    plain, prior_groups=groups, stage_index=len(stages) - 1, **extra
                )
        print("Validated. No model inference or Slurm submission performed.")
        return
    import json

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; run GPU inference within a Slurm allocation")
    if args.out_dir.exists() and not args.resume:
        parser.error("Output exists; use --resume with the identical manifest")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out_dir / "manifest.json"
        if args.resume:
            if not path.is_file() or json.loads(path.read_text()) != manifest:
                parser.error(
                    "Resume manifest differs or is missing; use a new output directory"
                )
        else:
            write_json(path, manifest)
        backend = ModelBackend(args)
        for query in queries:
            run_query(
                query,
                pipeline,
                args.seed,
                args.num_samples,
                args.out_dir / query.name,
                backend,
                direct=args.direct,
                conditioning=args.conditioning,
                provided_intermediates=provided_stage_paths(
                    query, args.provided_intermediates
                )
                if args.provided_intermediates
                else None,
            )


if __name__ == "__main__":
    main()
