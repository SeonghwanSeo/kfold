"""Prepare independent (target, seed) inputs and predict with one runner per GPU."""

import logging

from kfold.cli import predict


def create_parser(prog="kfold pipeline"):
    parser = predict.create_parser(prog=prog)
    parser.description = "Prepare and predict independent (target, seed) jobs"
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seed) or len(set(args.seed)) != len(args.seed):
        parser.error("--seed values must be unique and nonnegative")
    if args.num_gpus < 1 or args.num_samples < 1:
        parser.error("GPU and sample counts must be positive")
    if args.num_recycles < 1 or args.num_steps < 1 or args.num_apo < 1:
        parser.error("--num-recycles, --num-steps and --num-apo must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


def run(args):
    import yaml

    from kfold.cli.multigpu import launch
    from kfold.inference.preparation import needs_apo, prepare_job_inputs

    jobs = prepare_job_inputs(
        args.input,
        args.out_dir,
        seeds=args.seed,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if not jobs:
        logging.info("All target/seed jobs are complete.")
        return
    missing = any(needs_apo(yaml.safe_load(path.read_text())) for path, _ in jobs)
    if args.dry_run and not missing:
        _worker(0, args, jobs)
    else:
        launch(_worker, args, jobs)


def _worker(rank, args, jobs):
    import torch
    import yaml
    from tqdm import tqdm

    from kfold.inference.dataset import InferenceDataset
    from kfold.inference.preparation import needs_apo
    from kfold.runner import KFoldRunner

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    missing = any(needs_apo(yaml.safe_load(path.read_text())) for path, _ in jobs)
    use_cuda = missing or not args.dry_run
    if use_cuda:
        torch.cuda.set_device(rank)
        torch.set_float32_matmul_precision("highest")
    runner = KFoldRunner(
        device=f"cuda:{rank}" if use_cuda else "cpu",
        cache_dir=args.cache_dir,
        weight=args.weight,
        config=args.config,
    )
    try:
        for path, seed in tqdm(jobs, desc=f"Prepare GPU {rank}"):
            document = yaml.safe_load(path.read_text())
            prepared = runner.prepare_document(
                document,
                path.parent,
                source_dir=path.parent,
                seeds=[int(f"{seed}{index}") for index in range(1, args.num_apo + 1)],
                num_samples=5,
                overwrite=args.overwrite,
            )
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(prepared, sort_keys=False))
            temporary.replace(path)
    finally:
        runner.release_apo_models()

    queries = [
        query for path, seed in jobs for query in runner.read_queries(path, seeds=[seed])
    ]
    queries.sort(key=lambda query: query.priority)
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
