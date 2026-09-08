"""Generate ranked AtlasFold apo ensembles; model imports are deferred to execution."""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from kfold.training.preprocess.apo_preparation import (
    MAX_APO_LENGTH,
    ApoTask,
    eligible_tasks,
    multimer_tasks,
    prediction_dir,
    protein_prediction_tasks,
)

MODELS = {
    "protein": "SeonghwanSeo/atlasfold-260703",
    "protein-multimer": "SeonghwanSeo/atlasfold-m-260725",
}
logger = logging.getLogger(__name__)


def parse_args(kind: str, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Predict {kind} apo ensembles with AtlasFold"
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Dataset parent directory"
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    if kind == "protein-multimer":
        parser.add_argument("--groups_csv", type=Path, required=True)
    if kind == "protein":
        parser.add_argument(
            "--max_length",
            type=int,
            default=MAX_APO_LENGTH,
            help="Maximum monomer sequence length (default: 1280).",
        )
    else:
        parser.set_defaults(max_length=MAX_APO_LENGTH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--model", default=MODELS[kind])
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_tokens_per_batch", type=int, default=1280)
    parser.add_argument("--chunk", type=int, default=0, help="Slurm task/shard index")
    parser.add_argument("--num_chunk", type=int, default=1)
    parser.add_argument("--no_kernel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    if args.num_samples < 1 or args.max_tokens_per_batch < 1:
        parser.error("num_samples and max_tokens_per_batch must be positive")
    if args.max_length < 1:
        parser.error("max_length must be positive")
    if not 0 <= args.chunk < args.num_chunk:
        parser.error("Require 0 <= chunk < num_chunk")
    return args


def request_settings(args, kind: str) -> dict:
    return {
        "model": args.model,
        "num_samples": args.num_samples,
        "num_recycles": 4,
        "mlm_prob": 0.15 if kind == "protein" else 0.20,
        "num_steps": None if kind == "protein" else 100,
        "use_cuequiv_kernels": not args.no_kernel,
    }


def prediction_complete(
    directory: Path, task: ApoTask, seed: int, settings: dict
) -> bool:
    sample_indices = set()
    for rank in range(1, settings["num_samples"] + 1):
        pdb = directory / f"rank_{rank}.pdb"
        path = directory / f"rank_{rank}.json"
        if not pdb.is_file() or pdb.stat().st_size == 0 or not path.is_file():
            return False
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if (
            metadata.get("seed") != seed
            or metadata.get("rank") != rank
            or metadata.get("sequences") != list(task.sequences)
            or metadata.get("settings") != settings
        ):
            return False
        sample_index = metadata.get("sample_index")
        if not isinstance(sample_index, int):
            return False
        sample_indices.add(sample_index)
    return len(sample_indices) == settings["num_samples"]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_prediction(
    dataset_dir: Path,
    task: ApoTask,
    output,
    seed: int,
    settings: dict,
    max_length: int = MAX_APO_LENGTH,
):
    if task.length > max_length:
        raise ValueError(f"Apo input exceeds {max_length} residues: {task.name}")
    samples = [
        (index, sample)
        for (sample_seed, index), sample in output.outputs.items()
        if sample_seed == seed
    ]
    if len(samples) != settings["num_samples"]:
        raise ValueError(f"Unexpected sample count for {task.name}, seed {seed}")
    samples.sort(key=lambda pair: (-float(pair[1].ranking_score), pair[0]))
    directory = prediction_dir(dataset_dir, task, seed)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{task.name}-", dir=directory.parent))
    try:
        for rank, (index, sample) in enumerate(samples, start=1):
            model_type = "monomer" if task.kind == "protein" else "multimer"
            (temporary / f"rank_{rank}.pdb").write_text(sample.to_pdb(model=model_type))
            metadata = {
                **jsonable(sample.confidence_scores),
                "seed": seed,
                "rank": rank,
                "sample_index": index,
                "ranking_score": float(sample.ranking_score),
                "sequences": list(task.sequences),
                "settings": settings,
            }
            (temporary / f"rank_{rank}.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        if not prediction_complete(temporary, task, seed, settings):
            raise ValueError(f"Incomplete prediction output: {task.name}")
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_tasks(args, kind: str, tasks: list[ApoTask], dataset_dir: Path):
    # Enforce the cap at the model boundary too, including callers outside CLI.
    tasks = eligible_tasks(tasks, args.max_length)
    settings = request_settings(args, kind)
    pending = {
        seed: [
            task
            for task in tasks
            if args.overwrite
            or not prediction_complete(
                prediction_dir(dataset_dir, task, seed), task, seed, settings
            )
        ]
        for seed in args.seeds
    }
    if not any(pending.values()):
        return
    import torch
    from atlasfold.pretrained import load_model

    torch.set_float32_matmul_precision("highest")
    model = load_model(args.model, device=args.device, cache_dir=args.cache_dir)
    if args.no_kernel:
        model.set_forward_flags(use_cuequiv_kernels=False)
    if kind == "protein":
        from atlasfold.runner import FoldingInput, FoldingRunner

        runner = FoldingRunner(model)
        input_cls = FoldingInput
        extra = {}
    else:
        from atlasfold.model import SamplingConfig
        from atlasfold.runner_multimer import MultimerFoldingRunner, MultimerInput

        runner = MultimerFoldingRunner(model)
        input_cls = MultimerInput
        extra = {"sampling_config": SamplingConfig(num_steps=100)}
    for seed, seed_tasks in pending.items():
        by_name = {task.name: task for task in seed_tasks}
        inputs = [
            input_cls(
                task.name, task.sequences[0] if kind == "protein" else task.sequences
            )
            for task in seed_tasks
        ]
        if not inputs:
            continue
        with tqdm(
            total=len(seed_tasks),
            desc=f"{kind} seed={seed} chunk={args.chunk}",
            unit="seq",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as progress:
            for outputs in runner.fold_iter_batch(
                inputs,
                num_samples=args.num_samples,
                seeds=[seed],
                num_recycles=4,
                mlm_prob=settings["mlm_prob"],
                max_tokens_per_batch=args.max_tokens_per_batch,
                **extra,
            ):
                for output in outputs:
                    task = by_name.pop(output.name)
                    write_prediction(
                        dataset_dir, task, output, seed, settings, args.max_length
                    )
                    progress.update(1)
        if by_name:
            raise RuntimeError(f"Missing model outputs: {list(by_name)[:5]}")


def main(kind: str):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(kind)
    dataset_dir = args.data_dir / f"rcsb-{args.split}"
    tasks = (
        protein_prediction_tasks(dataset_dir)
        if kind == "protein"
        else multimer_tasks(dataset_dir, args.groups_csv)
    )
    eligible = eligible_tasks(tasks, args.max_length)
    skipped = [task.name for task in tasks if task.length > args.max_length]
    tasks = sorted(eligible, key=lambda task: (task.length, task.name))[
        args.chunk :: args.num_chunk
    ]
    logger.info(
        "%s: eligible=%d over_length=%d shard=%d seeds=%s",
        kind,
        len(eligible),
        len(skipped),
        len(tasks),
        args.seeds,
    )
    if args.dry_run:
        return
    report_dir = dataset_dir / "apo" / kind
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"skipped_length_{args.chunk}_{args.num_chunk}.json").write_text(
        json.dumps(skipped, indent=2) + "\n"
    )
    run_tasks(args, kind, tasks, dataset_dir)
