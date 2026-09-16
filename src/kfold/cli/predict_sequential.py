"""Predict assembly plans from apo inputs prepared by the standard CLI."""

import argparse
import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub.utils import disable_progress_bars

from kfold import __version__
from kfold.cli.multigpu import launch
from kfold.cli.prepare_apo import is_apo_prepared
from kfold.inference.query import Query

logger = logging.getLogger("kfold.sequential")


def _target_definition(query: Query) -> dict:
    """Serialize query identity while excluding rewritten preparation paths."""
    data = query.to_dict()
    for entry in data["sequences"]:
        fields = next(iter(entry.values()))
        fields.pop("apo", None)
        fields.pop("prior", None)
    return data


def _apo_dir(args: argparse.Namespace, query: Query, seed: int) -> Path:
    """Return the standard CLI preparation directory for one inference seed."""
    root = args.out_dir / query.name
    if args.share_apo_seeds is not None:
        return root
    return root / f"{query.name}_seed-{seed}"


def _load_prepared_path(args: argparse.Namespace, query: Query, seed: int) -> Path:
    """Validate a prepared query and return its serialized query path."""
    apo_dir = _apo_dir(args, query, seed)
    if not is_apo_prepared(query, apo_dir):
        raise ValueError(
            f"Apo preparation is incomplete: {apo_dir}. Run kfold --stage apo "
            "with the same inputs, seeds, and --share-apo-seeds setting first."
        )
    path = apo_dir / "query.json"
    prepared = Query.load(path)
    if _target_definition(prepared) != _target_definition(query):
        raise ValueError(f"Prepared query does not match {query.name}: {path}.")
    if any(not entry.apo for entry in prepared.protein_entries):
        raise ValueError(f"Prepared query has missing apo structures: {path}.")
    return path


def _settings(args: argparse.Namespace) -> dict:
    """Settings that must remain identical when resuming a sequential query."""
    provided = getattr(args, "provided_intermediates", None)
    return {
        "version": __version__,
        "seeds": args.seeds,
        "shared_apo": args.share_apo_seeds is not None,
        "num_samples": args.num_samples,
        "num_recycles": args.num_recycles,
        "num_steps": args.num_steps,
        "use_struct_encoder": not args.disable_struct_encoder,
        "use_rna_encoder": not args.disable_rna_encoder,
        "cpu_offload": args.cpu_offload,
        "conditioning": args.conditioning,
        "provided_intermediates": (
            str(provided.resolve()) if provided is not None else None
        ),
    }


def _sequential_dir(args: argparse.Namespace, query: Query) -> Path:
    return args.out_dir / query.name / "sequential"


def _provided_paths(args: argparse.Namespace, query: Query) -> dict[str, Path]:
    root = args.provided_intermediates
    if root is None:
        return {}
    paths = {
        stage["id"]: root / query.name / f"{stage['id']}.npz"
        for stage in query.assembly["stages"]
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing provided intermediate: {path}")
    return paths


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _prepared_hashes(paths: dict[int, Path]) -> dict[str, dict[str, str]]:
    """Hash prepared queries and every apo/prior ensemble they reference."""
    result = {}
    for seed, query_path in paths.items():
        prepared = Query.load(query_path)
        files = {query_path.resolve()}
        for entry in prepared.protein_entries:
            files.update(path.resolve() for path in (entry.apo or []))
            files.update(path.resolve() for path in (entry.prior or []))
        result[str(seed)] = {str(path): _sha256(path) for path in sorted(files)}
    return result


def _is_complete(
    args: argparse.Namespace,
    query: Query,
    prepared_paths: dict[int, Path],
) -> bool:
    root = _sequential_dir(args, query)
    settings_path = root / "settings.json"
    completed = root / "completed.json"
    if not settings_path.is_file() and not completed.is_file():
        return False
    if not settings_path.is_file():
        raise ValueError(f"Sequential settings are missing: {settings_path}.")
    if json.loads(settings_path.read_text()) != _settings(args):
        raise ValueError(
            f"Sequential settings changed for {query.name}; rerun with --overwrite."
        )
    hashes_path = root / "prepared_hashes.json"
    if not hashes_path.is_file():
        if not completed.is_file():
            return False
        raise ValueError(f"Prepared apo hashes are missing: {hashes_path}.")
    if json.loads(hashes_path.read_text()) != _prepared_hashes(prepared_paths):
        raise ValueError(
            f"Prepared apo inputs changed for {query.name}; rerun with --overwrite."
        )
    for stage, path in _provided_paths(args, query).items():
        receipt = root / stage / "provided_structure.json"
        if not receipt.is_file() and not completed.is_file():
            continue
        if not receipt.is_file() or json.loads(receipt.read_text())["sha256"] != _sha256(
            path
        ):
            raise ValueError(
                f"Provided intermediate changed for {query.name}; rerun with --overwrite."
            )
    published = args.out_dir / query.name / f"{query.name}_model.cif"
    return completed.is_file() and published.is_file()


def dry_run(args: argparse.Namespace, queries: list[Query]) -> None:
    """Validate saved preparations and count pending sequential queries."""
    pending = 0
    for query in queries:
        _provided_paths(args, query)
        prepared = {}
        apo_ready = True
        for seed in args.seeds:
            if is_apo_prepared(query, _apo_dir(args, query, seed)):
                prepared[seed] = _load_prepared_path(args, query, seed)
            else:
                apo_ready = False
        if apo_ready and not (args.stage == "all" and args.overwrite):
            _prepared_hashes(prepared)
        if apo_ready and not args.overwrite and _is_complete(args, query, prepared):
            continue
        if args.stage == "complex":
            for seed in args.seeds:
                _load_prepared_path(args, query, seed)
        pending += 1
    logger.info(
        "Sequential K-Fold inference: %d queries pending (%d complete); GPUs %s.",
        pending,
        len(queries) - pending,
        args.gpu_ids[: min(len(args.gpu_ids), pending)],
    )


def run(args: argparse.Namespace, queries: list[Query]) -> None:
    """Run unfinished assembly queries, keeping all seeds together per query."""
    start = perf_counter()
    pending = []
    for query in queries:
        _provided_paths(args, query)
        prepared = {seed: _load_prepared_path(args, query, seed) for seed in args.seeds}
        if not args.overwrite and _is_complete(args, query, prepared):
            continue
        root = _sequential_dir(args, query)
        if args.overwrite and root.exists():
            shutil.rmtree(root)
        pending.append((query, prepared))

    logger.info(
        "Sequential K-Fold inference: %d queries pending (%d complete); GPUs %s.",
        len(pending),
        len(queries) - len(pending),
        args.gpu_ids[: min(len(args.gpu_ids), len(pending))],
    )
    if pending:
        num_workers = min(len(args.gpu_ids), len(pending))
        worker_jobs = [pending[rank::num_workers] for rank in range(num_workers)]
        launch(_worker, args, worker_jobs, stage="Sequential K-Fold inference")
    logger.info("Sequential K-Fold inference complete in %.1f s.", perf_counter() - start)


def _publish_top1(args: argparse.Namespace, query: Query, best: dict) -> None:
    """Expose the final top-ranked prediction in the standard query directory."""
    target = args.out_dir / query.name
    source_cif = Path(best["cif"])
    source_dir = source_cif.parent
    stem = best["stem"]
    copies = {
        source_cif: target / f"{query.name}_model.cif",
        source_dir / f"{stem}_confidences.json": target / f"{query.name}_confidence.json",
        source_dir / f"{stem}_confidences.npz": target / f"{query.name}_confidence.npz",
    }
    for source, destination in copies.items():
        if source.is_file():
            shutil.copyfile(source, destination)
        else:
            destination.unlink(missing_ok=True)
    with (target / "sequential" / "candidates.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == "final"]
    rows.sort(
        key=lambda row: (
            -float(row["ranking_score"]),
            int(row["seed"]),
            int(row["sample"]),
        )
    )
    fields = [
        "seed",
        "sample",
        "ranking_score",
        "plddt",
        "ptm",
        "iptm",
        "pde",
        "has_clash",
    ]
    with (target / f"{query.name}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (target / "kfold_settings.json").write_text(
        json.dumps(_settings(args), indent=2) + "\n"
    )


def _worker(
    gpu_id: int,
    args: argparse.Namespace,
    jobs: list[tuple[Query, dict[int, Path]]],
) -> None:
    """Load one model and run complete assembly plans assigned to one GPU."""
    from kfold.inference.runner import KFoldRunner
    from kfold.inference.sequential import (
        ModelBackend,
        provided_stage_paths,
        run_query,
        write_json,
    )
    from kfold.inference.sequential_pipeline import InputDataPipeline
    from kfold.inference.sequential_query import parse_single_file
    from kfold.model import KFold

    work = "model loading"
    try:
        logger.info("Loading K-Fold model and sequential pipeline.")
        with disable_progress_bars():
            model = KFold.from_pretrained(
                device=torch.device("cuda", gpu_id),
                cache_dir=args.cache_dir,
                use_struct_encoder=not args.disable_struct_encoder,
                use_rna_encoder=not args.disable_rna_encoder,
                cpu_offload=args.cpu_offload,
            )
            runner = KFoldRunner(model, cache_dir=args.cache_dir, verbose=False)
        pipeline = InputDataPipeline(runner.ccd, args.num_samples)
        backend = ModelBackend(args, model=model)

        for index, (native_query, paths) in enumerate(jobs, start=1):
            work = f"sequential prediction for {native_query.name}"
            logger.info(
                "Sequential job started: %s [%d/%d].",
                native_query.name,
                index,
                len(jobs),
            )
            source_queries = {
                seed: parse_single_file(path, runner.ccd).copy(seed=seed)
                for seed, path in paths.items()
            }
            query = source_queries[args.seeds[0]]
            root = _sequential_dir(args, native_query)
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "settings.json", _settings(args))
            write_json(root / "prepared_hashes.json", _prepared_hashes(paths))
            provided = (
                provided_stage_paths(query, args.provided_intermediates)
                if args.provided_intermediates is not None
                else None
            )
            best = run_query(
                query,
                pipeline,
                args.seeds,
                args.num_samples,
                root,
                backend,
                conditioning=args.conditioning,
                provided_intermediates=provided,
                source_queries=source_queries,
            )
            _publish_top1(args, native_query, best)
            logger.info("Sequential job completed: %s.", native_query.name)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"K-Fold GPU {gpu_id}: out of memory during {work}. "
            "Enable --cpu-offload to reduce inference memory use."
        ) from None
