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

"""Predict complex structures from prepared apo inputs."""

import argparse
import csv
import json
import logging
import math
import shutil
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub.utils import disable_progress_bars

from kfold import __version__
from kfold.cli.multigpu import launch
from kfold.cli.prepare_apo import is_apo_prepared
from kfold.inference.query import Query
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold
from kfold.utils.runtime import is_cuequivariance_installed, is_triton_available

logger = logging.getLogger("kfold.predict")


def _load_prepared_query(query: Query, apo_dir: Path) -> Query:
    """Load a completed preparation and validate its target and protein entries."""
    if not is_apo_prepared(query, apo_dir):
        raise ValueError(
            f"Apo preparation is incomplete: {apo_dir}. "
            "Run kfold --stage apo with the same inputs and --share-apo-seeds "
            "setting first, and the same seeds unless apos are shared."
        )
    prepared = Query.load(apo_dir / "query.json")
    if prepared.name != query.name:
        raise ValueError(f"Prepared query name does not match {query.name}.")
    if any(not entry.apo for entry in prepared.protein_entries):
        raise ValueError(f"Prepared query has missing apo structures: {apo_dir}.")
    return prepared


def dry_run(args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]) -> None:
    """Validate saved preparations and count pending inference jobs without writing."""
    pending = 0
    for query, _, job_dir in jobs:
        apo_dir = job_dir.parent if args.share_apo_seeds is not None else job_dir
        apo_ready = is_apo_prepared(query, apo_dir)
        # Check saved apo/prior paths even for completed dry-run jobs.
        if apo_ready and not (args.stage == "all" and args.overwrite):
            Query.load(apo_dir / "query.json")

        if (job_dir / "done.txt").is_file() and not args.overwrite and apo_ready:
            continue

        # A default dry run includes queries whose preparation has not run yet.
        if args.stage == "complex":
            _load_prepared_query(query, apo_dir)
        pending += 1

    logger.info(
        "K-Fold inference: %d jobs pending (%d complete); GPUs %s.",
        pending,
        len(jobs) - pending,
        args.gpu_ids[: min(len(args.gpu_ids), pending)],
    )


def run(args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]) -> None:
    """Predict unfinished complex jobs and summarize their ranked outputs."""
    if args.kernel_backend is None:
        if is_triton_available():
            args.kernel_backend = "triton"
        elif is_cuequivariance_installed():
            args.kernel_backend = "cuequiv"
        else:
            args.kernel_backend = "torch"
    logger.info("K-Fold kernel backend: %s.", args.kernel_backend)
    start = perf_counter()
    pending = []
    for query, seed, job_dir in jobs:
        apo_dir = job_dir.parent if args.share_apo_seeds is not None else job_dir
        if (
            (job_dir / "done.txt").is_file()
            and not args.overwrite
            and is_apo_prepared(query, apo_dir)
        ):
            continue
        query = _load_prepared_query(query, apo_dir)
        pending.append((query, seed, job_dir))

    logger.info(
        "K-Fold inference: %d jobs pending (%d complete); GPUs %s.",
        len(pending),
        len(jobs) - len(pending),
        args.gpu_ids[: min(len(args.gpu_ids), len(pending))],
    )
    # Invalidate previous completion records and run the pending GPU jobs.
    if pending:
        for _, _, job_dir in pending:
            (job_dir / "done.txt").unlink(missing_ok=True)
        num_workers = min(len(args.gpu_ids), len(pending))
        worker_jobs = [pending[rank::num_workers] for rank in range(num_workers)]
        launch(_worker, args, worker_jobs, stage="K-Fold inference")

    # Rank all requested seeds and save one representative result per query.
    logger.info("Summarizing predictions.")
    for name in sorted({query.name for query, _, _ in jobs}):
        logger.info("Ranking samples and saving best prediction for %s.", name)
        summary_start = perf_counter()
        _summarize_predictions(args.out_dir / name, name, args.seeds, args.num_samples)
        # Record the K-Fold settings requested in this CLI invocation.
        settings = {
            "version": __version__,
            "seeds": args.seeds,
            "shared_apo": args.share_apo_seeds is not None,
            "num_samples": args.num_samples,
            "num_recycles": args.num_recycles,
            "num_steps": args.num_steps,
            "use_struct_encoder": not args.disable_struct_encoder,
            "use_rna_encoder": not args.disable_rna_encoder,
            "cpu_offload": args.cpu_offload,
            "kernel_backend": args.kernel_backend,
        }
        (args.out_dir / name / "kfold_settings.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        logger.info(
            "Saved ranked results for %s to %s in %.1f s.",
            name,
            args.out_dir / name,
            perf_counter() - summary_start,
        )
    logger.info("K-Fold inference complete in %.1f s.", perf_counter() - start)


def _summarize_predictions(
    out_dir: Path, name: str, seeds: list[int], num_samples: int
) -> None:
    """Rank requested seeds and copy the best prediction."""
    # Collect confidence summaries from completed prediction jobs.
    records = []
    for seed in seeds:
        job_dir = out_dir / f"{name}_seed-{seed}"
        if not (job_dir / "done.txt").is_file():
            raise ValueError(f"Prediction job is incomplete: {job_dir}.")
        prefix = f"{job_dir.name}_sample-"
        for sample in range(num_samples):
            path = job_dir / f"{prefix}{sample}_confidence.json"
            if not path.is_file():
                continue
            with path.open() as f:
                scores = json.load(f)["complex"]
            if not math.isfinite(scores["ranking_score"]):
                raise ValueError(f"Non-finite ranking score in {path}.")
            records.append({"seed": seed, "sample": sample, **scores})

    if not records:
        raise ValueError(f"No saved confidence summaries for {name} in {out_dir}.")

    # Rank samples and copy the best structure and available confidence outputs.
    records.sort(key=lambda row: (-row["ranking_score"], row["seed"], row["sample"]))
    best = records[0]
    rank_dir = out_dir / f"{name}_seed-{best['seed']}"
    prefix = f"{name}_seed-{best['seed']}_sample-{best['sample']}"
    for suffix in ("model.cif", "confidence.json"):
        shutil.copyfile(rank_dir / f"{prefix}_{suffix}", out_dir / f"{name}_{suffix}")

    confidence = rank_dir / f"{prefix}_confidence.npz"
    best_confidence = out_dir / f"{name}_confidence.npz"
    if confidence.is_file():
        shutil.copyfile(confidence, best_confidence)
    else:
        best_confidence.unlink(missing_ok=True)

    # Write the complete ranking across seeds and samples.
    with (out_dir / f"{name}_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "sample",
                "ranking_score",
                "plddt",
                "ptm",
                "iptm",
                "pde",
                "has_clash",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def _worker(
    gpu_id: int, args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]
) -> None:
    work = "model loading"
    try:
        # Load one model and runner for all jobs assigned to this GPU.
        logger.info("Loading K-Fold model and runner.")
        init_start = perf_counter()
        with disable_progress_bars():
            model = KFold.from_pretrained(
                device=torch.device("cuda", gpu_id),
                cache_dir=args.cache_dir,
                use_struct_encoder=not args.disable_struct_encoder,
                use_rna_encoder=not args.disable_rna_encoder,
                cpu_offload=args.cpu_offload,
                kernel_backend=args.kernel_backend,
            )
            runner = KFoldRunner(model, cache_dir=args.cache_dir, verbose=False)
        logger.info(
            "Model and runner initialized in %.1f s.", perf_counter() - init_start
        )
        for job_index, (query, seed, job_dir) in enumerate(jobs, start=1):
            progress = f"[{job_index}/{len(jobs)}]"
            target = f"{query.name} (seed={seed})"
            job_start = perf_counter()
            logger.info("Job started: %s %s.", target, progress)
            # Load aligned apo and prior candidates from the prepared query.
            work = f"structure loading for {target}"
            logger.info("Structure loading started.")
            stage_start = perf_counter()
            apos, priors = runner.load_apo_and_prior(query)
            logger.info(
                "Structure loading completed in %.1f s.",
                perf_counter() - stage_start,
            )
            # Encode apo structures and construct the complex model input.
            work = f"featurization for {target}"
            logger.info("Featurization started.")
            stage_start = perf_counter()
            item = runner.build_input(
                query, seed, args.num_samples, apos=apos, priors=priors
            )
            logger.info(
                "Featurization completed: chains=%d, tokens=%d, atoms=%d, %.1f s.",
                item.ref_struct.num_chains,
                item.ref_struct.num_tokens,
                item.ref_struct.num_atoms,
                perf_counter() - stage_start,
            )
            # Run complex prediction and report the highest-ranked sample.
            work = f"K-Fold prediction for {target}"
            logger.info("Inference started.")
            stage_start = perf_counter()
            result = runner.predict_from_input(
                item,
                seed=seed,
                num_samples=args.num_samples,
                num_recycles=args.num_recycles,
                num_steps=args.num_steps,
                return_trajectory=args.save_trajectory,
                return_embeddings=args.save_embeddings,
                return_distogram=args.save_distogram,
            )
            best_sample, best_summary = max(
                enumerate(result.confidence_summary),
                key=lambda item: item[1]["complex"]["ranking_score"],
            )
            scores = best_summary["complex"]
            logger.info(
                "Inference completed: %.1f s, best sample=%d, "
                "ipTM=%.3f, pTM=%.3f, pLDDT=%.2f, ranking_score=%.3f.",
                perf_counter() - stage_start,
                best_sample,
                scores["iptm"],
                scores["ptm"],
                scores["plddt"],
                scores["ranking_score"],
            )
            # Save outputs before marking the query/seed job complete.
            work = f"output saving for {target}"
            logger.info("Saving started.")
            stage_start = perf_counter()
            result.settings["shared_apo"] = args.share_apo_seeds is not None
            result.save(
                job_dir,
                save_confidence=args.save_confidence,
                save_embeddings=args.save_embeddings,
                save_distogram=args.save_distogram,
                save_trajectory=args.save_trajectory,
            )
            (job_dir / "done.txt").touch()
            logger.info(
                "Saving completed in %.1f s.",
                perf_counter() - stage_start,
            )
            del result, item, apos, priors
            logger.info(
                "Job completed in %.1f s: %s %s.",
                perf_counter() - job_start,
                target,
                progress,
            )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"K-Fold GPU {gpu_id}: out of memory during {work}. "
            "Enable --cpu-offload to reduce inference memory use."
        ) from None
