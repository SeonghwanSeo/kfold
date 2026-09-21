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

"""Prepare apo inputs across all queries before complex prediction."""

import argparse
import copy
import json
import logging
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import gemmi
import torch

from kfold.cli.multigpu import distribute, launch
from kfold.data.utils.io.structure import read_gemmi_structure
from kfold.inference.apo_runner import (
    ApoConfig,
    ApoRunner,
    apo_output_prefix,
    save_apo_and_prior,
)
from kfold.inference.query import ProteinPair, ProteinSequence, Query

logger = logging.getLogger("kfold.apo")


@dataclass
class ApoInput:
    query_name: str
    sequence_index: int
    entry: ProteinSequence | ProteinPair

    @property
    def name(self) -> str:
        return f"{self.query_name}_{self.output_prefix}"

    @property
    def output_prefix(self) -> str:
        return apo_output_prefix(
            self.sequence_index, multimer=isinstance(self.entry, ProteinPair)
        )

    @property
    def sequence(self) -> str | tuple[str, str]:
        """Sequence or sequence pair."""
        if isinstance(self.entry, ProteinPair):
            return self.entry.sequence1, self.entry.sequence2
        return self.entry.sequence


def is_apo_prepared(query: Query, save_dir: Path) -> bool:
    """Check for a saved query and completion of each protein sequence entry."""
    if not (save_dir / "query.json").is_file():
        return False
    for index, entry in enumerate(query.sequences, start=1):
        if not isinstance(entry, (ProteinSequence, ProteinPair)):
            continue
        prefix = apo_output_prefix(index, multimer=isinstance(entry, ProteinPair))
        if not (save_dir / "apo" / f"{prefix}.done").is_file():
            return False
    return True


def _save_pdb_ensemble(paths: list[Path], save_path: Path) -> None:
    """Combine provided structures into one PDB with consecutive model numbers."""
    ensemble = gemmi.Structure()
    for path in paths:
        for model in read_gemmi_structure(path):
            model = model.clone()
            model.num = len(ensemble) + 1
            ensemble.add_model(model)
    if not len(ensemble):
        raise ValueError(f"Cannot save an empty ensemble to {save_path}.")
    ensemble.write_pdb(str(save_path))


def _pending_inputs(
    args: argparse.Namespace, jobs: list[tuple[Query, int | None, Path]]
) -> tuple[list[tuple[Query, int | None, Path]], dict[int | None, list[ApoInput]]]:
    """Select unfinished jobs and group entries needing generation by CLI seed."""
    pending_jobs = [
        (query, seed, save_dir)
        for query, seed, save_dir in jobs
        if args.overwrite or not is_apo_prepared(query, save_dir)
    ]
    inputs: dict[int | None, list[ApoInput]] = {}
    for query, seed, save_dir in pending_jobs:
        for index, entry in enumerate(query.sequences, start=1):
            if not isinstance(entry, (ProteinSequence, ProteinPair)) or entry.apo:
                continue
            prefix = apo_output_prefix(index, multimer=isinstance(entry, ProteinPair))
            done = save_dir / "apo" / f"{prefix}.done"
            if done.is_file() and not args.overwrite:
                continue
            inputs.setdefault(seed, []).append(
                ApoInput(query_name=query.name, sequence_index=index, entry=entry)
            )
    num_entries = len({item.sequence for items in inputs.values() for item in items})
    logger.info(
        "Apo preparation: %d jobs pending (%d complete); GPUs %s.",
        len(pending_jobs),
        len(jobs) - len(pending_jobs),
        args.gpu_ids[: min(len(args.gpu_ids), num_entries)],
    )
    return pending_jobs, inputs


def _distribute_inputs(
    inputs: dict[int | None, list[ApoInput]], num_gpus: int, method: str
) -> list[dict[int | None, list[ApoInput]]]:
    """Distribute entries by length, keeping identical sequences on the same worker."""
    entries = {item.sequence: item for items in inputs.values() for item in items}
    ordered_entries = sorted(
        entries.values(),
        key=lambda item: (
            len(item.entry),
            item.entry.kind,
            item.query_name,
            item.sequence_index,
        ),
    )
    entry_order = {item.sequence: index for index, item in enumerate(ordered_entries)}
    num_workers = min(num_gpus, len(entries))
    costs = [0] * len(ordered_entries)
    for items in inputs.values():
        for sequence in {item.sequence for item in items}:
            index = entry_order[sequence]
            costs[index] += len(ordered_entries[index].entry) ** 2
    groups = distribute(costs, num_workers, method)
    entry_ranks = {index: rank for rank, group in enumerate(groups) for index in group}
    worker_inputs: list[dict[int | None, list[ApoInput]]] = [
        {} for _ in range(num_workers)
    ]
    for seed, items in inputs.items():
        for item in sorted(items, key=lambda item: entry_order[item.sequence]):
            rank = entry_ranks[entry_order[item.sequence]]
            worker_inputs[rank].setdefault(seed, []).append(item)
    return worker_inputs


def dry_run(args: argparse.Namespace, jobs: list[tuple[Query, int | None, Path]]) -> None:
    """Check apo configuration and reusable structures, and report pending jobs."""
    if args.apo_config:
        ApoConfig.load(args.apo_config)
    _pending_inputs(args, jobs)
    if args.overwrite:
        return

    for query, _, save_dir in jobs:
        # Validate reusable entries even when other entries are unfinished.
        for index, entry in enumerate(query.sequences, start=1):
            if not isinstance(entry, (ProteinSequence, ProteinPair)):
                continue
            prefix = apo_output_prefix(index, multimer=isinstance(entry, ProteinPair))
            if not (save_dir / "apo" / f"{prefix}.done").is_file():
                continue
            for role in ("apo", "prior"):
                path = save_dir / "apo" / f"{prefix}_{role}.pdb"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Prepared {role} structure does not exist: {path}."
                    )


def run(args: argparse.Namespace, jobs: list[tuple[Query, int | None, Path]]) -> None:
    """Prepare provided structures and generate unfinished apo entries."""
    config = ApoConfig.load(args.apo_config) if args.apo_config else ApoConfig()
    start = perf_counter()
    pending_jobs, inputs = _pending_inputs(args, jobs)

    # Save provided structures and record output paths before GPU generation.
    for query, seed, save_dir in pending_jobs:
        apo_dir = save_dir.resolve() / "apo"
        apo_dir.mkdir(parents=True, exist_ok=True)
        # New inputs invalidate any predictions previously marked complete.
        if seed is None:
            for path in save_dir.glob(f"{query.name}_seed-*/done.txt"):
                path.unlink()
        else:
            (save_dir / "done.txt").unlink(missing_ok=True)
        if args.overwrite:
            # Invalidate every entry before rewriting any provided structures.
            for kind in ("monomer", "multimer"):
                for path in apo_dir.glob(f"{kind}-*.done"):
                    path.unlink()
        prepared = copy.deepcopy(query)
        for index, entry in enumerate(prepared.sequences, start=1):
            if not isinstance(entry, (ProteinSequence, ProteinPair)):
                continue
            prefix = apo_output_prefix(index, multimer=isinstance(entry, ProteinPair))
            done = apo_dir / f"{prefix}.done"
            if entry.apo and not done.is_file():
                _save_pdb_ensemble(entry.apo, apo_dir / f"{prefix}_apo.pdb")
                prior_paths = entry.prior if entry.prior is not None else entry.apo
                _save_pdb_ensemble(prior_paths, apo_dir / f"{prefix}_prior.pdb")
                done.touch()
            entry.apo = [apo_dir / f"{prefix}_apo.pdb"]
            entry.prior = [apo_dir / f"{prefix}_prior.pdb"]
        # Paths are known before generation; per-sequence markers track readiness.
        prepared.save(save_dir / "query.json")
        settings = {
            "version": version("atlasfold"),
            "num_apos": len(args.share_apo_seeds) if seed is None else args.num_apos,
            **asdict(config),
        }
        if seed is None:
            settings["seeds"] = args.share_apo_seeds
        else:
            settings["seed"] = seed
        (save_dir / "apo_setting.json").write_text(json.dumps(settings, indent=2) + "\n")

    # Generate the remaining apo entries in GPU workers.
    if inputs:
        worker_inputs = _distribute_inputs(inputs, len(args.gpu_ids), args.distribution)
        launch(
            partial(_worker, config=config),
            args,
            worker_inputs,
            stage="Apo preparation",
        )

    logger.info("Apo preparation complete in %.1f s.", perf_counter() - start)


def _worker(
    gpu_id: int,
    args: argparse.Namespace,
    inputs: dict[int | None, list[ApoInput]],
    *,
    config: ApoConfig,
) -> None:
    try:
        # Initialize the shared language model before loading either folding model.
        work = "apo runner initialization"
        runner = ApoRunner(
            torch.device("cuda", gpu_id),
            kernel_backend=args.kernel_backend,
            config=config,
            cache_dir=args.cache_dir,
            verbose=False,
        )

        # Process monomers and multimers separately to release each model afterward.
        kinds = sorted({item.entry.kind for items in inputs.values() for item in items})
        for kind in kinds:
            model_name = "AtlasFold-Multimer" if kind == "protein_pair" else "AtlasFold"

            # Count output entries and unique prediction targets separately per seed.
            total = sum(
                item.entry.kind == kind for items in inputs.values() for item in items
            )
            unique_total = sum(
                len({item.sequence for item in items if item.entry.kind == kind})
                for items in inputs.values()
            )
            completed = 0
            logger.info(
                "%s prediction started: %d targets (%d unique).",
                model_name,
                total,
                unique_total,
            )

            # Load the apo folding model
            model_start = perf_counter()
            work = f"{model_name} model loading"
            runner.load_model(multimer=kind == "protein_pair")

            # Shared preparation batches each target once, independently of CLI seeds.
            for seed, seed_inputs in inputs.items():
                # Predict each sequence once per seed and retain every output target.
                inputs_by_sequence: dict[str | tuple[str, str], list[ApoInput]] = {}
                for item in seed_inputs:
                    if item.entry.kind == kind:
                        inputs_by_sequence.setdefault(item.sequence, []).append(item)
                inputs_by_name = {
                    items[0].name: items for items in inputs_by_sequence.values()
                }
                if not inputs_by_name:
                    continue
                apo_inputs = [
                    (name, items[0].sequence) for name, items in inputs_by_name.items()
                ]
                # Generate all requested apo seeds within each AtlasFold batch.
                apo_seeds = (
                    args.share_apo_seeds
                    if seed is None
                    else [seed * 10 + i for i in range(1, args.num_apos + 1)]
                )
                work = f"{model_name} prediction (seeds={apo_seeds})"
                batch_start = perf_counter()
                for results in runner.predict_iter_batch(apo_inputs, apo_seeds):
                    time_per_job = (perf_counter() - batch_start) / len(results)

                    # Save pdbs
                    for result in results:
                        for item in inputs_by_name[result.name]:
                            save_dir = args.out_dir / item.query_name
                            if seed is not None:
                                save_dir /= f"{item.query_name}_seed-{seed}"
                            save_apo_and_prior(
                                result.apos, result.priors, save_dir, item.sequence_index
                            )
                            (save_dir / "apo" / f"{item.output_prefix}.done").touch()
                        completed += 1
                        logger.info(
                            "%s completed: %s (seeds=%s), length=%d, %.2f s [%d/%d].",
                            model_name,
                            result.name,
                            apo_seeds,
                            len(inputs_by_name[result.name][0].entry),
                            time_per_job,
                            completed,
                            unique_total,
                        )
                    del result, results
                    batch_start = perf_counter()

            # Free GPU memory after prediction
            runner.unload_models()

            logger.info(
                "%s prediction completed: %d targets (%d unique), total=%.1f s.",
                model_name,
                total,
                completed,
                perf_counter() - model_start,
            )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(
            f"Apo GPU {gpu_id}: out of memory during {work}. "
            "Lower max_tokens_per_batch in the apo YAML configuration."
        )
        raise e
