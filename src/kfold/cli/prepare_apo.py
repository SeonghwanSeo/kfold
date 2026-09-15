"""Prepare apo inputs across all queries before complex prediction."""

import argparse
import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from functools import partial
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from time import perf_counter

import gemmi
import torch

from kfold.cli.multigpu import launch
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
    """One protein entry and its output directory for each pending CLI seed."""

    query_name: str
    sequence_index: int  # One-based position in query.sequences.
    entry: ProteinSequence | ProteinPair
    save_dirs: dict[int, Path] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Identifier returned with this input's AtlasFold predictions."""
        return f"{self.query_name}_{self.output_prefix}"

    @property
    def output_prefix(self) -> str:
        """Common prefix for this entry's apo, prior, and completion files."""
        return apo_output_prefix(
            self.sequence_index, multimer=isinstance(self.entry, ProteinPair)
        )

    @property
    def sequence(self) -> str | tuple[str, str]:
        """Sequence or sequence pair passed to AtlasFold."""
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


def run(args: argparse.Namespace, jobs: list[tuple[Query, int, Path]]) -> None:
    # Load apo settings and select query/seed jobs that still need preparation.
    config = ApoConfig.load(args.apo_config) if args.apo_config else ApoConfig()
    start = perf_counter()
    pending_jobs = [
        (query, seed, save_dir)
        for query, seed, save_dir in jobs
        if args.overwrite or not is_apo_prepared(query, save_dir)
    ]
    # Assign each entry to one GPU, keeping its pending seeds together.
    inputs_by_entry: dict[tuple[str, int], ApoInput] = {}
    for query, seed, save_dir in pending_jobs:
        for index, entry in enumerate(query.sequences, start=1):
            if not isinstance(entry, (ProteinSequence, ProteinPair)) or entry.apo:
                continue
            prefix = apo_output_prefix(index, multimer=isinstance(entry, ProteinPair))
            done = save_dir / "apo" / f"{prefix}.done"
            if done.is_file() and not args.overwrite:
                continue
            key = (query.name, index)
            if key not in inputs_by_entry:
                inputs_by_entry[key] = ApoInput(
                    query_name=query.name,
                    sequence_index=index,
                    entry=entry,
                )
            inputs_by_entry[key].save_dirs[seed] = save_dir
    apo_inputs = sorted(
        inputs_by_entry.values(),
        key=lambda item: (
            len(item.entry),
            item.entry.kind,
            item.query_name,
            item.sequence_index,
        ),
    )
    logger.info(
        "Apo preparation: %d jobs pending (%d complete); GPUs %s.",
        len(pending_jobs),
        len(jobs) - len(pending_jobs),
        args.gpu_ids[: min(len(args.gpu_ids), len(apo_inputs))],
    )
    if args.dry_run:
        for query, _, save_dir in jobs:
            if args.overwrite:
                continue
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
        return

    # Save provided structures and record output paths before GPU generation.
    for query, seed, save_dir in pending_jobs:
        apo_dir = save_dir.resolve() / "apo"
        apo_dir.mkdir(parents=True, exist_ok=True)
        # New inputs invalidate any predictions previously marked complete.
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
            "seed": seed,
            "num_apos": args.num_apos,
            **asdict(config),
        }
        (save_dir / "apo_setting.json").write_text(json.dumps(settings, indent=2) + "\n")

    # Generate the remaining apo entries in GPU workers.
    if apo_inputs:
        launch(
            partial(_worker, config=config),
            args,
            apo_inputs,
            stage="Apo preparation",
        )

    logger.info("Apo preparation complete in %.1f s.", perf_counter() - start)


def _worker(
    gpu_id: int,
    args: argparse.Namespace,
    inputs: list[ApoInput],
    *,
    config: ApoConfig,
) -> None:
    work = "model loading"
    try:
        # Initialize the shared language model before loading either folding model.
        runner = ApoRunner(
            torch.device("cuda", gpu_id),
            config=config,
            cache_dir=args.cache_dir,
            verbose=False,
        )
        # Process monomers and multimers separately to release each model afterward.
        inputs = sorted(inputs, key=lambda item: item.entry.kind)
        for kind, model_inputs in groupby(inputs, key=lambda item: item.entry.kind):
            model_inputs = list(model_inputs)
            model_name = "AtlasFold-Multimer" if kind == "protein_pair" else "AtlasFold"
            total = sum(len(item.save_dirs) for item in model_inputs)
            completed = 0
            model_start = perf_counter()
            logger.info("%s prediction started: %d jobs.", model_name, total)
            work = f"{model_name} model loading"
            # Keep this model for every CLI seed, then release it before the next type.
            runner.load_model(multimer=kind == "protein_pair")
            # Run CLI seeds separately, batching pending targets for each seed.
            for seed in args.seed:
                # Reuse the input metadata when AtlasFold returns results by length.
                inputs_by_name = {
                    item.name: item for item in model_inputs if seed in item.save_dirs
                }
                if not inputs_by_name:
                    continue
                apo_inputs = [
                    (name, item.sequence) for name, item in inputs_by_name.items()
                ]
                # Generate all requested apo seeds within each AtlasFold batch.
                apo_seeds = [seed * 10 + i for i in range(1, args.num_apos + 1)]
                work = f"{model_name} prediction (seeds={apo_seeds})"
                batch_start = perf_counter()
                for results in runner.predict_iter_batch(apo_inputs, apo_seeds):
                    time_per_job = (perf_counter() - batch_start) / len(results)
                    # Save both ensembles before marking an entry complete.
                    for result in results:
                        item = inputs_by_name[result.name]
                        save_dir = item.save_dirs[seed]
                        save_apo_and_prior(
                            result.apos,
                            result.priors,
                            save_dir,
                            item.sequence_index,
                        )
                        (save_dir / "apo" / f"{item.output_prefix}.done").touch()
                        completed += 1
                        logger.info(
                            "%s completed: %s (seeds=%s), length=%d, %.2f s [%d/%d].",
                            model_name,
                            item.name,
                            apo_seeds,
                            len(item.entry),
                            time_per_job,
                            completed,
                            total,
                        )
                    del result, results
                    batch_start = perf_counter()
            runner.unload_models()
            logger.info(
                "%s prediction completed: %d jobs, total=%.1f s.",
                model_name,
                completed,
                perf_counter() - model_start,
            )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"Apo GPU {gpu_id}: out of memory during {work}. "
            "Lower max_tokens_per_batch in the apo YAML configuration."
        ) from None
