"""Build source-specific apo and prior stores from ranked AtlasFold predictions."""

import argparse
import json
import multiprocessing as mp
import shutil
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from kfold.data.utils.io.structure import read_protein_multimer_structure
from kfold.training.dataset.utils.apo_io import (
    pack_apo_multimer_record,
    pack_apo_record,
    pack_prior_multimer_stack_record,
    pack_prior_stack_record,
)
from kfold.training.preprocess.apo_preparation import (
    MAX_APO_LENGTH,
    ApoTask,
    multimer_tasks,
    prediction_dir,
    protein_tasks,
    source_name,
)
from kfold.training.preprocess.atlasfold_prediction import prediction_complete


def read_prediction(path: Path, task: ApoTask) -> dict[int, dict]:
    """Map AtlasFold output chains to prepared chain IDs, retaining the shared frame.

    AtlasFold writes chains in input order. Sequence checks also allow repeated
    sequences belonging to distinct prepared entities without ambiguous matching.
    """
    chains = list(read_protein_multimer_structure(path).values())
    if tuple(chain["seq"] for chain in chains) != task.sequences:
        raise ValueError(f"Prediction sequences/order differ from input: {path}")
    asym_ids = (0,) if task.kind == "protein" else task.asym_ids
    for chain in chains:
        if chain["coords"].shape != (len(chain["seq"]), 37, 3):
            raise ValueError(f"Invalid atom37 coordinates: {path}")
    return dict(zip(asym_ids, chains, strict=True))


def pack_task(task: ApoTask, dataset_dir: Path, seeds: list[int], num_samples: int):
    products = []
    common_settings = None
    for seed in seeds:
        directory = prediction_dir(dataset_dir, task, seed)
        metadata_path = directory / "rank_1.json"
        metadata = json.loads(metadata_path.read_text())
        settings = metadata["settings"]
        if settings["num_samples"] != num_samples or not prediction_complete(
            directory, task, seed, settings
        ):
            raise ValueError(f"Missing or inconsistent prediction ranks: {directory}")
        if common_settings is not None and settings != common_settings:
            raise ValueError(f"Prediction settings differ across seeds: {directory}")
        common_settings = settings
        samples = [
            read_prediction(directory / f"rank_{rank}.pdb", task)
            for rank in range(1, num_samples + 1)
        ]
        scores = [
            json.loads((directory / f"rank_{rank}.json").read_text())
            for rank in range(1, num_samples + 1)
        ]
        confidence = {
            key: [
                float(score[key]) if score.get(key) is not None else np.nan
                for score in scores
            ]
            for key in ("ptm", "avg_plddt")
        }
        source = source_name(task.kind, seed)
        names = [f"{source}/rank_{rank}" for rank in range(1, num_samples + 1)]
        stacks = {
            aid: {
                **chain,
                "coords": np.stack([sample[aid]["coords"] for sample in samples]),
            }
            for aid, chain in samples[0].items()
        }
        if task.kind == "protein":
            first = samples[0][0]
            apo = pack_apo_record(first["seq"], first["coords"], "protein")
            prior = pack_prior_stack_record(
                first["seq"], stacks[0]["coords"], "protein", names, **confidence
            )
        else:
            apo = pack_apo_multimer_record(samples[0])
            prior = pack_prior_multimer_stack_record(stacks, names, **confidence)
        products.append((source, apo, prior))
    return task, products


def eligible_lmdb_tasks(tasks: list[ApoTask], max_length: int) -> list[ApoTask]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return [
        task
        for task in tasks
        if 0 < task.length <= (max_length if task.kind == "protein" else MAX_APO_LENGTH)
    ]


def build_lmdbs(
    dataset_dir: Path,
    tasks: list[ApoTask],
    seeds: list[int],
    num_samples: int,
    num_workers: int = 1,
    map_size_gb: int = 128,
    overwrite: bool = False,
    max_length: int = MAX_APO_LENGTH,
):
    """Rebuild lookup and LMDB products; parse failures leave existing products intact."""
    tasks = eligible_lmdb_tasks(tasks, max_length)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Seeds must be distinct nonnegative integers")
    if min(num_samples, num_workers, map_size_gb) < 1:
        raise ValueError("num_samples, num_workers and map_size_gb must be positive")
    output_names = [
        "apo_lmdb",
        "prior_lmdb",
        "apo_lookup.msgpack",
        "apo_multimer_lookup.msgpack",
    ]
    existing = [name for name in output_names if (dataset_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Outputs already exist: {existing}; use --overwrite to rebuild"
        )

    with tempfile.TemporaryDirectory(prefix=".apo-build-", dir=dataset_dir) as temporary:
        staging = Path(temporary)
        for root in ("apo_lmdb", "prior_lmdb"):
            (staging / root).mkdir()
        lookup, multimer_lookup = {}, {}
        worker = partial(
            pack_task, dataset_dir=dataset_dir, seeds=seeds, num_samples=num_samples
        )
        with ExitStack() as context:
            if num_workers == 1:
                results = map(worker, tasks)
            else:
                pool = context.enter_context(mp.get_context("spawn").Pool(num_workers))
                results = pool.imap_unordered(worker, tasks)
            stores = {}
            for kind in sorted({task.kind for task in tasks}):
                for seed in seeds:
                    source = source_name(kind, seed)
                    for root in ("apo_lmdb", "prior_lmdb"):
                        path = staging / root / kind / f"{source}.lmdb"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        env = lmdb.open(str(path), map_size=map_size_gb * 1024**3)
                        context.callback(env.close)
                        stores[root, kind, source] = env
            for task, products in tqdm(
                results, total=len(tasks), desc="Building apo/prior stores"
            ):
                for source, apo, prior in products:
                    with stores["apo_lmdb", task.kind, source].begin(write=True) as txn:
                        if not txn.put(task.name.encode(), apo, overwrite=False):
                            raise ValueError(f"Duplicate apo key: {task.name}")
                    if task.kind == "protein":
                        prior_keys = [f"{entry}_{eid}" for entry, eid in task.targets]
                        for entry, eid in task.targets:
                            lookup.setdefault(entry, {}).setdefault(str(eid), []).append(
                                {
                                    "chain_type": "protein",
                                    "source": source,
                                    "name": task.name,
                                }
                            )
                    else:
                        prior_keys = [task.name]
                        multimer_lookup.setdefault(task.entry_id, []).append(
                            {
                                "chain_type": "protein",
                                "source": source,
                                "name": task.name,
                                "asym_ids": list(task.asym_ids),
                                "apo_uid": min(task.asym_ids),
                            }
                        )
                    with stores["prior_lmdb", task.kind, source].begin(write=True) as txn:
                        for key in prior_keys:
                            if not txn.put(key.encode(), prior, overwrite=False):
                                raise ValueError(f"Duplicate prior key: {key}")
        # Stable source order also makes seeded runtime selection reproducible.
        for entry in lookup.values():
            for records in entry.values():
                records.sort(key=lambda record: record["source"])
        for records in multimer_lookup.values():
            records.sort(key=lambda record: (record["name"], record["source"]))
        (staging / "apo_lookup.msgpack").write_bytes(msgpack.packb(lookup))
        (staging / "apo_multimer_lookup.msgpack").write_bytes(
            msgpack.packb(multimer_lookup)
        )

        backup = staging / "previous"
        backup.mkdir()
        installed = []
        try:
            for name in output_names:
                target = dataset_dir / name
                if target.exists():
                    target.rename(backup / name)
                (staging / name).rename(target)
                installed.append(name)
        except Exception:
            for name in installed:
                target = dataset_dir / name
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            for previous in backup.iterdir():
                previous.rename(dataset_dir / previous.name)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--groups_csv", type=Path, help="Optional prepared multimer groups"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_APO_LENGTH,
        help="Monomer length limit; match f1 --max_length (default: 1280).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--map_size_gb", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.data_dir / f"rcsb-{args.split}"
    tasks = protein_tasks(dataset_dir)
    if args.groups_csv:
        tasks += multimer_tasks(dataset_dir, args.groups_csv)
    print(
        f"Prediction tasks: {len(tasks)}; "
        f"eligible tasks: {len(eligible_lmdb_tasks(tasks, args.max_length))}; "
        f"seeds: {args.seeds}"
    )
    if not args.dry_run:
        build_lmdbs(
            dataset_dir,
            tasks,
            args.seeds,
            args.num_samples,
            args.num_workers,
            args.map_size_gb,
            args.overwrite,
            max_length=args.max_length,
        )
