"""Create apo/prior multimer LMDBs from antibody H/L sampler outputs."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import pathlib
import shutil
from collections import defaultdict
from dataclasses import dataclass

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from kfold.data.types.metadata import Metadata
from kfold.data.utils.io.structure import read_protein_multimer_structure
from kfold.training.dataset.utils.apo_io import (
    pack_apo_multimer_record,
    pack_prior_multimer_stack_record,
)


@dataclass(frozen=True)
class PairInfo:
    pdb_id: str
    h_label: str
    l_label: str
    h_asym_id: int
    l_asym_id: int
    h_sequence: str
    l_sequence: str


@dataclass(frozen=True)
class ApoMultimerTask:
    key: str
    path: str
    pair: PairInfo


@dataclass(frozen=True)
class PriorMultimerTask:
    key: str
    paths: tuple[str, ...]
    sample_names: tuple[str, ...]
    confidence_paths: tuple[str | None, ...]
    pair: PairInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build apo_multimer_lmdb/protein/{source}.lmdb from ranked H/L "
            "complexes and prior_multimer_lmdb/protein/{source}.lmdb from "
            "non-ranked sample stacks."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to processed dataset root.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="RCSB split to process.",
    )
    parser.add_argument(
        "--source",
        default="prot_m_sampler",
        help="Multimer source name used for LMDB file names and sample names.",
    )
    parser.add_argument(
        "--source_dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory containing {pdb_id}_{Hlabel}_{Llabel}/ sampler outputs. "
            "Defaults to rcsb-*/apo_multimer/protein/{source}."
        ),
    )
    parser.add_argument(
        "--sabdab_csv",
        type=pathlib.Path,
        default=None,
        help="CSV with SAbDab H/L label asym IDs and sequences.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parser workers.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=128,
        help="LMDB map size in GB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing multimer LMDB outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Collect and validate inputs without writing LMDBs.",
    )
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> dict[str, Metadata]:
    with path.open("rb") as f:
        metadata_dicts = msgpack.unpack(f, raw=False)
    return {m.id: m for m in (Metadata.from_dict(d) for d in metadata_dicts)}


def chain_label_to_asym_id(metadata: Metadata) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for chain in metadata.chains:
        if chain.label_asym_id is not None:
            mapping[chain.label_asym_id] = chain.asym_id
        mapping[str(chain.asym_id)] = chain.asym_id
    return mapping


def read_sabdab_pairs(
    path: pathlib.Path,
    metadata_by_id: dict[str, Metadata],
) -> dict[tuple[str, str, str], PairInfo]:
    pairs: dict[tuple[str, str, str], PairInfo] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "pdb_id",
            "asym_id_Hchain",
            "asym_id_Lchain",
            "sequence_Hchain",
            "sequence_Lchain",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required SAbDab columns: {sorted(missing)}")
        for row in reader:
            pdb_id = row["pdb_id"].strip().lower()
            h_label = row["asym_id_Hchain"].strip()
            l_label = row["asym_id_Lchain"].strip()
            metadata = metadata_by_id.get(pdb_id)
            if metadata is None:
                continue
            label_to_asym = chain_label_to_asym_id(metadata)
            if h_label not in label_to_asym or l_label not in label_to_asym:
                continue
            pairs[(pdb_id, h_label, l_label)] = PairInfo(
                pdb_id=pdb_id,
                h_label=h_label,
                l_label=l_label,
                h_asym_id=int(label_to_asym[h_label]),
                l_asym_id=int(label_to_asym[l_label]),
                h_sequence=row["sequence_Hchain"].strip(),
                l_sequence=row["sequence_Lchain"].strip(),
            )
    return pairs


def remove_structure_suffix(path: pathlib.Path) -> str:
    for suffix in (".pdb.zst", ".cif.zst", ".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def parse_group_dir_name(group_dir: pathlib.Path) -> tuple[str, str, str] | None:
    parts = group_dir.name.split("_")
    if len(parts) < 3:
        return None
    return parts[0].lower(), parts[1], "_".join(parts[2:])


def first_existing_name(
    files_by_name: dict[str, pathlib.Path], names: list[str]
) -> pathlib.Path | None:
    for name in names:
        path = files_by_name.get(name)
        if path is not None:
            return path
    return None


def confidence_path_for_structure(path: pathlib.Path) -> pathlib.Path | None:
    name = remove_structure_suffix(path)
    if name.endswith("_model"):
        name = name.removesuffix("_model")
    candidate = path.with_name(f"{name}_confidence.json")
    if candidate.exists():
        return candidate
    candidate = path.with_name(f"{name}_confidences.json")
    if candidate.exists():
        return candidate
    return None


def load_confidence(path: pathlib.Path | None) -> dict[str, float | None]:
    if path is None or not path.exists():
        return {"ptm": None, "avg_plddt": None}
    with path.open() as f:
        data = json.load(f)
    return {
        "ptm": data.get("ptm"),
        "avg_plddt": data.get("avg_plddt"),
    }


def collect_tasks(
    source_dir: pathlib.Path,
    pairs: dict[tuple[str, str, str], PairInfo],
    source: str,
) -> tuple[list[ApoMultimerTask], list[PriorMultimerTask], dict[str, int]]:
    apo_tasks: list[ApoMultimerTask] = []
    prior_tasks: list[PriorMultimerTask] = []
    stats = defaultdict(int)

    for group_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        group_key = parse_group_dir_name(group_dir)
        if group_key is None:
            stats["invalid_group_dir"] += 1
            continue
        pair = pairs.get(group_key)
        if pair is None:
            stats["missing_sabdab_pair"] += 1
            continue

        key = f"{pair.pdb_id}_{pair.h_label}_{pair.l_label}"
        files_by_name = {
            path.name: path for path in group_dir.iterdir() if path.is_file()
        }
        ranked_path = first_existing_name(
            files_by_name,
            [
                f"{key}_ranked_model.cif.zst",
                f"{key}_ranked_model.pdb.zst",
                f"{key}_ranked.cif.zst",
                f"{key}_ranked.pdb.zst",
                f"{key}_ranked_model.cif",
                f"{key}_ranked_model.pdb",
            ],
        )
        if ranked_path is None:
            stats["missing_ranked"] += 1
            continue
        apo_tasks.append(ApoMultimerTask(key=key, path=str(ranked_path), pair=pair))
        stats["apo_tasks"] += 1

        sample_paths = sorted(
            path
            for name, path in files_by_name.items()
            if name.startswith(f"{key}_seed-")
            and "_sample-" in name
            and name.endswith((".pdb.zst", ".cif.zst", ".pdb", ".cif"))
        )
        if not sample_paths:
            stats["missing_prior_samples"] += 1
            continue
        prior_tasks.append(
            PriorMultimerTask(
                key=key,
                paths=tuple(str(path) for path in sample_paths),
                sample_names=tuple(
                    f"{source}/{remove_structure_suffix(path)}" for path in sample_paths
                ),
                confidence_paths=tuple(
                    None
                    if (conf := confidence_path_for_structure(path)) is None
                    else str(conf)
                    for path in sample_paths
                ),
                pair=pair,
            )
        )
        stats["prior_tasks"] += 1
        stats["prior_samples"] += len(sample_paths)

    return apo_tasks, prior_tasks, dict(stats)


def match_pair_chains(raw_chains: dict[str, dict], pair: PairInfo) -> dict[int, dict]:
    seq_to_chain_ids: dict[str, list[str]] = defaultdict(list)
    for chain_id, chain in raw_chains.items():
        seq_to_chain_ids[chain["seq"]].append(chain_id)

    h_matches = seq_to_chain_ids.get(pair.h_sequence, [])
    l_matches = seq_to_chain_ids.get(pair.l_sequence, [])
    if (
        pair.h_sequence == pair.l_sequence
        and len(raw_chains) >= 2
        and len(h_matches) >= 2
        and len(l_matches) >= 2
    ):
        h_chain_id, l_chain_id = sorted(raw_chains)[:2]
        h_chain = raw_chains[h_chain_id]
        l_chain = raw_chains[l_chain_id]
        return {
            pair.h_asym_id: {
                "seq": h_chain["seq"],
                "coords": h_chain["coords"],
                "chain_type": "protein",
            },
            pair.l_asym_id: {
                "seq": l_chain["seq"],
                "coords": l_chain["coords"],
                "chain_type": "protein",
            },
        }

    if len(h_matches) != 1 or len(l_matches) != 1:
        raise ValueError(
            "Could not uniquely match H/L sequences: "
            f"H matches={h_matches}, L matches={l_matches}, "
            f"raw_chain_ids={sorted(raw_chains)}"
        )
    if h_matches[0] == l_matches[0]:
        raise ValueError(f"H/L sequences matched the same raw chain {h_matches[0]}")

    h_chain = raw_chains[h_matches[0]]
    l_chain = raw_chains[l_matches[0]]
    return {
        pair.h_asym_id: {
            "seq": h_chain["seq"],
            "coords": h_chain["coords"],
            "chain_type": "protein",
        },
        pair.l_asym_id: {
            "seq": l_chain["seq"],
            "coords": l_chain["coords"],
            "chain_type": "protein",
        },
    }


def apo_worker(task: ApoMultimerTask) -> tuple[str, bytes | None, str | None]:
    try:
        raw_chains = read_protein_multimer_structure(task.path)
        chains = match_pair_chains(raw_chains, task.pair)
        return task.key, pack_apo_multimer_record(chains), None
    except Exception as e:
        return task.key, None, f"{task.path}: {e}"


def prior_worker(task: PriorMultimerTask) -> tuple[str, bytes | None, str | None]:
    try:
        chain_samples: dict[int, list[np.ndarray]] = defaultdict(list)
        chain_sequences: dict[int, set[str]] = defaultdict(set)
        ptms: list[float] = []
        avg_plddts: list[float] = []

        for path_str, confidence_path in zip(
            task.paths, task.confidence_paths, strict=True
        ):
            raw_chains = read_protein_multimer_structure(path_str)
            chains = match_pair_chains(raw_chains, task.pair)
            for asym_id, chain in chains.items():
                chain_sequences[asym_id].add(chain["seq"])
                chain_samples[asym_id].append(chain["coords"])
            confidence = load_confidence(
                None if confidence_path is None else pathlib.Path(confidence_path)
            )
            ptms.append(np.nan if confidence["ptm"] is None else float(confidence["ptm"]))
            avg_plddts.append(
                np.nan
                if confidence["avg_plddt"] is None
                else float(confidence["avg_plddt"])
            )

        chains_out: dict[int, dict] = {}
        for asym_id in (task.pair.h_asym_id, task.pair.l_asym_id):
            if len(chain_sequences[asym_id]) != 1:
                raise ValueError(
                    f"Sequence mismatch across samples for asym_id {asym_id}: "
                    f"{[len(seq) for seq in chain_sequences[asym_id]]}"
                )
            shapes = {coords.shape for coords in chain_samples[asym_id]}
            if len(shapes) != 1:
                raise ValueError(
                    f"Coordinate shape mismatch across samples for asym_id "
                    f"{asym_id}: {shapes}"
                )
            chains_out[asym_id] = {
                "seq": next(iter(chain_sequences[asym_id])),
                "coords": np.stack(chain_samples[asym_id], axis=0),
                "chain_type": "protein",
            }

        value = pack_prior_multimer_stack_record(
            chains_out,
            list(task.sample_names),
            ptm=np.asarray(ptms, dtype=np.float32),
            avg_plddt=np.asarray(avg_plddts, dtype=np.float32),
        )
        return task.key, value, None
    except Exception as e:
        return task.key, None, f"{task.key}: {e}"


def ensure_overwrite(path: pathlib.Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_lmdb(
    path: pathlib.Path,
    tasks: list[ApoMultimerTask] | list[PriorMultimerTask],
    worker,
    num_workers: int,
    map_size_gb: int,
    overwrite: bool,
    desc: str,
) -> dict[str, int]:
    ensure_overwrite(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        map_size=map_size_gb * 1024 * 1024 * 1024,
        meminit=False,
        map_async=True,
        sync=False,
    )
    written = 0
    failed = 0
    errors: list[str] = []
    with mp.Pool(processes=num_workers) as pool:
        txn = env.begin(write=True)
        try:
            for key, value, error in tqdm(
                pool.imap_unordered(worker, tasks, chunksize=4),
                total=len(tasks),
                desc=desc,
            ):
                if value is None:
                    failed += 1
                    if error is not None and len(errors) < 20:
                        errors.append(error)
                    continue
                txn.put(key.encode("utf-8"), value)
                written += 1
                if written % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except Exception:
            txn.abort()
            raise
    env.sync()
    env.close()

    if errors:
        print(f"First {len(errors)} errors while writing {path}:")
        for error in errors:
            print(f"  {error}")
    return {"written": written, "failed": failed}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / f"rcsb-{args.split}"
    source_dir = args.source_dir or data_dir / "apo_multimer" / "protein" / args.source
    sabdab_csv = (
        args.sabdab_csv or data_dir / "sequences" / "sabdab_heavy_light_pairs.csv"
    )
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    metadata_by_id = load_manifest(data_dir / "manifest.msgpack")
    pairs = read_sabdab_pairs(sabdab_csv, metadata_by_id)
    apo_tasks, prior_tasks, stats = collect_tasks(source_dir, pairs, args.source)

    print("Collection summary:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  sabdab_pairs_loaded: {len(pairs)}")
    print(f"  apo_tasks: {len(apo_tasks)}")
    print(f"  prior_tasks: {len(prior_tasks)}")

    if args.dry_run:
        return

    apo_lmdb = data_dir / "apo_multimer_lmdb" / "protein" / f"{args.source}.lmdb"
    prior_lmdb = data_dir / "prior_multimer_lmdb" / "protein" / f"{args.source}.lmdb"
    apo_result = write_lmdb(
        apo_lmdb,
        apo_tasks,
        apo_worker,
        args.num_workers,
        args.map_size_gb,
        args.overwrite,
        desc=f"Writing apo multimer {args.source}",
    )
    prior_result = write_lmdb(
        prior_lmdb,
        prior_tasks,
        prior_worker,
        args.num_workers,
        args.map_size_gb,
        args.overwrite,
        desc=f"Writing prior multimer {args.source}",
    )
    print(f"apo_multimer_lmdb: {apo_result}")
    print(f"prior_multimer_lmdb: {prior_result}")


if __name__ == "__main__":
    main()
