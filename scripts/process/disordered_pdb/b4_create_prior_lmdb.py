"""Create disordered PDB prior LMDBs from matched RCSB train prior stacks."""

import argparse
import json
import pathlib
import shutil
from collections import defaultdict

import lmdb
import msgpack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory containing disordered_pdb and rcsb-train.",
    )
    parser.add_argument(
        "--rcsb_dir",
        type=pathlib.Path,
        default=None,
        help="Path to rcsb-train directory. Defaults to {data_dir}/rcsb-train.",
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
        help="Overwrite existing prior_lmdb outputs.",
    )
    return parser.parse_args()


def load_apo_lookup(path: pathlib.Path) -> dict[str, dict[str, list[dict]]]:
    with path.open("rb") as f:
        raw_lookup = msgpack.unpack(f, raw=False)
    out: dict[str, dict[str, list[dict]]] = {}
    for entry_id, entry_lookup in raw_lookup.items():
        out[str(entry_id)] = {
            str(entity_id): list(records) for entity_id, records in entry_lookup.items()
        }
    return out


def collect_prior_tasks(
    apo_lookup: dict[str, dict[str, list[dict]]],
) -> dict[str, dict[str, str]]:
    """Collect output prior keys and their matched RCSB source keys."""
    tasks: dict[str, dict[str, str]] = {}
    for entry_id, chains in apo_lookup.items():
        for entity_id, records in chains.items():
            for record in records:
                copied_from = record.get("copied_from")
                chain_type = record.get("chain_type")
                if copied_from is None or chain_type is None:
                    continue
                output_key = f"{entry_id}_{entity_id}"
                task_key = f"{chain_type}:{output_key}"
                existing = tasks.get(task_key)
                input_key = str(copied_from)
                if existing is not None and existing["input_key"] != input_key:
                    raise ValueError(
                        f"Conflicting RCSB prior sources for {task_key}: "
                        f"{existing['input_key']} vs {input_key}"
                    )
                tasks[task_key] = {
                    "chain_type": str(chain_type),
                    "input_key": input_key,
                    "output_key": output_key,
                }
    return tasks


def ensure_overwrite(path: pathlib.Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_prior_lmdbs(
    tasks: dict[str, dict[str, str]],
    *,
    rcsb_dir: pathlib.Path,
    disordered_root: pathlib.Path,
    map_size_gb: int,
    overwrite: bool,
) -> dict[str, int]:
    tasks_by_chain_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks.values():
        tasks_by_chain_type[task["chain_type"]].append(task)

    out_root = disordered_root / "prior_lmdb"
    ensure_overwrite(out_root, overwrite)
    out_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = defaultdict(int)
    for chain_type, chain_tasks in sorted(tasks_by_chain_type.items()):
        in_lmdb = rcsb_dir / "prior_lmdb" / f"{chain_type}.lmdb"
        if not in_lmdb.exists():
            stats[f"source_missing:{chain_type}"] += len(chain_tasks)
            continue

        out_lmdb = out_root / f"{chain_type}.lmdb"
        env_in = lmdb.open(str(in_lmdb), readonly=True, lock=False, readahead=False)
        env_out = lmdb.open(
            str(out_lmdb),
            map_size=map_size_gb * 1024 * 1024 * 1024,
            meminit=False,
            map_async=True,
            sync=False,
        )
        copied = 0
        missing = 0
        with env_in.begin(write=False) as txn_in:
            txn_out = env_out.begin(write=True)
            try:
                for task in sorted(chain_tasks, key=lambda item: item["output_key"]):
                    value = txn_in.get(task["input_key"].encode("utf-8"))
                    if value is None:
                        missing += 1
                        continue
                    txn_out.put(task["output_key"].encode("utf-8"), value)
                    copied += 1
                    if copied % 10000 == 0:
                        txn_out.commit()
                        txn_out = env_out.begin(write=True)
                txn_out.commit()
            except Exception:
                txn_out.abort()
                raise
        env_in.close()
        env_out.sync()
        env_out.close()
        stats[f"copied:{chain_type}"] = copied
        stats[f"missing:{chain_type}"] = missing
        print(f"{chain_type}: copied={copied}, missing={missing}, output={out_lmdb}")

    return dict(stats)


def main() -> None:
    args = parse_args()
    disordered_root = args.data_dir / "disordered_pdb"
    rcsb_dir = args.rcsb_dir or args.data_dir / "rcsb-train"
    apo_lookup_path = disordered_root / "apo_lookup.msgpack"
    if not apo_lookup_path.exists():
        raise FileNotFoundError(f"apo_lookup.msgpack not found: {apo_lookup_path}")

    apo_lookup = load_apo_lookup(apo_lookup_path)
    tasks = collect_prior_tasks(apo_lookup)
    print(f"Collected prior copy tasks: {len(tasks)}")
    stats = copy_prior_lmdbs(
        tasks,
        rcsb_dir=rcsb_dir,
        disordered_root=disordered_root,
        map_size_gb=args.map_size_gb,
        overwrite=args.overwrite,
    )

    report_path = disordered_root / "sequences" / "rcsb_prior_fetch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump({"stats": stats}, f, indent=2)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
