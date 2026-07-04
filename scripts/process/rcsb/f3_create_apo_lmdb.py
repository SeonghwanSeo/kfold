"""Create apo LMDB for RCSB PDB entries using Multiprocessing."""

import argparse
import multiprocessing as mp
import pathlib

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from kfold.data.utils.io.apo import chain_type_to_name, pack_apo_record
from kfold.data.utils.io.structure import read_dna_structure, read_protein_structure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        choices=["train", "val", "test"],
        help="Data split to process (train/val/test).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "source", "entity"],
        default="auto",
        help=(
            "Apo archive layout. 'source' is legacy apo/{source}/*.pdb; "
            "'entity' is apo/{shard}/{pdb_id}/{entity_id}/*.pdb."
        ),
    )
    parser.add_argument(
        "--lookup_path",
        type=pathlib.Path,
        default=None,
        help="Optional apo_lookup.msgpack path for entity layout.",
    )
    parser.add_argument(
        "--map_size_gb",
        type=int,
        default=128,
        help="LMDB map size in GB.",
    )
    args = parser.parse_args()
    return args


def infer_chain_type(source: str) -> str:
    if source in {"dna", "dna_helix"}:
        return "dna"
    if source in {"rna", "rna_apo"}:
        return "rna"
    return "protein"


def get_apo_key(apo_info: dict) -> str:
    if "key" in apo_info:
        return apo_info["key"]
    return f"{apo_info['source']}:{apo_info['name']}"


def normalize_lookup_entry(entry_lookup: dict) -> dict:
    if "chains" in entry_lookup or "complex_groups" in entry_lookup:
        return entry_lookup.get("chains", {})
    return entry_lookup


def resolve_entity_apo_file(
    data_dir: pathlib.Path,
    entry_id: str,
    entity_id: int,
    apo_info: dict,
) -> pathlib.Path | None:
    entity_dir = data_dir / "apo" / entry_id[1:3] / entry_id / str(entity_id)
    source = apo_info["source"]
    if source == "afdb":
        return entity_dir / "af2.pdb"
    if source == "esmfold":
        return entity_dir / "esmfold2.pdb"
    if source in {"protein", "protein_apo"}:
        return entity_dir / "protein_apo.pdb.zst"
    if source in {"dna", "dna_helix"}:
        return entity_dir / "helix.pdb.zst"
    if source in {"rna", "rna_apo"}:
        sample_id = apo_info.get("sample_id")
        if sample_id is None and "_sample_" in apo_info["name"]:
            sample_id = apo_info["name"].rsplit("_sample_", 1)[1]
        if sample_id is None:
            return None
        return entity_dir / f"rna_apo_sample_{sample_id}.cif.zst"
    return None


def collect_entity_layout_tasks(
    data_dir: pathlib.Path,
    lookup_path: pathlib.Path,
) -> list[tuple[pathlib.Path, str, str]]:
    with lookup_path.open("rb") as f:
        lookup = msgpack.unpack(f, raw=False, strict_map_key=False)

    tasks: list[tuple[pathlib.Path, str, str]] = []
    seen_apo_keys: set[str] = set()
    missing_files = 0
    unresolved = 0
    for entry_id, entry_lookup in lookup.items():
        chains = normalize_lookup_entry(entry_lookup)
        for entity_id_raw, apo_infos in chains.items():
            entity_id = int(entity_id_raw)
            for apo_info in apo_infos:
                apo_key = get_apo_key(apo_info)
                if apo_key in seen_apo_keys:
                    continue
                seen_apo_keys.add(apo_key)
                file_path = resolve_entity_apo_file(
                    data_dir, entry_id, entity_id, apo_info
                )
                if file_path is None:
                    unresolved += 1
                    continue
                if not file_path.exists():
                    missing_files += 1
                    continue
                chain_type = chain_type_to_name(
                    apo_info.get("chain_type", infer_chain_type(apo_info["source"]))
                )
                tasks.append((file_path, apo_key, chain_type))

    print(
        "Entity-layout collection: "
        f"tasks={len(tasks)}, missing_files={missing_files}, unresolved={unresolved}"
    )
    return tasks


def collect_source_layout_tasks(
    apo_dir: pathlib.Path,
) -> list[tuple[pathlib.Path, str, str]]:
    tasks = []
    seen_apo_keys: set[str] = set()
    for apo_subdir in sorted(apo_dir.iterdir()):
        if not apo_subdir.is_dir():
            continue
        apo_type = apo_subdir.name
        print(f"Collecting files for apo type: {apo_type}")
        suffixes = (
            "*.pdb.zst",
            "*.pdb",
            "*.pdb.gz",
            "*.cif.zst",
            "*.cif",
            "*.cif.gz",
            "*.npz",
        )
        for suffix in suffixes:
            files = sorted(apo_subdir.rglob(suffix))
            for f in files:
                apo_key = f"{apo_type}:{f.name.split('.')[0]}"
                if apo_key in seen_apo_keys:
                    continue
                seen_apo_keys.add(apo_key)
                tasks.append((f, apo_key, infer_chain_type(apo_type)))
    return tasks


def detect_layout(data_dir: pathlib.Path, lookup_path: pathlib.Path) -> str:
    if lookup_path.exists():
        return "entity"
    return "source"


def worker(task):
    """Worker function to parse a single PDB file."""
    file_path, key, chain_type = task
    try:
        if file_path.suffix == ".npz":
            with np.load(file_path) as data:
                seq = "".join(data["seq"].astype(str).tolist())
                coords = data["coords"].copy()
        elif chain_type == "dna":
            seq, coords = read_dna_structure(file_path)
        elif chain_type == "rna":
            from kfold.data.utils.io.structure import read_rna_structure

            seq, coords = read_rna_structure(file_path)
        else:
            seq, coords = read_protein_structure(file_path)
        value = pack_apo_record(seq, coords, chain_type)
        return key, value
    except Exception as e:
        # Return None to handle errors gracefully in the main loop
        print(f"Error processing {file_path}: {e}")
        return key, None


def main():
    """Main function using multiprocessing pool for heavy parsing tasks."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / f"rcsb-{args.split}"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    apo_dir = data_dir / "apo/"
    out_lmdb = data_dir / "apo.lmdb"
    lookup_path = args.lookup_path or data_dir / "apo_lookup.msgpack"

    # Pre-collect all tasks (file paths and their types)
    layout = args.layout
    if layout == "auto":
        layout = detect_layout(data_dir, lookup_path)
    print(f"Using apo archive layout: {layout}")
    if layout == "entity":
        tasks = collect_entity_layout_tasks(data_dir, lookup_path)
    else:
        tasks = collect_source_layout_tasks(apo_dir)

    print(f"Total files to process: {len(tasks)}")

    # Open LMDB Environment
    env = lmdb.open(
        str(out_lmdb),
        map_size=args.map_size_gb * 1024 * 1024 * 1024,
        meminit=False,
        map_async=True,
        sync=False,
    )

    # Use Multiprocessing Pool
    # chunksize controls how many tasks are sent to workers at once
    with mp.Pool(processes=args.num_workers) as pool:
        # Start transaction
        with env.begin(write=True) as txn:
            # imap_unordered yields results as soon as they are ready
            for k, v in tqdm(
                pool.imap_unordered(worker, tasks, chunksize=10), total=len(tasks)
            ):
                if v is not None:
                    txn.put(k.encode("utf-8"), v)

    env.close()
    print("LMDB creation finished successfully.")


if __name__ == "__main__":
    main()
