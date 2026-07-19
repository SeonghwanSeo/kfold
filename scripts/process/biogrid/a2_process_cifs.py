"""Convert selected BioGRID Protenix CIFs to validated RefStructure NPZs."""

import argparse
import csv
import logging
import multiprocessing
import os
import pathlib
from collections import Counter

import gemmi
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import PredictionRecord
from kfold.data.types.structure import RefStructure

DATASET_NAME = "Biogrid"
SUCCESS = "success"
FILTERED = "filtered"
FAILED = "failed"
_CCD_CACHE: CCD | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--metadata_path", type=pathlib.Path, default=None)
    parser.add_argument("--ccd_path", type=pathlib.Path, default=None)
    parser.add_argument("--model", default="Protenix-v1")
    parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_worker(ccd_path: pathlib.Path) -> None:
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_cif(task: dict) -> dict[str, str]:
    entry_id = task["entry_id"]
    out_path = pathlib.Path(task["out_path"])
    if out_path.exists() and not task["overwrite"]:
        return {"entry_id": entry_id, "status": SUCCESS, "error": "existing"}

    try:
        ccd = _CCD_CACHE
        assert ccd is not None, "CCD was not initialized in worker."
        raw_struct = gemmi.read_structure(task["cif_path"])
        cif_factory.clean_up_gemmi_structure(raw_struct)

        metadata = cif_factory.prepare_metadata_from_synthetic_data(
            entry_id, task["model"]
        )
        plddt = task.get("criterion_plddt_complex", "")
        metadata.pred = PredictionRecord(
            model=task["model"], plddt=float(plddt) if plddt else None
        )
        ref_struct: RefStructure = cif_factory.prepare_ref_structure(
            raw_struct, metadata, ccd, smiles_dict={}
        )
        if ref_struct.num_chains != 2 or not all(
            chain.ctype.is_protein for chain in ref_struct.chains
        ):
            return {
                "entry_id": entry_id,
                "status": FILTERED,
                "error": f"expected two protein chains, got {ref_struct.num_chains}",
            }

        cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)
        invalid_chains: set[int] = set()
        cif_factory.validate_chain_geometry(ref_struct, invalid_chains)
        cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)
        if invalid_chains:
            return {
                "entry_id": entry_id,
                "status": FILTERED,
                "error": f"invalid chains: {sorted(invalid_chains)}",
            }
        if len(ref_struct.metadata.interfaces) != 1:
            return {
                "entry_id": entry_id,
                "status": FILTERED,
                "error": (
                    f"expected one interface, got {len(ref_struct.metadata.interfaces)}"
                ),
            }

        expected = Counter(
            [task["protein_0"].strip().upper(), task["protein_1"].strip().upper()]
        )
        observed = Counter(
            chain.get_sequence(map_to_standard=True).upper()
            for chain in ref_struct.chains
        )
        if observed != expected:
            return {
                "entry_id": entry_id,
                "status": FILTERED,
                "error": (
                    f"sequence mismatch: expected lengths "
                    f"{sorted(map(len, expected.elements()))}, observed lengths "
                    f"{sorted(map(len, observed.elements()))}"
                ),
            }

        cluster_id = task["cluster_id"]
        for chain in ref_struct.metadata.chains:
            chain.cluster_id = cluster_id
        for interface in ref_struct.metadata.interfaces:
            interface.cluster_id = cluster_id

        ref_struct.validate()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ref_struct.save_npz(out_path)
        return {"entry_id": entry_id, "status": SUCCESS, "error": ""}
    except Exception as exc:
        return {"entry_id": entry_id, "status": FAILED, "error": repr(exc)}


def write_report(path: pathlib.Path, results: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entry_id", "status", "error"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["entry_id"]))


def main() -> None:
    args = parse_args()
    dataset_dir = args.data_dir / DATASET_NAME
    metadata_path = args.metadata_path or (dataset_dir / "metadata.csv")
    ccd_path = args.ccd_path or (args.data_dir.parent / "ccd-train.pkl")
    rows = load_rows(metadata_path)
    tasks = []
    for row in rows:
        task = dict(row)
        task.update(
            {
                "cif_path": str(dataset_dir / row["cif_path"]),
                "out_path": str(dataset_dir / "npz" / f"{row['entry_id']}.npz"),
                "model": args.model,
                "overwrite": args.overwrite,
            }
        )
        tasks.append(task)

    print(f"Prepared CIF parse tasks: {len(tasks)}")
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parse_cif, tasks, chunksize=4),
                total=len(tasks),
                desc="Processing BioGRID CIFs",
            )
        )

    write_report(dataset_dir / "processing_report.csv", results)
    counts = Counter(row["status"] for row in results)
    print(f"Processing results: {dict(sorted(counts.items()))}")
    failures = [row for row in results if row["status"] == FAILED]
    if failures:
        print("First processing failures:")
        for row in failures[:20]:
            print(f"  {row['entry_id']}: {row['error']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
