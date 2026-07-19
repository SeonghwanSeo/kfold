"""Convert filtered TPD CIF files to RefStructure NPZ files."""

import argparse
import logging
import multiprocessing
import os
import pathlib

import gemmi
import pandas as pd
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure

SUCCESS = 0
FILTERED = 1
FAILED = 2
DATASET_NAME = "tpd"

_CCD_CACHE: CCD | None = None


def get_nonpolymer_residue_names(raw_struct: gemmi.Structure) -> list[str]:
    residue_names: list[str] = []
    seen: set[str] = set()
    for entity in raw_struct.entities:
        if entity.entity_type not in {
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        }:
            continue
        if not entity.subchains:
            continue
        raw_chain = raw_struct[0].get_subchain(entity.subchains[0])
        for residue in raw_chain:
            if residue.name not in seen:
                residue_names.append(residue.name)
                seen.add(residue.name)
    return residue_names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The tpd/ folder is appended.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        default=None,
        help="CCD pickle path. Defaults to data_dir parent / ccd.pkl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Boltz-2",
        help="Prediction model name stored in metadata.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    return parser.parse_args()


def init_worker(ccd_path: pathlib.Path):
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def parse_cif(
    cif_path: pathlib.Path,
    out_path: pathlib.Path,
    entry_metadata: dict,
    model: str,
    overwrite: bool,
) -> int:
    if out_path.exists() and not overwrite:
        return SUCCESS

    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    raw_struct: gemmi.Structure = gemmi.read_structure(str(cif_path))
    cif_factory.clean_up_gemmi_structure(raw_struct)

    smiles_dict: dict[str, str] = {}
    ligand_values = [
        value
        for key, value in entry_metadata.items()
        if key.startswith("ligand_") and pd.notna(value)
    ]
    if len(ligand_values) > 1:
        raise ValueError(f"Multi-ligand TPD input is not supported: {cif_path}")
    if ligand_values:
        ligand_names = get_nonpolymer_residue_names(raw_struct)
        if len(ligand_names) != 1:
            raise ValueError(
                f"Expected one ligand residue name in TPD CIF, got {ligand_names}: "
                f"{cif_path}"
            )
        smiles_dict[ligand_names[0]] = str(ligand_values[0])

    metadata = cif_factory.prepare_metadata_from_synthetic_data(out_path.stem, model)

    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd, smiles_dict
    )
    if ref_struct.num_chains == 0:
        return FILTERED

    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    invalid_chains: set[int] = set()
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)
    if invalid_chains:
        return FILTERED

    ref_struct.validate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(task: dict):
    try:
        return parse_cif(
            task["cif_path"],
            task["out_path"],
            task["entry_metadata"],
            task["model"],
            task["overwrite"],
        )
    except Exception as exc:
        print(f"Failed to process {task['cif_path']}: {exc}")
        return FAILED


def main():
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    ccd_path = args.ccd_path or (args.data_dir / "ccd.pkl")
    metadata_path = data_dir / "metadata" / "filtered_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Prepare TPD cif/ and metadata first."
        )
    df = pd.read_csv(metadata_path)

    tasks = []
    for row in df.itertuples():
        cif_path = data_dir / row.cif_path
        out_path = data_dir / "npz" / row.subdb / f"{row.entry_id}.npz"
        if not cif_path.exists():
            continue
        tasks.append(
            {
                "cif_path": cif_path,
                "out_path": out_path,
                "entry_metadata": row._asdict(),
                "model": args.model,
                "overwrite": args.overwrite,
            }
        )
    print(f"Prepared CIF parse tasks: {len(tasks)}")

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_fn, tasks, chunksize=10),
                total=len(tasks),
                desc="Processing TPD CIFs",
            )
        )

    print(f"Successful: {results.count(SUCCESS)}")
    print(f"Filtered: {results.count(FILTERED)}")
    print(f"Failed: {results.count(FAILED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
