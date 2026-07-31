"""Convert JASPAR protein/DNA CIF files to RefStructure NPZ files."""

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
from kfold.data.types.metadata import PredictionRecord
from kfold.data.types.structure import RefStructure

SUCCESS = 0
FILTERED = 1
FAILED = 2
DATASET_NAME = "JASPAR"

_CCD_CACHE: CCD | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Root preprocessed data directory. The JASPAR/ folder is appended.",
    )
    parser.add_argument(
        "--metadata_path",
        type=pathlib.Path,
        default=None,
        help="Metadata CSV path. Defaults to JASPAR/metadata.csv.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        default=None,
        help="CCD pickle path. Defaults to data_dir / ccd.pkl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Boltz-2",
        help="Prediction model name stored in metadata.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all metadata rows instead of distillation == 1 rows.",
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


def init_worker(ccd_path: pathlib.Path) -> None:
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def load_metadata(path: pathlib.Path, distillation_only: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if distillation_only:
        if "distillation" not in df.columns:
            raise KeyError(f"{path} does not contain a distillation column.")
        df = df[df["distillation"] == 1]
    return df


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

    metadata = cif_factory.prepare_metadata_from_synthetic_data(out_path.stem, model)
    plddt = entry_metadata.get("plddt")
    if pd.notna(plddt):
        metadata.pred = PredictionRecord(model=model, plddt=float(plddt))

    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd, smiles_dict={}
    )
    if ref_struct.num_chains == 0:
        return FILTERED

    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    invalid_chains: set[int] = set()
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)
    if invalid_chains:
        return FILTERED

    if not any(chain.ctype.is_protein for chain in ref_struct.chains):
        return FILTERED
    if not any(chain.ctype.is_dna for chain in ref_struct.chains):
        return FILTERED
    if any(chain.ctype.is_ligand for chain in ref_struct.chains):
        return FILTERED

    ref_struct.validate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(task: dict) -> int:
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


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / DATASET_NAME
    metadata_path = args.metadata_path or (data_dir / "metadata.csv")
    ccd_path = args.ccd_path or (args.data_dir / "ccd.pkl")
    df = load_metadata(metadata_path, distillation_only=not args.all)

    tasks = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        data_idx = str(row_dict["data_idx"])
        cif_path = data_dir / "cif" / f"{data_idx}.cif"
        out_path = data_dir / "npz" / f"{data_idx}.npz"
        if not cif_path.exists():
            continue
        tasks.append(
            {
                "cif_path": cif_path,
                "out_path": out_path,
                "entry_metadata": row_dict,
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
                desc="Processing JASPAR CIFs",
            )
        )

    print(f"Successful: {results.count(SUCCESS)}")
    print(f"Filtered: {results.count(FILTERED)}")
    print(f"Failed: {results.count(FAILED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
