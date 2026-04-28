"""Preprocess synthetic data mmCIF files.

TODO: multi-ligand input is not yet supported.

Input:
<input_dir>/
    apo/
    holo/
    metadata.csv

WARN: (Seonghwan) I do not test this script on previous synthetic datasets
(I update the code after processing those datasets) so there may be some bugs.
I have to check it.

metadata.csv format:
I> python read.py
    data_idx  structure_idx     protein_0     protein_1  ligand_1   ...
0        0_0              0  STGSSGHDS...  SGPEESGPE...       NaN   ...
1        0_0              1  STGSSGHDS...  SGPEESGPE...       NaN   ...
2        0_0              2  STGSSGHDS...  SGPEESGPE...       NaN   ...
3        0_0              3  STGSSGHDS...  SGPEESGPE...       NaN   ...
4        0_0              4  STGSSGHDS...  SGPEESGPE...       NaN   ...
...      ...            ...           ...           ...       ...   ...


--use_dir_name=True:
holo/
    2/2_3/   (2: shard idx, 3: structure idx)
        structure_0.cif  (structure_idx)
        structure_1.cif
        ...

--use_dir_name=False:
holo/
    3/2_88/
        3_2_88.cif  ({data_idx}_{structure_idx}.cif)
        ...
"""

import argparse
import multiprocessing
import os
import pathlib

import gemmi
import pandas as pd
from tqdm import tqdm

from kfold.data.pipelines import cif_factory
from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure

# Error handling
SUCCESS = 0
FILTERED = 1
FAILED = 2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process synthetic mmCIF files.")
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `.cif` files directory.",
    )
    parser.add_argument(
        "--ccd_path",
        type=pathlib.Path,
        required=True,
        help="Path to CCD pickled file.",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name for synthetic data (e.g., 'synthetic_v1').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for predicted structures.",
    )
    parser.add_argument(
        "--use_dir_name",
        action="store_true",
        help=(
            "Whether to use the parent directory name as the entry ID "
            "(instead of the file name)"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


_CCD_CACHE: CCD | None = None


def init_worker(ccd_path):
    """Initialize worker process with global CCD data."""
    global _CCD_CACHE
    _CCD_CACHE = CCD.load(ccd_path)


def parse_cif(
    cif_path: pathlib.Path,
    out_path: pathlib.Path,
    entry_metadata: dict,
    model: str,
) -> int:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    if out_path.exists():
        return SUCCESS

    global _CCD_CACHE
    ccd = _CCD_CACHE
    assert ccd is not None, "CCD data not initialized in worker."

    smiles_dict: dict[str, str] = {}
    # HACK: this is hard-coded to our data-synthesis pipeline.
    if model == "Boltz-1":
        assert len([k for k in entry_metadata if k.startswith("ligand_")]) <= 1, (
            "Expected at most one ligand in entry metadata for non-Boltz models."
        )
        smiles_dict["LIG"] = entry_metadata.get("ligand_0", None)
    elif model == "Boltz-2":
        for i in range(3):
            ligand_key = f"ligand_{i}"
            if ligand_key in entry_metadata:
                smiles_dict[f"LIG{i + 1}"] = entry_metadata[ligand_key]
    else:
        raise NotImplementedError(f"Unsupported model: {model}")

    assert len(smiles_dict) <= 1, "Multi-ligand input is not supported yet."

    # Get metadata
    name = out_path.stem
    metadata = cif_factory.prepare_metadata_from_synthetic_data(name, model)

    # Prepare raw structure
    raw_struct: gemmi.Structure = gemmi.read_structure(str(cif_path))
    cif_factory.clean_up_gemmi_structure(raw_struct)

    # Prepare reference structure
    ref_struct: RefStructure = cif_factory.prepare_ref_structure(
        raw_struct, metadata, ccd, smiles_dict
    )
    # Insert coordinates
    cif_factory.insert_coordinates(ref_struct, raw_struct, metadata)

    # Identify invalid chains
    invalid_chains: set[int] = set()

    # Validate chain geometry
    cif_factory.validate_chain_geometry(ref_struct, invalid_chains)

    # Get interfaces and detect clashes
    cif_factory.detect_interfaces_and_detect_clashes(ref_struct, invalid_chains)

    if len(invalid_chains) > 0:
        return FILTERED

    # Save output if path is given
    ref_struct.save_npz(out_path)
    return SUCCESS


def worker_fn(task: dict):
    # only retrain the required fields for metadata
    entry_metadata = {
        k: v
        for k, v in task["entry_metadata"].items()
        if k.startswith("ligand_") and pd.notna(v)
    }
    try:
        return parse_cif(
            cif_path=task["cif_path"],
            out_path=task["out_path"],
            entry_metadata=entry_metadata,
            model=task["model"],
        )
    except Exception as e:
        print(f"Failed to process ({task['cif_path']}): {e}")
        # raise e
        return FAILED


def main():
    """Main function to process mmCIF files in parallel."""
    args = parse_args()

    input_dir = args.input_dir
    metadata_path = input_dir / "metadata.csv"
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata from {metadata_path}, total entries: {len(df)}")

    cif_dir: pathlib.Path = input_dir / "holo"
    print(f"Scanning for mmCIF files in {cif_dir}...")
    cif_paths = sorted(cif_dir.rglob("*.cif*")) + sorted(cif_dir.rglob("*.pdb*"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    # create data_idx/structure_idx -> cif path mapping
    # this may depend on `--use_dir_name` flag
    cif_mapping = {}
    for cif_path in cif_paths:
        if args.use_dir_name:
            data_idx = cif_path.parent.name
            structure_idx = int(cif_path.name.split(".")[0].split("_")[-1])
            entry_id = f"{data_idx}_{structure_idx}"
        else:
            filename = cif_path.name.split(".")[0]
            assert filename.count("_") >= 2, (
                f"Expected file name to contain at least two underscores, got {cif_path}"
            )
            entry_id = "_".join(filename.split("_")[:2])
        cif_mapping[entry_id] = cif_path

    print(f"Created CIF mapping for {len(cif_mapping)} entries.")

    data_dir: pathlib.Path = args.out_dir / args.name
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Match each row and CIF file
    tasks = []
    for _, row in df.iterrows():
        data_idx = row["data_idx"]
        structure_idx = row["structure_idx"]
        entry_id = f"{data_idx}_{structure_idx}"

        if entry_id not in cif_mapping:
            continue

        tasks.append(
            {
                "entry_id": entry_id,
                "cif_path": cif_mapping[entry_id],
                "out_path": out_dir / f"{entry_id}.npz",
                "entry_metadata": row.to_dict(),
                "model": args.model,
            }
        )
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(args.ccd_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_fn, tasks, chunksize=100),
                total=len(tasks),
                desc="Processing synthetic data",
            )
        )
    print("Processing completed.")
    n_success = results.count(SUCCESS)
    n_filtered = results.count(FILTERED)
    n_failed = results.count(FAILED)
    print(f"Successful: {n_success}, Filtered: {n_filtered}, Failed: {n_failed}")


if __name__ == "__main__":
    main()
