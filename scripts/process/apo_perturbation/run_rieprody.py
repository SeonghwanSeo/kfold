import argparse
import multiprocessing
from functools import partial
from pathlib import Path

from tqdm import tqdm

from kfold.data.utils.io.structure import read_protein_structure, write_protein_structure
from kfold.data.utils.simulation.rieprody import RiePrody, RieProdyConfig

_RIEPRODY: RiePrody | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RiePrody perturbation on apo protein structures."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/cache/wykim_lab/kfold_data/v260103/dataset/rcsb-train/"),
        help="Directory containing the RiePrody metric LMDB and apo structures.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save the perturbed structures.",
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="Whether to save the original apo structures alongside perturbed ones.",
    )
    parser.add_argument(
        "--num_perturbations",
        type=int,
        default=5,
        help="Number of perturbations to generate per structure.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def _initialize_rieprody(data_dir: Path):
    global _RIEPRODY
    module = RiePrody(
        config=RieProdyConfig(
            metric_lmdb_path=data_dir / "rieprody_metric.lmdb",
            rmsd_threshold=10.0,  # disable rmsd filtering for testing
            fallback_on_error=False,
            fallback_on_rmsd_exceed=False,
            disable_log=True,
        )
    )
    _RIEPRODY = module


def run_perturbation(
    file: Path,
    root_dir: Path,
    num_perturbations: int,
    save_original: bool = False,
):
    global _RIEPRODY
    assert _RIEPRODY is not None, "RiePrody module is not initialized."

    name = file.name.split(".")[0]

    try:
        seq, apo_coords = read_protein_structure(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return

    save_dir = root_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_original:
        output_file = save_dir / f"{name}.pdb"
        write_protein_structure(seq, apo_coords, output_file)

    sampled_coords_list = _RIEPRODY.sample(
        apo_coords, key=name, num_samples=num_perturbations
    )
    for i, sampled_coords in enumerate(sampled_coords_list):
        output_file = save_dir / f"{name}_perturb_{i}.pdb"
        write_protein_structure(seq, sampled_coords, output_file)


if __name__ == "__main__":
    args = parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    assert (data_dir / "rieprody_metric.lmdb").exists(), (
        f"RiePrody metric LMDB not found in {data_dir}."
    )
    assert (data_dir / "apo").exists(), (
        f"Apo structures directory not found in {data_dir}."
    )

    files = sorted(list((data_dir / "apo").rglob("*.pdb.gz")))

    print(f"Found {len(files)} apo structures to process.")

    func = partial(
        run_perturbation,
        root_dir=out_dir,
        num_perturbations=5,
        save_original=args.save_original,
    )

    with multiprocessing.Pool(
        128, initializer=_initialize_rieprody, initargs=(data_dir,)
    ) as pool:
        list(
            tqdm(
                pool.imap_unordered(func, files),
                total=len(files),
            )
        )
