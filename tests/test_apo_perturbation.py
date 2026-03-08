import random
from pathlib import Path

import numpy as np

from kfold.data.pipelines._protein_perturbation import (
    ProteinPerturbation,
    ProteinPerturbationConfig,
)
from kfold.data.utils.io.structure import read_protein_structure, write_protein_structure
from kfold.data.utils.simulation.bioprior import BioPriorConfig
from kfold.data.utils.simulation.rieprody import RieProdyConfig
from kfold.utils.geometry.rigid_align import compute_rmsd

ROOT_DIR = Path("/cache/wykim_lab/kfold_data/v260227/")

if __name__ == "__main__":
    data_dir = ROOT_DIR / "dataset" / "rcsb-train"
    save_dir = Path("./tmp/apo_perturbation")
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_path = data_dir / "rieprody_metric.lmdb"
    assert metric_path.exists(), f"RiePrody metric not found at {metric_path}"

    module = ProteinPerturbation(
        ProteinPerturbationConfig(
            rieprody=RieProdyConfig(
                metric_lmdb_path=metric_path,
                rmsd_threshold=100.0,  # disable rmsd filtering for testing
            ),
            bioprior=BioPriorConfig(noise_scale=1.0, max_steps=15),
        )
    )

    # Example usage
    source = "esmfold"
    files = sorted(list((data_dir / "apo" / source).rglob("*.pdb.gz")))
    random.seed(42)
    random.shuffle(files)

    for pdb_file in files[:20]:  # test on one file
        print(f"Processing {pdb_file}")
        name = pdb_file.name.split(".")[0]
        lmdb_key = f"{source}:{name}"

        seq, apo_coords = read_protein_structure(pdb_file)
        mask: np.ndarray = np.isfinite(apo_coords).all(axis=-1)

        rng = np.random.default_rng(42)

        print(f"Original structure for {name} has {len(seq)} residues.")

        # Save original structure
        output_file = save_dir / f"{name}.pdb"
        write_protein_structure(seq, apo_coords, output_file)

        # Rieprody perturbation
        for _ in range(10):  # run multiple times to check variability
            perturb_coords = module.rieprody_perturbation(
                apo_coords, mask, key=lmdb_key, rng=rng
            )
            if perturb_coords is not None:
                mask = np.isfinite(perturb_coords).all(axis=-1) & mask
                rmsd = compute_rmsd(
                    apo_coords[mask], perturb_coords[mask], mask=None, align=True
                )
                print(f"RiePrody perturbation RMSD for {name}: {rmsd:.2f} Å")
            else:
                print(f"RiePrody perturbation failed for {name}, skipping...")

        # Save the last RiePrody perturbation
        if perturb_coords is not None:
            output_file = save_dir / f"{name}_rieprody.pdb"
            write_protein_structure(seq, perturb_coords, output_file)
        else:
            print(f"RiePrody perturbation failed for {name}, skipping...")
        print()

        # BioPrior perturbation
        for _ in range(10):  # run multiple times to check variability
            perturb_coords = module.bioprior_perturbation(seq, apo_coords, rng=rng)
            if perturb_coords is not None:
                mask = np.isfinite(perturb_coords).all(axis=-1) & mask
                rmsd = compute_rmsd(
                    apo_coords[mask], perturb_coords[mask], mask=None, align=True
                )
                print(f"BioPrior perturbation RMSD for {name}: {rmsd:.2f} Å")

        # Save the last BioPrior perturbation
        if perturb_coords is not None:
            output_file = save_dir / f"{name}_bioprior.pdb"
            write_protein_structure(seq, perturb_coords, output_file)
        else:
            print(f"BioPrior perturbation failed for {name}, skipping...")
        print()
