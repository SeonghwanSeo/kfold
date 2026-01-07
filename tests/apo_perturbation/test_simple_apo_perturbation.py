import random
from pathlib import Path

from kfold.data.pipelines._apo_perturbation import (
    ApoPerturbation,
    ApoPerturbationConfig,
    LangevinConfig,
)
from kfold.data.utils.io.structure import read_protein_structure, write_protein_structure
from kfold.data.utils.simulation.rieprody import RieProdyConfig

if __name__ == "__main__":
    data_dir = Path("/cache/wykim_lab/kfold_data/v260103/dataset/rcsb-train/")
    save_dir = Path("./tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_path = data_dir / "rieprody_metric.lmdb"
    assert metric_path.exists(), f"RiePrody metric not found at {metric_path}"

    # create rieprody config
    rieprody_config = RieProdyConfig(
        metric_lmdb_path=metric_path,
        rmsd_threshold=100.0,  # disable rmsd filtering for testing
        fallback_on_error=True,
        fallback_on_rmsd_exceed=True,
        disable_log=False,
    )
    langevin_config = LangevinConfig(
        min_steps=1,
        max_steps=5,
        dt=0.09,  # N(0, 0.04) noise
        res_r=4.0,
        bond_r=4.0,
        ent_r=10.0,
        sphere_r=10.0,
    )

    module = ApoPerturbation(
        ApoPerturbationConfig(
            rieprody=rieprody_config,
            langevin=langevin_config,
        )
    )

    # Example usage
    files = sorted(list((data_dir / "apo").rglob("*.pdb.gz")))
    random.seed(42)
    random.shuffle(files)

    for pdb_file in files[:10]:  # test on one file
        print(f"Processing {pdb_file}")
        name = pdb_file.name.split(".")[0]
        seq, apo_coords = read_protein_structure(pdb_file)

        # Save original structure
        output_file = save_dir / f"{name}.pdb"
        write_protein_structure(seq, apo_coords, output_file)

        # # Rieprody perturbation
        module.prob_rieprody = 1.0  # always use rieprody for testing
        perturb_coords = module.run(apo_coords, key=name)
        output_file = save_dir / f"{name}_rieprody.pdb"
        write_protein_structure(seq, perturb_coords, output_file)

        # Langevin perturbation
        module.prob_rieprody = 0.0  # always use langevin for testing
        perturb_coords = module.run(apo_coords)
        output_file = save_dir / f"{name}_langevin.pdb"
        write_protein_structure(seq, perturb_coords, output_file)
