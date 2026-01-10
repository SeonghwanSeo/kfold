import pathlib

import numpy as np
from tqdm import tqdm

from kfold.data.pipelines._apo_prior import PolymerPriorConfig
from kfold.data.pipelines.apo_initialization import ApoInitializerConfig
from kfold.data.types.ccd import CCD
from kfold.training.dataset.dataset import ValidationDataset, ValidationDatasetConfig
from kfold.utils.geometry.rigid_align import compute_rmsd

ROOT_DIR = pathlib.Path("/cache/wykim_lab/kfold_data/v260109/")

if __name__ == "__main__":
    CCD_PATH = ROOT_DIR / "ccd-train.pkl"
    ccd = CCD.load(CCD_PATH)

    DATASET_DIR = ROOT_DIR / "dataset" / "rcsb-val"
    dataset = ValidationDataset(
        config=ValidationDatasetConfig(
            name="rcsb-val",
            data_path=DATASET_DIR,
            apo_init=ApoInitializerConfig(
                use_random_augmentation=True,
                use_ot_permutation=True,
                training=True,
                translation_scale=20.0,
                prior_sampler=PolymerPriorConfig(type="null"),
            ),
            seed=42,
        ),
        ccd=ccd,
        pretrained_embedding={},
        featurization_args={},
        safe_load=False,
    )

    print("Starting apo initialization RMSD evaluation...")
    rmsd_list: list[float] = []
    for i in tqdm(range(len(dataset)), desc="Evaluating apo initialization RMSD"):
        rng = np.random.default_rng(i)
        metadata = dataset.metadatas[i]
        struct = dataset.load_ref_structure(metadata)
        dataset.load_apo_structure(struct, rng=rng)

        apo_coords = np.concatenate(
            [chain.atom.apo_coords for chain in struct.chains], axis=0
        )
        label_coords = np.concatenate(
            [chain.atom.coords for chain in struct.chains], axis=0
        )
        assert apo_coords.shape == label_coords.shape, f"Shape mismatch for {metadata.id}"
        align_mask = np.isfinite(label_coords).all(axis=-1) & np.isfinite(apo_coords).all(
            axis=-1
        )
        if not align_mask.any():
            continue
        rmsd = compute_rmsd(apo_coords, label_coords, align_mask, align=True).item()
        rmsd_list.append(rmsd)

    rmsd_array = np.array(rmsd_list)
    print(f"Mean RMSD after apo initialization: {rmsd_array.mean():.8f} Å")
    print(f"Std RMSD after apo initialization: {rmsd_array.std():.8f} Å")
    print(f"Min RMSD after apo initialization: {rmsd_array.min():.8f} Å")
    print(f"Max RMSD after apo initialization: {rmsd_array.max():.8f} Å")
