import pathlib

import numpy as np

from kfold.data.pipelines.apo_initialization import ApoInitializerConfig
from kfold.data.types.ccd import CCD
from kfold.data.utils.writer.writer import KFoldWriter
from kfold.training.dataset.dataset import ValidationDataset, ValidationDatasetConfig

ROOT_DIR = pathlib.Path("/cache/wykim_lab/icl_shwan/v260107/")

if __name__ == "__main__":
    writer = KFoldWriter()

    CCD_PATH = ROOT_DIR / "ccd-train.pkl"
    ccd = CCD.load(CCD_PATH)

    DATASET_DIR = ROOT_DIR / "dataset" / "rcsb-val"
    dataset = ValidationDataset(
        config=ValidationDatasetConfig(
            name="rcsb-val",
            data_path=DATASET_DIR,
            apo_init=ApoInitializerConfig(
                use_perturbation=False,
                use_random_augmentation=True,
                use_ot_permutation=True,
            ),
            seed=42,
        ),
        ccd=ccd,
        pretrained_embedding={},
        featurization_args={},
        safe_load=False,
    )

    rng = np.random.default_rng(42)
    for i in range(301, 302):
        metadata = dataset.metadatas[i]
        print(metadata.id)
        struct = dataset.load_ref_structure(metadata)
        dataset.load_apo_structure(struct, rng=rng)
        print("write")
        writer.write_mmcif(struct, f"./tmp/{metadata.id}-apo.cif", save_apo=True)
        writer.write_mmcif(struct, f"./tmp/{metadata.id}.cif", save_apo=False)
