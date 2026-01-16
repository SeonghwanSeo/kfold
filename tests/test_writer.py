import pathlib

import numpy as np

from kfold.data.pipelines.apo_initialization import ApoInitializerConfig
from kfold.data.types.ccd import CCD
from kfold.data.utils.writer.writer import KFoldWriter
from kfold.training.dataset.dataset import ValidationDataset, ValidationDatasetConfig

ROOT_DIR = pathlib.Path("/cache/wykim_lab/kfold_data/v260109/")

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
                use_ot_permutation=False,
            ),
            seed=42,
        ),
        ccd=ccd,
        pretrained_embedding={},
        featurization_args={},
        safe_load=False,
    )

    save_dir = pathlib.Path("./tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(300, 310):
        rng = np.random.default_rng(42 + i)
        metadata = dataset.metadatas[i]
        struct_id = metadata.id
        struct = dataset.load_ref_structure(metadata)
        dataset.load_apo_structure(struct, rng=rng)
        print(f"Writing {struct_id}...")
        writer.write(struct, save_dir / f"{struct_id}-apo.cif", save_apo=True)
        writer.write(struct, save_dir / f"{struct_id}-apo.pdb", save_apo=True)
        writer.write(struct, save_dir / f"{struct_id}.pdb")
        writer.write(struct, save_dir / f"{struct_id}.cif")
