import numpy as np

from kfold.data.types.ccd import CCD
from kfold.data.utils.writer.writer import KFoldWriter
from kfold.training.dataset.dataset import ValidationDataset, ValidationDatasetConfig

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    writer = KFoldWriter()

    ccd = CCD.load("/cache/wykim_lab/icl_shwan/debug-parse-1/ccd-train.pkl")
    dataset = ValidationDataset(
        ValidationDatasetConfig(
            name="rcsb-val",
            data_path="/cache/wykim_lab/icl_shwan/debug-parse-1/rcsb-val/",
        ),
        ccd=ccd,
        pretrained_embedding={
            "seq": None,
            "seq_dim": 0,
            "struct": None,
            "struct_dim": 0,
            "max_struct_ensembles": 0,
        },
        featurization_args={},
    )
    for i in range(301, 302):
        metadata = dataset.metadatas[i]
        print(metadata.id)
        struct = dataset.load_ref_structure(metadata)
        dataset.load_apo_structure(struct, rng=rng)
        print("write")
        writer.write_mmcif(struct, f"./tmp/{metadata.id}-apo.cif", save_apo=True)
        writer.write_mmcif(struct, f"./tmp/{metadata.id}.cif", save_apo=False)
