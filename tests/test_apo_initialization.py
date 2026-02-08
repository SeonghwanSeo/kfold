import pathlib

import numpy as np

from kfold.data.pipelines.apo_initialization import ApoInitializerConfig
from kfold.data.pipelines.prior_sampling import PriorSamplerConfig
from kfold.data.types.ccd import CCD
from kfold.data.utils.writer import KFoldWriter
from kfold.training.dataset.dataset import ValidationDataset, ValidationDatasetConfig
from kfold.utils.geometry.rigid_align import compute_rmsd

ROOT_DIR = pathlib.Path("/cache/wykim_lab/kfold_data/v260130/")

if __name__ == "__main__":
    CCD_PATH = ROOT_DIR / "ccd-train.pkl"
    ccd = CCD.load(CCD_PATH)

    num_prior_samples = 1
    DATASET_DIR = ROOT_DIR / "dataset" / "rcsb-val"
    dataset = ValidationDataset(
        config=ValidationDatasetConfig(
            name="rcsb-val",
            data_path=DATASET_DIR,
            apo_init=ApoInitializerConfig(),
            prior_sampler=PriorSamplerConfig(
                num_samples=num_prior_samples,
                use_ot_permutation=False,
                translation_scale=10.0,
            ),
            seed=42,
        ),
        ccd=ccd,
        pretrained_embedding={},
        safe_load=False,
    )
    writer = KFoldWriter()
    save_dir = pathlib.Path("./tmp/apo_init/")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Evaluate apo initialization RMSD (residue-wise assignment)
    # ============================================================

    # To compare the effect of OT-based residue permutation,
    # evaluate RMSD on the 100 smallest structures
    print("Starting apo initialization RMSD evaluation...")
    for i in range(100):
        metadata = dataset.metadatas[-i]
        struct = dataset.load_ref_structure(metadata)

        label_coords = np.concatenate(
            [chain.atom.coords for chain in struct.chains], axis=0
        )
        label_mask = np.isfinite(label_coords).all(-1)

        dataset.apo_initializer.use_residue_permutation = False
        struct1 = struct.clone()
        rng = np.random.default_rng(i)
        dataset.load_apo_structure(struct1, rng=rng)

        dataset.apo_initializer.use_residue_permutation = True
        struct2 = struct.clone()
        rng = np.random.default_rng(i)
        dataset.load_apo_structure(struct2, rng=rng)

        apo_coords1 = np.concatenate(
            [chain.atom.apo_coords for chain in struct1.chains], axis=0
        )
        apo_coords2 = np.concatenate(
            [chain.atom.apo_coords for chain in struct2.chains], axis=0
        )

        rng = np.random.default_rng(i)
        dataset.prior_sampler.use_ot_permutation = False
        tokenized1 = dataset.tokenize(struct1, rng)
        f_input = dataset.featurize(tokenized1, metadata, rng)
        prior_coords1 = np.ascontiguousarray(
            f_input.atom.prior_coords.numpy().transpose(1, 0, 2)
        )

        rng = np.random.default_rng(i)
        dataset.prior_sampler.use_ot_permutation = True
        tokenized2 = dataset.tokenize(struct1, rng)
        f_input = dataset.featurize(tokenized2, metadata, rng)
        prior_coords2 = np.ascontiguousarray(
            f_input.atom.prior_coords.numpy().transpose(1, 0, 2)
        )

        align_mask1 = label_mask & np.isfinite(apo_coords1).all(-1)
        align_mask2 = label_mask & np.isfinite(apo_coords2).all(-1)
        if not align_mask1.any() or not align_mask2.any():
            continue
        rmsd1 = compute_rmsd(apo_coords1, label_coords, align_mask1, align=True).item()
        rmsd2 = compute_rmsd(apo_coords2, label_coords, align_mask2, align=True).item()
        print(f"RMSD: {rmsd1:.3f} -> {rmsd2:.3f}, delta: {rmsd1 - rmsd2:.3f}")

        align_mask1 = label_mask & np.isfinite(prior_coords1).all(-1)
        align_mask2 = label_mask & np.isfinite(prior_coords2).all(-1)
        if not align_mask1.any() or not align_mask2.any():
            continue
        rmsd1 = compute_rmsd(prior_coords1, label_coords, align_mask1, align=True).item()
        rmsd2 = compute_rmsd(prior_coords2, label_coords, align_mask2, align=True).item()
        print(f"RMSD: {rmsd1:.3f} -> {rmsd2:.3f}, delta: {rmsd1 - rmsd2:.3f}")
        print()

        key = metadata.id
        writer.write_new_coords(struct, label_coords, save_dir / f"{key}_label.cif")
        writer.write_new_coords(struct, apo_coords2, save_dir / f"{key}_apo.cif")
        for i in range(num_prior_samples):
            writer.write_new_coords(
                struct,
                prior_coords2[i],
                save_dir / f"{key}_prior_{i}.cif",
            )
