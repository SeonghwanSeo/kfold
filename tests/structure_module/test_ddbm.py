from pathlib import Path

import lightning.pytorch as pl
import torch

import kfold.model.modules as submodules
from kfold.config import load_config
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.tokenized import TokenizedStructure
from kfold.model.modules.structure_module.kfold_ddbm import KFoldBridgeDiffusion
from kfold.training.dataset.datamodule import TrainingDataModule
from kfold.utils import errors
from kfold.utils.registry import Registry

TEST_CONFIG_PATH = Path("./configs/train-boltz1-ddbm.yaml")
SAVE_PATH = Path("./tmp/test_ddbm/")


if __name__ == "__main__":
    pl.seed_everything(2)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Validation settings
    num_samples = 20

    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0
    global_config.train.data.safe_load = False

    # Load validation loader
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    dataloader = data_module.val_dataloader()

    # instantiate ddbm module
    model_config = global_config.model
    score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
        model_config.score_model
    )
    structure_module: KFoldBridgeDiffusion = Registry.instantiate(
        model_config.structure_module, score_model=score_model
    )
    assert isinstance(structure_module, KFoldBridgeDiffusion)

    # Turn off gradient
    torch.set_grad_enabled(False)

    # For testing, turn off coordinate augmentation
    structure_module.coordinate_augmentation = False

    f_input: FoldingInput
    struct: TokenizedStructure
    for iter, (f_input, full_dict_list) in enumerate(dataloader):
        assert f_input.batch_size == 1

        if iter == 10:
            break

        full_dict = full_dict_list[0]
        name: str = full_dict["id"]
        struct = full_dict["structure"]

        print(f"Test {name} with {num_samples} samples")

        # ====== Save original structure ====== #
        try:
            struct.to_pdb(
                SAVE_PATH / f"{name}-original-gt.pdb",
            )
            struct.to_pdb(
                SAVE_PATH / f"{name}-original-apo.pdb",
                save_apo=True,
            )
        except errors.PDBWriterMaxChainError:
            print(f"Skipping {name} due to PDB writing error.")

        # ====== Save augmented structure ====== #
        # Replace coords
        struct = struct.replace_atom_coords(
            atom_coords=f_input.atom.label_coords[0].numpy().transpose(1, 0, 2),
        )
        struct = struct.replace_atom_coords(
            atom_coords=f_input.atom.apo_coords[0].numpy().transpose(1, 0, 2),
            is_apo=True,
        )
        struct.to_pdb(
            SAVE_PATH / f"{name}-feat-gt.pdb",
        )
        struct.to_pdb(
            SAVE_PATH / f"{name}-feat-apo.pdb",
            save_apo=True,
        )

        # ====== Save kabsch-aligned structures ====== #
        # Compared to augmented coords, apo structure must be
        # aligned to the holo structure.

        # Get label coords # [1, num_samples, Natom, 3]
        label_coords = structure_module.sample_holo(f_input, num_samples)
        # Get apo coords # [1, num_samples, Natom, 3]
        apo_coords = structure_module.sample_prior(f_input, num_samples, label_coords)

        # Replace coords and save
        struct = struct.replace_atom_coords(
            atom_coords=label_coords[0].numpy(),
        )
        struct = struct.replace_atom_coords(
            atom_coords=apo_coords[0].numpy(),
            is_apo=True,
        )
        struct.to_pdb(
            SAVE_PATH / f"{name}-ddbm-x0.pdb",
        )
        struct.to_pdb(
            SAVE_PATH / f"{name}-ddbm-xT.pdb",
            save_apo=True,
        )

        # ====== Save interpolated structures ====== #
        # Get sigmas
        t_hat = torch.linspace(
            structure_module.sigma_min, structure_module.sigma_max, num_samples
        )[None, :]  # [1, num_samples]
        noisy_coords = structure_module.interpolate(
            apo_coords,
            label_coords,
            t_hat,
            f_input.atom.resolved_mask,
        )

        struct = struct.replace_atom_coords(
            atom_coords=noisy_coords[0].numpy(),
        )
        for i in range(num_samples):
            # Save struct
            struct.to_pdb(
                SAVE_PATH / f"{name}-ddbm-t{i}.pdb",
                conformer_id=i,
            )
