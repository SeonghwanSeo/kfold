from pathlib import Path

import torch

from kfold.config import load_config
from kfold.data.model_input import FoldingInput
from kfold.data.structure import TokenizedStructure
from kfold.model.models.boltz1 import Boltz1
from kfold.training.folding.dataset.datamodule import TrainingDataModule

TEST_CONFIG_PATH = Path("./configs/train-boltz1.yaml")
SAVE_FEATURE_PATH = Path("./tmp/features.pt")

if __name__ == "__main__":
    # Load Config
    global_config = load_config(TEST_CONFIG_PATH)

    # === Get data loader === #
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    dataloader = data_module.val_dataloader()

    # === Load Model === #
    model = Boltz1(global_config)
    model = model.eval()
    model = model.cuda()

    # === Get Embeddings and Save === #
    SAVE_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Turn off gradient
    torch.set_grad_enabled(False)

    f_input: FoldingInput
    for f_input, full_dict_list in dataloader:
        assert f_input.batch_size == 1

        # Move to GPU
        full_dict = full_dict_list[0]
        pdb_id: str = full_dict["id"]
        struct: TokenizedStructure = full_dict["structure"]

        print(
            f"Loaded sample {pdb_id}: {struct.num_chains} chains and "
            f"{struct.num_residues} residues."
        )

        f_input = f_input.to(device="cuda")
        print(f_input)

        s_inputs, s_init, z_init = model.input_embedder(f_input)

        s_trunk, z_trunk = model.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_cycles=2,
        )

        p_distogram = model.distogram_head(z_trunk)

        feature_save_path = SAVE_FEATURE_PATH.parent / f"{pdb_id}_features.pt"
        # Save embedding without padding
        n_tokens = struct.num_tokens
        features = {
            "s_inputs": s_inputs.cpu()[0, :n_tokens],
            "s_init": s_init.cpu()[0, :n_tokens],
            "z_init": z_init.cpu()[0, :n_tokens, :n_tokens],
            "s_trunk": s_trunk.cpu()[0, :n_tokens],
            "z_trunk": z_trunk.cpu()[0, :n_tokens, :n_tokens],
            "p_distogram": p_distogram.cpu()[0, :n_tokens, :n_tokens],
        }
        torch.save(features, feature_save_path)
        print(f"Saved features for {pdb_id} to {feature_save_path}")
