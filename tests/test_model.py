from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from kfold.config import load_config
from kfold.data.featurize import featurize_structure
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.training.folding.loss.diffusion import (
    BondLoss,
    SmoothLDDTLoss,
    WeightedMSELoss,
)
from kfold.utils.boltz.process import tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure

TEST_CONFIG_PATH = Path("./configs/af3-mini.yaml")

DEVICE = torch.device("cuda")
PRECISION = torch.bfloat16
BOLTZ_PATH = Path(
    "/mnt/parallel_storage/wykim_lab/icl_mseok/BOLTZ1/rcsb_processed_targets/"
)
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


def test_model_train(model: KFold, folding_input: FoldingInput):
    # Forward/Backward pass
    torch.set_grad_enabled(True)
    forward_out = model.forward(
        f_input=folding_input,
        num_cycles=4,
        num_steps=20,
        num_diffusion_samples=1,
        diffusion_batch_size=32,
        sample_structures=False,
        train_structure_module=True,
        train_confidence_module=False,
    )

    diffusion_out = forward_out["diffusion"]
    t_hat = diffusion_out["t_hat"]
    x_pred = diffusion_out["denoised_atom_coords"]
    x_true = diffusion_out["true_atom_coords"]
    diffusion_loss_weights = diffusion_out["loss_weights"]

    # Calculate loss
    l_mse = mse_loss(
        x_pred=x_pred,
        x_true=x_true,
        f_input=folding_input,
    )
    l_bond = bond_loss(
        x_pred=x_pred,
        x_true=x_true,
        f_input=folding_input,
    )

    l_smooth_lddt = smooth_lddt_loss(
        x_pred=x_pred,
        x_true=x_true,
        f_input=folding_input,
        chunk_size=8,
    )
    print(t_hat)
    print(l_mse)
    print(l_bond)
    print(l_smooth_lddt)

    loss = (diffusion_loss_weights * (l_mse + l_bond) + l_smooth_lddt).mean()
    loss.backward()


def test_model_sample(model: KFold, folding_input: FoldingInput):
    # Model sampling
    torch.set_grad_enabled(False)
    sample_out, time_logs = model.sample(
        f_input=folding_input,
        num_cycles=4,
        num_steps=200,
        num_diffusion_samples=5,
    )
    print(time_logs)


if __name__ == "__main__":
    global_config = load_config(TEST_CONFIG_PATH)

    # print config
    print(OmegaConf.to_yaml(global_config))

    # instantiate model
    model = KFold(global_config)
    model = model.to(device=DEVICE, dtype=PRECISION)
    # print(model)

    # Print the number of parameters for each sub-module
    total_params = 0
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {num_params / 1e6:.2f}M parameters")
        total_params += num_params
    print(f"Total parameters: {total_params / 1e6:.2f}M parameters")

    # Test a forward pass with dummy data

    dummy_data_path = BOLTZ_STRUCTURE_DIR / "10gs.npz"
    boltz_structure = BoltzStructure.load(dummy_data_path)
    tokenized_structure = tokenize_structure(boltz_structure)
    # Crop first 200 tokens
    tokenized_structure = tokenized_structure.crop(np.arange(200))
    folding_input = featurize_structure(tokenized_structure)
    folding_input = folding_input.to(device=DEVICE)
    print(folding_input)

    # pad
    max_chains = 20
    max_tokens = 384
    max_atoms = max_tokens * 24

    mse_loss = WeightedMSELoss(align=True)
    bond_loss = BondLoss()
    smooth_lddt_loss = SmoothLDDTLoss()

    with torch.autocast(device_type=DEVICE.type, dtype=PRECISION):
        if False:
            # Train
            torch.set_grad_enabled(True)
            test_model_train(model, folding_input)

        if True:
            for max_tokens in [384, 512, 1024, 1536, 2048]:
                max_chains = 20
                max_atoms = max_tokens * 24
                folding_input_pad = FoldingInput(
                    chain=folding_input.chain.pad(max_chains),
                    atom=folding_input.atom.pad(max_atoms),
                    token=folding_input.token.pad(max_tokens),
                    bond=folding_input.bond.pad(max_tokens * 10),
                )
                print(f"Sampling with max tokens: {max_tokens}")
                for _ in range(3):
                    test_model_sample(model, folding_input_pad)
