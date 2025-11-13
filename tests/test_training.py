import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from kfold.config import load_config
from kfold.data.featurize import featurize_structure
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.training.folding.dataset.cropper.boltz import BoltzCropper
from kfold.training.folding.loss.diffusion import (
    SmoothLDDTLoss,
    WeightedMSELoss,
)
from kfold.training.folding.loss.distogram import (
    DistogramLoss,
)
from kfold.utils.boltz.process import parse_record, tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure

BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
TEST_CONFIG_PATH = Path("./configs/af3-mini.yaml")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


if __name__ == "__main__":
    global_config = load_config(TEST_CONFIG_PATH)

    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    manifest = {v["id"]: v for v in manifest}
    keys = sorted(list(manifest.keys()))
    random.seed(42)
    random.shuffle(keys)

    # data cropping
    cropper = BoltzCropper(BoltzCropper.Config())

    # loss functions
    mse_loss = WeightedMSELoss(align=True)
    distogram_loss = DistogramLoss(2.0, 22.0, 64).cuda()
    smooth_lddt_loss = SmoothLDDTLoss()

    # instantiate model
    model = KFold(global_config)
    model = model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    keys = keys[:10000]
    for it, key in enumerate(tqdm(keys)):
        record = parse_record(manifest[key])
        if record.num_chains > 20:
            # Skip large structures for testing
            continue

        # Set random seed for reproducibility
        random.seed(key)
        path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
        boltz_structure = BoltzStructure.load(path)
        tokenized = tokenize_structure(boltz_structure)

        # Crop structure
        tokenized = cropper.crop(tokenized, 384, None)
        # print(tokenized)

        # Featurize
        f_input = featurize_structure(tokenized)
        f_input = f_input.pad_to_max_token(384)
        f_input = f_input.to(device="cuda")
        f_input = FoldingInput.from_list([f_input])
        # print(f_input)

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            optimizer.zero_grad()
            forward_out = model.forward(
                f_input=f_input,
                num_cycles=1,
                num_steps=20,
                num_diffusion_samples=1,
                diffusion_batch_size=16,
                sample_structures=False,
                train_structure_module=True,
                train_confidence_module=False,
            )

            distogram_out = forward_out["distogram"]
            distogram_pred = distogram_out["logits"]
            l_distogram = distogram_loss.forward(distogram_pred, f_input)

            diffusion_out = forward_out["diffusion"]
            t_hat = diffusion_out["t_hat"]
            x_pred = diffusion_out["denoised_atom_coords"]
            x_true = diffusion_out["true_atom_coords"]
            diffusion_loss_weights = diffusion_out["loss_weights"]

            # Calculate loss
            l_mse = mse_loss(
                x_pred=x_pred,
                x_true=x_true,
                f_input=f_input,
            )

            l_smooth_lddt = smooth_lddt_loss(
                x_pred=x_pred,
                x_true=x_true,
                f_input=f_input,
                chunk_size=8,
            )
            loss = (diffusion_loss_weights * l_mse + l_smooth_lddt + l_distogram).mean()
            loss.backward()

            print("Iteration:", it, "Loss:", loss.item())

            # Check for unused parameters
            unused_params = []
            for name, p in model.named_parameters():
                if p.grad is None:
                    if p.requires_grad:
                        unused_params.append(name)
            if len(unused_params) > 0:
                print("Warning: Unused parameters detected:")
                for name in unused_params:
                    print(f" - {name}")

            optimizer.step()
