import json
import random
from pathlib import Path

import lightning.pytorch as pl
import torch
from tqdm import tqdm

from kfold.config import load_config
from kfold.data.featurize import featurize_structure
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.training.folding import loss as losses
from kfold.training.folding.dataset.cropper.boltz import BoltzCropper
from kfold.utils.boltz.process import parse_record, tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure

BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
TEST_CONFIG_PATH = Path("./configs/af3.yaml")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


if __name__ == "__main__":
    # Train settings
    batch_size = 2
    diffusion_batch_size = 32
    max_tokens = 512

    use_mse_loss = True
    use_smooth_lddt_loss = True
    use_distogram_loss = True

    pl.seed_everything(42)

    # Load keys
    global_config = load_config(TEST_CONFIG_PATH)

    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    manifest = {v["id"]: v for v in manifest}
    keys = sorted(list(manifest.keys()))
    random.shuffle(keys)
    keys = keys[:10000]

    # instantiate model
    model = KFold(global_config)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # data cropping
    cropper = BoltzCropper(BoltzCropper.Config())

    # loss functions
    mse_loss = losses.diffusion.WeightedMSELoss(align=True)
    smooth_lddt_loss = losses.diffusion.SmoothLDDTLoss()
    distogram_loss = losses.distogram.DistogramLoss(2.0, 22.0, 64).cuda()

    for it in tqdm(range(10000)):
        in_batch = []
        for key in keys[it * batch_size : (it + 1) * batch_size]:
            record = parse_record(manifest[key])
            if record.num_chains > 20:
                # Skip large structures for testing
                continue

            path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
            boltz_structure = BoltzStructure.load(path)
            try:
                tokenized = tokenize_structure(boltz_structure)
            except Exception as e:
                print(f"Error tokenizing structure {key}: {e}")
                continue

            # Crop structure
            tokenized = cropper.crop(tokenized, max_tokens, None)

            # Featurize
            f_input = featurize_structure(tokenized)
            f_input = f_input.pad_to_max_token(max_tokens)
            in_batch.append(f_input)

        # Collate
        f_input = FoldingInput.from_list(in_batch)

        # Move to GPU
        f_input = f_input.to(device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            optimizer.zero_grad()
            forward_out = model(
                f_input=f_input,
                num_cycles=4,
                num_steps=20,
                num_diffusion_samples=1,
                diffusion_batch_size=diffusion_batch_size,
                sample_structures=False,
                train_structure_module=True,
                train_confidence_module=False,
            )

        distogram_out = forward_out["distogram"]
        distogram_pred = distogram_out["logits"]

        diffusion_out = forward_out["diffusion"]
        t_hat = diffusion_out["t_hat"]
        x_pred = diffusion_out["denoised_atom_coords"]
        x_true = diffusion_out["true_atom_coords"]
        diffusion_loss_weights = diffusion_out["loss_weights"]

        # Calculate loss
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            if use_distogram_loss:
                l_distogram = distogram_loss(distogram_pred, f_input).mean()
            else:
                l_distogram = torch.tensor(0.0).to(x_pred.device)

            if use_mse_loss:
                l_mse = mse_loss(
                    x_pred=x_pred,
                    x_true=x_true,
                    f_input=f_input,
                    loss_weights=diffusion_loss_weights,
                )
            else:
                l_mse = torch.tensor(0.0).to(x_pred.device)

            if use_smooth_lddt_loss:
                l_smooth_lddt = smooth_lddt_loss(
                    x_pred=x_pred,
                    x_true=x_true,
                    f_input=f_input,
                    chunk_size=8,
                ).mean()
            else:
                l_smooth_lddt = torch.tensor(0.0).to(x_pred.device)

            l_diffusion = l_mse + l_smooth_lddt

            # AF3 loss weights
            w_diffusion = 4.0
            w_distogram = 3e-2
            loss = w_diffusion * l_diffusion + w_distogram * l_distogram

            print("Loss breakdown:")
            print("  Diffusion Loss:", l_diffusion.item())
            print("    MSE Loss:", l_mse.item())
            print("    Smooth LDDT Loss:", l_smooth_lddt.item())
            print("  Distogram Loss:", l_distogram.item())

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
