from pathlib import Path

import lightning.pytorch as pl
import torch

from kfold.config import load_config
from kfold.data.featurize import featurize_structure
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.training.folding import metrics as validation_metrics
from kfold.utils.boltz.process import tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure

TEST_CONFIG_PATH = Path("./configs/train-af3-mini.yaml")
BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"
VALIDATION_KEY_PATH = Path("./assets/splits/boltz1/validation_ids.txt")


if __name__ == "__main__":
    pl.seed_everything(42)

    # Validation settings
    batch_size = 1
    num_samples = 5

    # Load keys
    with open(VALIDATION_KEY_PATH) as f:
        split = [v.strip().lower() for v in f]

    # instantiate model
    global_config = load_config(TEST_CONFIG_PATH)
    model = KFold(global_config)
    model = model.to("cuda")
    model = model.eval()

    # Turn off gradient
    torch.set_grad_enabled(False)

    for key in split:
        path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
        boltz_structure = BoltzStructure.load(path)
        try:
            tokenized = tokenize_structure(boltz_structure)
        except Exception as e:
            print(f"Error tokenizing structure {key}: {e}")
            continue

        # Featurize
        f_input = featurize_structure(tokenized)
        f_input = f_input.pad_to_multiple_of(32)

        # Collate
        f_input = FoldingInput.from_list([f_input])

        # Move to GPU
        f_input = f_input.to(device="cuda")

        if False:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward_out, _ = model.sample(
                    f_input=f_input,
                    num_cycles=4,
                    num_steps=20,
                    num_diffusion_samples=num_samples,
                )
            x_sample = forward_out["coordinates"]
        else:
            x_label = f_input.atom.label_coords[:, None, :, 0, :]
            x_label = x_label.expand(-1, num_samples, -1, -1)  # [B, Nsample, Natom, 3]

            noise_level = torch.rand(num_samples, device=x_label.device) * 5  # 0 - 5
            noise_level = torch.sort(noise_level).values
            print("noise level", noise_level)

            noise = torch.randn_like(x_label) * noise_level[None, :, None, None]
            x_sample = x_label + noise

        # Calculate metric
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x_label_aligned, atom_mask = validation_metrics.permute_label_coordinates(
                f_input, x_sample, {}, symmetry_correction=False
            )
            metrics = validation_metrics.compute_validation_metrics(
                f_input, x_label_aligned, x_sample, atom_mask
            )
        for k, v in metrics.items():
            print(k, v[0])
        breakpoint()
