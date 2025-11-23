from pathlib import Path

import lightning.pytorch as pl
import torch

from kfold.config import load_config
from kfold.training.folding import metrics as validation_metrics
from kfold.training.folding.dataset.datamodule import TrainingDataModule

TEST_CONFIG_PATH = Path("./configs/train-af3-mini.yaml")


if __name__ == "__main__":
    pl.seed_everything(42)

    # Validation settings
    num_samples = 5

    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.val_batch_size = 1
    global_config.train.data.num_workers = 0

    # Load validation loader
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("validate")
    dataloader = data_module.val_dataloader()

    # Turn off gradient
    torch.set_grad_enabled(False)

    for f_input, _ in dataloader:
        # Move to GPU
        f_input = f_input.to(device="cuda")

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
                f_input, x_sample, [], symmetry_correction=False
            )
            metrics = validation_metrics.compute_validation_metrics(
                f_input, x_label_aligned, x_sample, atom_mask
            )
        for k, v in metrics.items():
            print(k, v[0])
        breakpoint()
