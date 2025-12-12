from pathlib import Path

import lightning.pytorch as pl
import torch
from tqdm import tqdm

from kfold.config import load_config
from kfold.training.folding.dataset.datamodule import TrainingDataModule

TEST_CONFIG_PATH = Path("./configs/train-af3-tiny.yaml")


if __name__ == "__main__":
    pl.seed_everything(42)

    # Validation settings
    num_samples = 5

    global_config = load_config(TEST_CONFIG_PATH)
    global_config.train.data.train_batch_size = 32
    global_config.train.data.safe_load = False
    global_config.train.data.num_workers = 64

    # Load training loader
    data_module = TrainingDataModule(global_config.train.data)
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()

    # Turn off gradient
    torch.set_grad_enabled(False)

    for f_input, _ in tqdm(dataloader):
        batch_size = f_input.batch_size
        num_tokens = f_input.num_tokens
        pass
