"""Define training modules for k-fold"""

import random

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.utils.registry import Registry


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Initialize model here
        self.model: KFold = Registry.instantiate(self.config)

        # Setup training parameters
        training_config = config.training
        self.training_config = training_config
        self.validation_config = config.validation

        # Whether to train structure and confidence modules
        self.train_structure_module: bool = training_config.train_structure_module
        self.train_confidence_module: bool = training_config.train_confidence_module

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        pass

    def training_step(self, batch: FoldingInput, batch_idx: int) -> torch.Tensor:
        training_config = self.training_config

        # Sample recycling steps
        recycling_steps = random.randint(0, training_config.recycling_steps)

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=training_config.sampling_steps,
            diffusion_batch_size=training_config.diffusion_batch_size,
            num_samples=training_config.num_samples,
        )

        # Compute losses
        if training_config.train_structure_module:
            distogram_loss, distogram_metrics = self.distogram_loss(out, batch)
            try:
                diffusion_loss, diffusion_metrics = self.diffusion_loss(
                    out,
                    batch,
                    **self.diffusion_loss_args,
                )
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                return None

        else:
            distogram_loss, distogram_metrics = 0.0, {}
            diffusion_loss, diffusion_metrics = 0.0, {}

        if training_config.train_confidence_module:
            raise NotImplementedError("Confidence loss not implemented yet.")
        else:
            confidence_loss, confidence_metrics = 0.0, {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss = (
            training_config.confidence_loss_weight * confidence_loss
            + training_config.diffusion_loss_weight * diffusion_loss
            + training_config.distogram_loss_weight * distogram_loss
        )
        # Log losses
        self.log("train/loss", loss)

        all_metrics = distogram_metrics | diffusion_metrics | confidence_metrics
        for k in self.metrics.keys():
            v = all_metrics[k]
            self.metrics[k].update(v.detach() if torch.is_tensor(v) else v)
        self.training_log()

        return loss
