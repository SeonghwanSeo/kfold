"""Define training modules for k-fold"""

import random
from typing import Any

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.utils.registry import Registry

from .loss import diffusion as diffusion_losses
from .loss import distogram as distogram_losses
from .optim.ema import ExponentialMovingAverage
from .utils import (
    gradient_norm,
    parameter_norm,
)


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config = config

        self.config = config.training
        self.training_config = config.training  # training_step
        self.validation_config = config.validation  # validation_step

        # Initialize model here
        self.model: KFold = Registry.instantiate(self.global_config)

        # EMA (See Section 5.6 Inference Setup)
        self.use_ema: bool = self.config.use_ema

        # Whether to train structure and confidence modules
        self.train_structure_module: bool = self.config.train_structure_module
        self.train_confidence_module: bool = self.config.train_confidence_module

        # Setup losses
        self.setup_losses()

    def freeze_model(self):
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when we train the confidence module only (Final-training-stage)
        if self.train_structure_module is False:
            self.model.trunk.eval()
            self.model.score_model.eval()
            self.model.trunk.requires_grad_(False)
            self.model.score_model.requires_grad_(False)

    def setup_losses(self):
        """Setup loss functions for training"""
        loss_config = self.config.loss
        self.loss_weights = loss_config.weights

        if self.train_structure_module:
            # Distogram loss
            self.distogram_loss = distogram_losses.DistogramLoss(
                **loss_config.distogram_loss
            )

            diffusion_loss_config = loss_config.diffusion_loss

            # Diffusion loss
            self.weighted_mse_loss = diffusion_losses.WeightedMSELoss(
                **diffusion_loss_config.mse_loss,
                align=True,
            )

            if self.loss_weights.bond > 0:
                # Only used in fine-tuning stage
                self.bond_loss = diffusion_losses.BondLoss()

            if self.loss_weights.smooth_lddt > 0:
                # Only used in regular training stage
                self.smooth_lddt_loss = diffusion_losses.SmoothLDDTLoss(
                    **diffusion_loss_config.smooth_lddt_loss
                )

        if self.train_confidence_module:
            raise NotImplementedError("Confidence loss not implemented yet.")

    def setup_metrics(self):
        """Setup metrics for training and validation"""
        pass

    def configure_optimizers(self):
        pass

    def forward(
        self,
        f_input: FoldingInput,
        num_cycles: int = 4,
        num_steps: int = 20,
        num_diffusion_samples: int = 1,
        diffusion_batch_size: int = 48,
        mode: str = "train",
    ) -> dict[str, dict[str, torch.Tensor]]:
        if mode == "train":
            return self.model(
                f_input,
                num_cycles=num_cycles,
                num_steps=num_steps,
                num_diffusion_samples=num_diffusion_samples,
                diffusion_batch_size=diffusion_batch_size,
                train_structure_module=self.train_structure_module,
                train_confidence_module=self.train_confidence_module,
                sample_structures=self.train_confidence_module,
            )
        elif mode == "validation":
            return self.model(
                f_input,
                num_cycles=num_cycles,
                num_steps=num_steps,
                num_diffusion_samples=num_diffusion_samples,
                diffusion_batch_size=diffusion_batch_size,
                train_structure_module=self.train_structure_module,
                train_confidence_module=self.train_confidence_module,
                sample_structures=self.train_confidence_module,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(self, batch: FoldingInput, batch_idx: int) -> torch.Tensor:
        training_config = self.training_config

        # Sample recycling steps
        num_cycles = random.randint(1, training_config.num_cycles)

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(
            f_input=batch,
            num_cycles=num_cycles,
            diffusion_batch_size=training_config.diffusion_batch_size,
            num_diffusion_steps=training_config.num_steps,
            num_diffusion_samples=training_config.num_diffusion_samples,
        )

        # Compute losses
        if self.train_structure_module:
            distogram_loss, distogram_metrics = self.compute_distogram_loss(
                logits=out["distogram"]["logits"],
                f_input=batch,
            )
            try:
                diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                    x_pred=out["diffusion"]["denoised_atom_coords"],
                    x_true=out["diffusion"]["true_atom_coords"],
                    weights=out["diffusion"]["loss_weights"],
                    f_input=batch,
                )
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                return None

        else:
            distogram_loss, distogram_metrics = 0.0, {}
            diffusion_loss, diffusion_metrics = 0.0, {}

        if self.train_confidence_module:
            confidence_loss, confidence_metrics = self.compute_confidence_loss()
        else:
            confidence_loss, confidence_metrics = 0.0, {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss_weights = self.loss_weights
        loss = (
            loss_weights.confidence * confidence_loss
            + loss_weights.diffusion * diffusion_loss
            + loss_weights.distogram * distogram_loss
        )

        # Log loss and metrics
        all_metrics = distogram_metrics | diffusion_metrics | confidence_metrics
        all_metrics["loss"] = loss.detach()

        all_metrics = {f"train/{k}": v for k, v in all_metrics.items()}
        self.log_dict(all_metrics)

        if self.global_step % 10 == 0:
            self.log_model_state()

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: sample molecules and compute validation metrics
        raise NotImplementedError()

    # === Loss functions === #
    def compute_distogram_loss(
        self, logits: torch.Tensor, f_input: FoldingInput
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute distogram loss.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape (B, Lt, Lt, num_bins) containing distogram logits.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        disto_loss : torch.Tensor
            The computed distogram loss.
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        loss = self.distogram_loss(logits, f_input).mean()
        metrics = {"distogram_loss": loss.detach()}
        return loss, metrics

    def compute_diffusion_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        weights: torch.Tensor,
        f_input: FoldingInput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute diffusion structure loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            The predicted atom coordinates of shape (B, Nsample, Latom, 3).
        x_true : torch.Tensor
            The ground truth atom coordinates of shape (B, Nsample, Latom, 3).
        weights : torch.Tensor
            The per-sample loss weights of shape (B, Nsample),
            which is computed from the diffusion noise scale.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        diffusion_loss : torch.Tensor
            The computed diffusion loss.
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}

        # Equation 3
        L_mse = self.weighted_mse_loss(x_pred, x_true, f_input)  # [B, Nsample]
        metrics["mse_loss"] = L_mse.mean().detach()

        # Equation 5
        alpha_bond = self.loss_weights.bond
        if alpha_bond > 0:
            L_bond = self.bond_loss(x_pred, x_true, f_input)  # [B, Nsample]
            metrics["bond_loss"] = L_bond.mean().detach()
        else:
            L_bond = 0.0

        # Algorithm 27; Section 5.2
        alpha_smooth_lddt = self.loss_weights.smooth_lddt
        if alpha_smooth_lddt > 0:
            L_smooth_lddt = self.smooth_lddt_loss(
                x_pred, x_true, f_input
            ).mean()  # scalar
            metrics["smooth_lddt_loss"] = L_smooth_lddt.detach()
        else:
            L_smooth_lddt = 0.0

        # See Section 3.7.1 Equation 6
        L_diffusion = (weights * (L_mse + alpha_bond * L_bond)).mean() + L_smooth_lddt
        metrics["diffusion_loss"] = L_diffusion.detach()

        return L_diffusion, metrics

    def compute_confidence_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError("Confidence loss not implemented yet.")

    # === Validation metric functions === #

    # === Training logs === #
    def log_model_state(self):
        """Log model parameter and gradient norms."""

        model = self.model
        self.log("train/grad_norm", gradient_norm(model), prog_bar=False)
        self.log("train/param_norm", parameter_norm(model), prog_bar=False)

        if self.train_structure_module:
            self.log(
                "train/grad_norm_trunk",
                gradient_norm(model.trunk),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_trunk",
                parameter_norm(model.trunk),
                prog_bar=False,
            )

            self.log(
                "train/grad_norm_score_model",
                gradient_norm(model.score_model),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_score_model",
                parameter_norm(model.score_model),
                prog_bar=False,
            )

        if self.train_confidence_module:
            self.log(
                "train/grad_norm_confidence_module",
                gradient_norm(model.confidence_module),
                prog_bar=False,
            )
            self.log(
                "train/param_norm_confidence_module",
                parameter_norm(model.confidence_module),
                prog_bar=False,
            )

        pass

    # === EMA === #
    # Started from https://github.com/jwohlwend/boltz

    @property
    def is_ema_initialized(self) -> bool:
        return hasattr(self, "_ema")

    @property
    def ema(self) -> ExponentialMovingAverage:
        assert hasattr(self, "_ema"), "EMA has not been initialized."
        return self._ema

    @ema.setter
    def ema(self, value: ExponentialMovingAverage):
        self._ema = value

    def on_train_start(self):
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.config.ema_decay
                self.ema = ExponentialMovingAverage(
                    parameters=self.parameters(), decay=ema_decay
                )
            self.ema.to(self.device)

    def on_train_epoch_start(self) -> None:
        if self.use_ema:
            self.ema.restore(self.parameters())

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        if self.use_ema:
            self.ema.update(self.parameters())

    def prepare_eval(self) -> None:
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.config.ema_decay
                self.ema = ExponentialMovingAverage(
                    parameters=self.parameters(), decay=ema_decay
                )
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def on_validation_start(self):
        self.prepare_eval()

    def on_predict_start(self) -> None:
        self.prepare_eval()

    def on_test_start(self) -> None:
        self.prepare_eval()
