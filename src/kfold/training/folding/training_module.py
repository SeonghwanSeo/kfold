"""Define training modules for k-fold"""

import gc
import random
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MeanMetric

from kfold import constants as C
from kfold.config import to_dict
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.utils.registry import MAIN_MODULE

from . import loss as loss_fn
from . import metrics as validation_metrics
from .optim.ema import ExponentialMovingAverage
from .optim.lr_scheduler import AF3LRScheduler
from .utils import gradient_norm, parameter_norm


@dataclass(kw_only=True)
class TrainConfig:
    """Configuration for training and validation steps."""

    training: "TrainingConfig"
    validation: "ValidationConfig"
    optimizer: "OptimizerConfig"
    loss: "LossConfig"


@dataclass(kw_only=True)
class OptimizerConfig:
    """Optimizer configuration.
    See Section 5.4 of the AlphaFold3 paper.
    """

    # optimizer
    opt: str = "adam"
    beta_1: float = 0.9
    beta_2: float = 0.95
    eps: float = 1e-8
    # lr scheduler
    lr_scheduler: str = "af3"  # or "none"
    base_lr: float = 0
    max_lr: float = 0.0018
    lr_warmup_no_steps: int = 1000
    lr_start_decay_after_n_steps: int = 50000
    lr_decay_every_n_steps: int = 50000
    lr_decay_factor: float = 0.95
    # ema
    use_ema: bool = True
    ema_decay: float = 0.999


@dataclass(kw_only=True)
class TrainingConfig:
    """Training step configuration."""

    # Whether to train each submodules
    train_trunk: bool = True
    train_distogram_head: bool = True
    train_structure_module: bool = True
    train_confidence_head: bool = False

    # trunk recycling
    num_cycles: int = 4
    # for structure model training
    diffusion_batch_size: int = 48
    # for confidence module training
    num_steps: int = 20
    num_diffusion_samples: int = 1


@dataclass(kw_only=True)
class ValidationConfig:
    """Validation step configuration."""

    num_cycles: int = 4
    num_steps: int = 20
    num_diffusion_samples: int = 5
    symmetry_correction: bool = False


@dataclass(kw_only=True)
class LossConfig:
    """Training step configuration."""

    # TODO: better configuration

    weights: dict[str, float]
    distogram_loss: Any
    diffusion_loss: Any
    confidence_loss: Any


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config = config
        self.config: TrainConfig = self.global_config.train
        self.training_config: TrainingConfig = self.config.training
        self.validation_config: ValidationConfig = self.config.validation
        self.optimizer_config: OptimizerConfig = self.config.optimizer
        self.loss_config: LossConfig = self.config.loss

        # Whether to train structure and confidence modules
        self.train_trunk: bool = self.training_config.train_trunk
        self.train_distogram_head: bool = self.training_config.train_distogram_head
        self.train_structure_module: bool = self.training_config.train_structure_module
        self.train_confidence_head: bool = self.training_config.train_confidence_head

        # Initialize model here
        self.model: KFold
        model_cls = MAIN_MODULE[self.global_config.model._class_]
        self.model = model_cls(self.global_config)

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup losses
        self.setup_losses()

        self.save_hyperparameters(to_dict(self.global_config))

        self.validation_metrics: dict[str, MeanMetric] = nn.ModuleDict()  # type: ignore
        self.validation_metrics["rmsd"] = MeanMetric()
        self.validation_metrics["best_rmsd"] = MeanMetric()
        for m in C.training.LDDTType:
            self.validation_metrics[f"lddt_{m.value}"] = MeanMetric()
            self.validation_metrics[f"complex_lddt_{m.value}"] = MeanMetric()

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when we train the confidence module only (Final-training-stage)

        self.frozen_modules = []
        if self.train_trunk is False:
            self.frozen_modules += ["input_embedder", "trunk"]

        if self.train_distogram_head is False:
            self.frozen_modules += ["distogram_head"]

        if self.train_structure_module is False:
            self.frozen_modules += ["score_model"]

        if self.train_confidence_head is False:
            # TODO: freeze confidence module after they are implemented
            pass

        for module_name in self.frozen_modules:
            module = getattr(self.model, module_name)
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        """Override train() to set sub-modules to eval mode if frozen."""
        out = super().train(mode)
        for module_name in self.frozen_modules:
            module = getattr(self.model, module_name)
            module.eval()
        return out

    def setup_losses(self):
        """Setup loss functions for training"""
        loss_config = self.loss_config
        self.loss_weights = loss_config.weights

        if self.train_structure_module:
            # Distogram loss
            if self.loss_weights["distogram"] > 0:
                self.distogram_loss = loss_fn.distogram.DistogramLoss(
                    **loss_config.distogram_loss
                )

            # Diffusion loss
            diffusion_loss_config = loss_config.diffusion_loss
            self.weighted_mse_loss = loss_fn.diffusion.WeightedMSELoss(
                **diffusion_loss_config.mse_loss
            )
            if self.loss_weights["bond"] > 0:
                # Only used in fine-tuning stage
                self.bond_loss = loss_fn.diffusion.BondLoss()
            if self.loss_weights["smooth_lddt"] > 0:
                # Only used in regular training stage
                self.smooth_lddt_loss = loss_fn.diffusion.SmoothLDDTLoss(
                    **diffusion_loss_config.smooth_lddt_loss
                )

        if self.train_confidence_head:
            raise NotImplementedError("Confidence loss not implemented yet.")

    def setup_metrics(self):
        """Setup metrics for validation"""
        pass

    def configure_optimizers(self):  # type: ignore
        config = self.optimizer_config
        parameters = [p for p in self.parameters() if p.requires_grad]

        if config.opt.lower() == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                betas=(config.beta_1, config.beta_2),
                eps=config.eps,
                lr=config.base_lr,
            )
        else:
            raise NotImplementedError(f"Optimizer {config.opt} not implemented yet.")
        if config.lr_scheduler == "af3":
            scheduler = AF3LRScheduler(
                optimizer,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                warmup_no_steps=config.lr_warmup_no_steps,
                start_decay_after_n_steps=config.lr_start_decay_after_n_steps,
                decay_every_n_steps=config.lr_decay_every_n_steps,
                decay_factor=config.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

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
                train_confidence_module=self.train_confidence_head,
                sample_structures=self.train_confidence_head,
            )
        elif mode == "validation":
            dict_out, _ = self.model.sample(
                f_input,
                num_cycles=num_cycles,
                num_steps=num_steps,
                num_diffusion_samples=num_diffusion_samples,
            )
            return {"sample": dict_out}
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        training_config = self.training_config

        f_input, _ = batch  # second one is full_structure_dict, not used in training step

        # Sample recycling steps
        num_cycles = random.randint(1, training_config.num_cycles)

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(
            f_input=f_input,
            num_cycles=num_cycles,
            num_steps=training_config.num_steps,
            num_diffusion_samples=training_config.num_diffusion_samples,
            diffusion_batch_size=training_config.diffusion_batch_size,
            mode="train",
        )
        try:
            loss, metrics = self.compute_losses(batch, out)
        except Exception as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return None

        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"))

        self.log_model_state()

        return loss

    def compute_losses(
        self, batch: FoldingInput, model_output: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses of given the model output."""
        with torch.autocast("cuda", dtype=torch.float32):
            # NOTE: Compute the losses in float32 for better numerical stability
            # Compute losses
            if self.train_structure_module:
                distogram_loss, distogram_metrics = self.compute_distogram_loss(
                    logits=model_output["distogram"]["logits"],
                    f_input=batch,
                )
                diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                    x_pred=model_output["diffusion"]["denoised_atom_coords"],
                    x_true=model_output["diffusion"]["true_atom_coords"],
                    per_sample_weights=model_output["diffusion"]["loss_weights"],
                    f_input=batch,
                )

            else:
                distogram_loss, distogram_metrics = 0.0, {}
                diffusion_loss, diffusion_metrics = 0.0, {}

            if self.train_confidence_head:
                confidence_loss, confidence_metrics = self.compute_confidence_loss()
            else:
                confidence_loss, confidence_metrics = 0.0, {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss_weights = self.loss_weights
        loss = (
            loss_weights["confidence"] * confidence_loss
            + loss_weights["diffusion"] * diffusion_loss
            + loss_weights["distogram"] * distogram_loss
        )  # [B,]
        assert torch.is_tensor(loss), "Loss must be a torch.Tensor."

        # Mean over batch
        loss = loss.mean()

        # Log loss and metrics
        all_metrics = distogram_metrics | diffusion_metrics | confidence_metrics
        all_metrics["loss"] = loss.detach()

        return loss, all_metrics

    def validation_step(self, batch, batch_idx):
        # TODO: sample molecules and compute validation metrics
        val_config = self.validation_config
        num_diffusion_samples = val_config.num_diffusion_samples
        f_input, full_structure_dict = batch

        assert f_input.batch_size == 1, "Validation batch size should be 1"

        try:
            out = self(
                f_input=f_input,
                num_cycles=val_config.num_cycles,
                num_steps=val_config.num_steps,
                num_diffusion_samples=num_diffusion_samples,
                mode="validation",
            )
            sample_coords = out["sample"]
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        try:
            # symmetry correction
            # TODO: get_true_coordinates function to use symmetry correction
            true_coords, atom_mask = validation_metrics.permute_label_coordinates(
                f_input=batch,
                pred_coords=sample_coords,
                full_structure_dict=batch.full_structure_dict,
                symmetry_correction=val_config.symmetry_correction,
            )
            metrics = validation_metrics.compute_validation_metrics(
                f_input=batch,
                true_coords=true_coords,
                pred_coords=sample_coords,
                atom_mask=atom_mask,
            )
            for k, (v, w) in metrics.items():
                self.validation_metrics[k].update(v, w)

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

    def on_validation_epoch_end(self):
        """Aggregate and log validation metrics at the end of the epoch."""
        # Aggregate validation metrics
        avg_metrics: dict[str, torch.Tensor] = {
            k: self.validation_metrics[k].compute() for k in self.validation_metrics
        }
        for k in self.validation_metrics:
            self.validation_metrics[k].reset()

        # Compute weighted lddt scores (Monitored metrics)
        # NOTE: this is equivalent to Boltz1's `lddt` metric.
        lddt_weights = C.training.LDDTWeightsBoltz
        sum_weights = sum(lddt_weights.values())

        weighted_lddt = 0
        for m, w in lddt_weights.items():
            weighted_lddt += avg_metrics[m.value] * w
        weighted_lddt /= sum_weights
        avg_metrics["weighted_lddt"] = weighted_lddt  # type: ignore

        self.log_dict(avg_metrics)

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
            The computed distogram loss of shape (B,).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        loss = self.distogram_loss(logits, f_input)
        metrics = {"distogram_loss": loss.detach().mean()}
        return loss, metrics

    def compute_diffusion_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        per_sample_weights: torch.Tensor,
        f_input: FoldingInput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute diffusion structure loss.
        See Section 3.7.1 Diffusion Training.

        Parameters
        ----------
        x_pred : torch.Tensor
            The predicted atom coordinates of shape (B, Nsample, Latom, 3).
        x_true : torch.Tensor
            The ground truth atom coordinates of shape (B, Nsample, Latom, 3).
        per_sample_weights : torch.Tensor
            The per-sample loss weights of shape (B, Nsample),
            which is computed from the diffusion noise scale.
        f_input : FoldingInput
            The input features containing the target distogram and masks.

        Returns
        -------
        diffusion_loss : torch.Tensor
            The computed diffusion loss of shape (B,).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}

        # Equations 3-4
        L_mse = self.weighted_mse_loss(x_pred, x_true, f_input)  # [B, Nsample]
        L_mse = L_mse * per_sample_weights  # [B, Nsample]
        metrics["mse_loss"] = L_mse.detach().mean()

        # Equation 5
        alpha_bond = self.loss_weights["bond"]
        if alpha_bond > 0:
            L_bond = self.bond_loss(x_pred, x_true, f_input)
            L_bond = L_bond * per_sample_weights  # [B, Nsample]
            metrics["bond_loss"] = L_bond.detach().mean()
        else:
            L_bond = 0.0

        # Algorithm 27
        alpha_smooth_lddt = self.loss_weights["smooth_lddt"]
        if alpha_smooth_lddt > 0:
            L_smooth_lddt = self.smooth_lddt_loss(x_pred, x_true, f_input)
            metrics["smooth_lddt_loss"] = L_smooth_lddt.detach().mean()
        else:
            L_smooth_lddt = 0.0

        # Equation 6
        # NOTE: per-sample weights are already applied in L_mse and L_bond
        # L_diff = loss_weights(L_mse + α_bond * L_bond) + L_smooth_lddt
        L_diffusion = (L_mse + alpha_bond * L_bond) + L_smooth_lddt

        # Mean over diffusion samples
        L_diffusion = L_diffusion.mean(-1)  # [B, Nsample] -> [B,]
        metrics["diffusion_loss"] = L_diffusion.detach().mean()

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

        if self.train_confidence_head:
            raise NotImplementedError(
                "Logging for confidence module not implemented yet."
            )
            # self.log(
            #     "train/grad_norm_confidence_head",
            #     gradient_norm(model.confidence_head),
            #     prog_bar=False,
            # )
            # self.log(
            #     "train/param_norm_confidence_head",
            #     parameter_norm(model.confidence_head),
            #     prog_bar=False,
            # )

        pass

    # === EMA === #
    # Started from https://github.com/jwohlwend/boltz
    @property
    def use_ema(self) -> bool:
        return self.optimizer_config.use_ema

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

    def on_train_start(self) -> None:
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.optimizer_config.ema_decay
                self.ema = ExponentialMovingAverage(
                    parameters=self.parameters(), decay=ema_decay
                )
            self.ema.to(self.device)
            self.ema.store(self.parameters())

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        if self.use_ema:
            self.ema.update(self.parameters())

    def prepare_train(self) -> None:
        if self.use_ema:
            self.ema.restore(self.parameters())

    def prepare_eval(self) -> None:
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.optimizer_config.ema_decay
                self.ema = ExponentialMovingAverage(
                    parameters=self.parameters(), decay=ema_decay
                )
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def on_validation_start(self):
        self.prepare_eval()

    def on_validation_end(self) -> None:
        self.prepare_train()
