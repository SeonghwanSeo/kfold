"""Define training modules for k-fold"""

import gc
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig
from torchmetrics import MeanMetric, MetricCollection

from kfold.config import to_dict
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model.models.kfold import KFold, KFoldConfig
from kfold.training.utils.binned_loss_logging import (
    EntityBinConfig,
    EntityBinnedLossLogger,
    TimeBinConfig,
    TimeBinnedLossLogger,
)
from kfold.training.utils.gradient_logging import gradient_norm, parameter_norm
from kfold.utils.registry import MAIN_MODULE

from . import loss as loss_fn
from .metrics import structure_metrics as validation_metrics
from .optim.ema import ExponentialMovingAverage
from .optim.lr_scheduler import AF3LRScheduler


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
    train_interaction_head: bool = True
    train_structure_module: bool = True
    train_confidence_head: bool = False

    # trunk recycling
    num_recycles: int = 3
    # for structure model training
    diffusion_batch_size: int = 48
    # for confidence module training
    num_steps: int = 20
    num_diffusion_samples: int = 1

    # Logging: time-binned train losses (epoch-level)
    # If enabled, logs
    # `train_bin/uXX_YY/{loss,mse_loss,bond_loss,smooth_lddt_loss,diffusion_loss,
    # interaction_loss}`
    # where u is normalized diffusion time in [0, 1] with bins of width `time_bin_width`.
    log_time_binned_losses: bool = False
    time_bin_width: float = 0.1

    # Logging: entity-count binned train losses (epoch-level)
    # Entity count is computed as unique(token.asym_id) among valid tokens.
    # Logs `train/{metric}_entity_interval{1..10}` where interval10 means >=10.
    log_entity_binned_losses: bool = False


@dataclass(kw_only=True)
class ValidationConfig:
    """Validation step configuration."""

    num_recycles: int = 3
    num_steps: int = 20
    num_diffusion_samples: int = 5
    return_traj: bool = False
    traj_format: str = "cif"
    symmetry_correction: bool = True
    # Validation output logging
    save_predictions: bool = True


@dataclass(kw_only=True)
class LossConfig:
    """Training step configuration."""

    # TODO: better configuration

    weights: dict[str, float]
    distogram_loss: Any
    diffusion_loss: Any
    interaction_loss: Any
    confidence_loss: Any


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config: DictConfig = config
        self.config: TrainConfig = self.global_config.train
        self.training_config: TrainingConfig = self.config.training
        self.validation_config: ValidationConfig = self.config.validation
        self.optimizer_config: OptimizerConfig = self.config.optimizer
        self.loss_config: LossConfig = self.config.loss
        self.use_ema: bool = self.optimizer_config.use_ema

        # Time-binned logging state (populated only when enabled)
        self._timebin_enabled: bool = bool(self.training_config.log_time_binned_losses)

        # These are set inside loss computation to avoid recomputation
        self._timebin_last_distogram_loss_per_batch: torch.Tensor | None = None
        self._timebin_last_interaction_loss_per_batch: torch.Tensor | None = None
        self._timebin_last_diffusion_per_sample: dict[str, torch.Tensor] | None = None

        # Entity-count binned logging state
        self._entitybin_enabled: bool = bool(
            self.training_config.log_entity_binned_losses
        )

        # Cache controls: used for both time-bin and entity-bin logging
        self._binned_cache_enabled: bool = (
            self._timebin_enabled or self._entitybin_enabled
        )

        # Binned loss loggers (registered as modules for checkpointing)
        self.time_binned_logger = TimeBinnedLossLogger(
            TimeBinConfig(
                enabled=self._timebin_enabled,
                width=float(self.training_config.time_bin_width),
            )
        )
        self.entity_binned_logger = EntityBinnedLossLogger(
            EntityBinConfig(enabled=self._entitybin_enabled, nbins=10)
        )

        # Whether to train structure and confidence modules
        self.train_trunk: bool = self.training_config.train_trunk
        self.train_distogram_head: bool = self.training_config.train_distogram_head
        self.train_interaction_head: bool = self.training_config.train_interaction_head
        self.train_structure_module: bool = self.training_config.train_structure_module
        self.train_confidence_head: bool = self.training_config.train_confidence_head

        # Initialize model here
        model_config: KFoldConfig = self.global_config.model
        model_cls = MAIN_MODULE[model_config._class_]
        self.model: KFold = model_cls(model_config)

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup losses and metrics
        self.setup_losses()
        self.setup_metrics()

        # Create writer
        self.writer: KFoldWriter = KFoldWriter()

        # Save hyperparameters
        self.save_hyperparameters(to_dict(self.global_config))

        # Pre-sample recycling steps for training
        # This ensures all GPUs use the same recycling schedule
        rng = np.random.default_rng(seed=42)
        self.recycles_per_step: np.ndarray = rng.integers(
            0,
            self.training_config.num_recycles + 1,
            size=100_000,
        )

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when we train the confidence module only (Final-training-stage)

        self.frozen_modules = []
        if self.train_trunk is False:
            self.frozen_modules += ["input_embedder", "trunk"]

        if self.train_distogram_head is False:
            self.frozen_modules += ["distogram_head"]

        if (
            self.train_interaction_head is False
            and self.model.interaction_head is not None
        ):
            self.frozen_modules += ["interaction_head"]

        if self.train_structure_module is False:
            self.frozen_modules += ["score_model"]

        if self.train_confidence_head is False:
            # TODO: freeze confidence module after they are implemented
            pass

        for module_name in self.frozen_modules:
            module = getattr(self.model, module_name)
            if module is None:
                continue
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

        if self.train_interaction_head:
            if self.model.interaction_head is None:
                raise ValueError(
                    "interaction_head is not configured but interaction loss is enabled."
                )
            if loss_config.interaction_loss is None:
                raise ValueError(
                    "interaction_loss config is required "
                    "when interaction loss is enabled."
                )
            self.interaction_loss = loss_fn.interaction.InteractionLoss(
                **loss_config.interaction_loss
            )

        if self.train_confidence_head:
            raise NotImplementedError("Confidence loss not implemented yet.")

    def setup_metrics(self):
        """Setup metrics for validation"""
        # NOTE (Seonghwan): MeanMetric is required since the number of values
        # per each metric key are different for each batch during validation.
        # self.log() raises deadlock error when aggregating metrics in DDP.
        self.val_dataset_names: list[str] = [
            ds.name for ds in self.global_config.train.data.val_datasets
        ]
        val_metrics = []
        for name in self.val_dataset_names:
            dataset_metrics = {}
            for prefix in ["top1", "top5"]:
                for k in validation_metrics.main_metric_names:
                    dataset_metrics[f"{prefix}/{k}"] = MeanMetric()
            for k in validation_metrics.monitor_metric_names:
                dataset_metrics[f"monitor/{k}"] = MeanMetric()
            val_metrics.append(MetricCollection(dataset_metrics, prefix=f"{name}/"))
        self.val_metrics = torch.nn.ModuleList(val_metrics)

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
        num_recycles: int = 3,
        num_steps: int = 20,
        num_diffusion_samples: int = 1,
        diffusion_batch_size: int = 48,
        mode: str = "train",
    ) -> dict[str, dict[str, torch.Tensor]]:
        if mode == "train":
            return self.model(
                f_input,
                num_recycles=num_recycles,
                num_steps=num_steps,
                num_diffusion_samples=num_diffusion_samples,
                diffusion_batch_size=diffusion_batch_size,
                train_structure_module=self.train_structure_module,
                train_interaction_head=self.train_interaction_head,
                train_confidence_module=self.train_confidence_head,
                sample_structures=self.train_confidence_head,
            )
        elif mode == "validation":
            return_traj = self.validation_config.return_traj
            dict_out, _ = self.model.sample(
                f_input,
                num_recycles=num_recycles,
                num_steps=num_steps,
                num_diffusion_samples=num_diffusion_samples,
                return_traj=return_traj,
            )
            return {"sample": dict_out}
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(
        self,
        batch: tuple[FoldingInput, list[dict]],
        batch_idx: int,
    ) -> torch.Tensor:
        training_config = self.training_config

        f_input, _ = batch  # second one is full_structure_dict, not used in training step

        # Sample recycling steps
        # Use shared recycling schedule across all the gpus
        idx = self.global_step % len(self.recycles_per_step)
        num_recycles = int(self.recycles_per_step[idx])

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(
            f_input=f_input,
            num_recycles=num_recycles,
            num_steps=training_config.num_steps,
            num_diffusion_samples=training_config.num_diffusion_samples,
            diffusion_batch_size=training_config.diffusion_batch_size,
            mode="train",
        )
        try:
            loss, metrics = self.compute_losses(batch, out)
        except Exception as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if self._binned_cache_enabled and self.train_structure_module:
            t_hat = out.get("diffusion", {}).get("t_hat", None)
            diffusion_per_sample = self._timebin_last_diffusion_per_sample
            distogram_loss_per_batch = self._timebin_last_distogram_loss_per_batch
            interaction_loss_per_batch = self._timebin_last_interaction_loss_per_batch

            # These caches are populated inside compute_losses/compute_diffusion_loss.
            # Skip if anything is missing for this batch.
            if (
                torch.is_tensor(t_hat)
                and diffusion_per_sample is not None
                and distogram_loss_per_batch is not None
            ):
                if self._timebin_enabled:
                    self.time_binned_logger.update(
                        t_hat=t_hat,
                        structure_module=self.model.structure_module,
                        diffusion_per_sample=diffusion_per_sample,
                        distogram_loss_per_batch=distogram_loss_per_batch,
                        interaction_loss_per_batch=interaction_loss_per_batch,
                        loss_weights=self.loss_weights,
                    )
                if self._entitybin_enabled:
                    self.entity_binned_logger.update(
                        f_input=f_input,
                        diffusion_per_sample=diffusion_per_sample,
                        distogram_loss_per_batch=distogram_loss_per_batch,
                        interaction_loss_per_batch=interaction_loss_per_batch,
                        loss_weights=self.loss_weights,
                    )

        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), sync_dist=False)

        return loss

    def compute_losses(
        self, batch: tuple[FoldingInput, list[dict]], model_output: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses of given the model output."""
        f_input, _ = batch

        with torch.autocast("cuda", dtype=torch.float32):
            # NOTE: Compute the losses in float32 for better numerical stability
            # Compute losses
            if self.train_structure_module:
                distogram_loss, distogram_metrics = self.compute_distogram_loss(
                    logits=model_output["distogram"]["logits"],
                    f_input=f_input,
                )
                if "logits_aug" in model_output["distogram"]:
                    # Augmented distogram loss
                    distogram_loss_aug, _ = self.compute_distogram_loss(
                        logits=model_output["distogram"]["logits_aug"],
                        f_input=f_input,
                    )
                    # TODO (Seonghwan): This is hard-coded weight for augmented loss...
                    # Do we want to make it configurable?
                    distogram_loss = 0.8 * distogram_loss + 0.2 * distogram_loss_aug

                diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                    x_pred=model_output["diffusion"]["denoised_atom_coords"],
                    x_true=model_output["diffusion"]["true_atom_coords"],
                    per_sample_weights=model_output["diffusion"]["loss_weights"],
                    f_input=f_input,
                )

            else:
                distogram_loss, distogram_metrics = 0.0, {}
                diffusion_loss, diffusion_metrics = 0.0, {}

            interaction_weight = self.loss_weights.get("interaction", 0.0)
            interaction_loss_per_batch: torch.Tensor | None = None
            if self.train_interaction_head and interaction_weight > 0:
                interaction_loss, interaction_metrics = self.compute_interaction_loss(
                    logits=model_output["interaction"]["logits"],
                    f_input=f_input,
                )
                interaction_loss_per_batch = interaction_loss.detach()
            else:
                interaction_loss, interaction_metrics = 0.0, {}

            if self.train_confidence_head:
                confidence_loss, confidence_metrics = self.compute_confidence_loss()
            else:
                confidence_loss, confidence_metrics = 0.0, {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss_weights = self.loss_weights
        loss = (
            loss_weights["diffusion"] * diffusion_loss
            + loss_weights["distogram"] * distogram_loss
            + loss_weights["interaction"] * interaction_loss
            + loss_weights["confidence"] * confidence_loss
        )  # [B,]
        assert torch.is_tensor(loss), "Loss must be a torch.Tensor."

        # Mean over batch
        loss = loss.mean()

        # Log loss and metrics
        all_metrics = (
            distogram_metrics
            | diffusion_metrics
            | interaction_metrics
            | confidence_metrics
        )
        all_metrics["loss"] = loss.detach()

        if self._binned_cache_enabled and self.train_structure_module:
            # Used to compute per-time-bin total loss without recomputing distogram head.
            self._timebin_last_distogram_loss_per_batch = distogram_loss.detach()
            self._timebin_last_interaction_loss_per_batch = interaction_loss_per_batch

        return loss, all_metrics

    def validation_step(
        self,
        batch: tuple[FoldingInput, list[dict]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        val_config = self.validation_config
        num_diffusion_samples = val_config.num_diffusion_samples

        f_input, full_struct_list = batch
        assert f_input.batch_size == 1, "Validation batch size should be 1"
        struct_info = full_struct_list[0]
        ref_struct: RefStructure = struct_info["structure"]

        try:
            out = self(
                f_input=f_input,
                num_recycles=val_config.num_recycles,
                num_steps=val_config.num_steps,
                num_diffusion_samples=num_diffusion_samples,
                mode="validation",
            )
            sample_out = out["sample"]
            sample_coords = sample_out["sample_coordinates"]
            traj = sample_out.get("traj")
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                torch.cuda.empty_cache()
                return
            else:
                raise e

        # Remove padding atoms
        assert sample_coords.shape[:2] == (1, num_diffusion_samples), (
            "Expected sample_coords shape is (1, Nsample, Natom, 3)."
        )
        num_atoms: int = ref_struct.num_atoms
        assert f_input.atom.pad_mask[:, :num_atoms].all(), (
            "Non-padding atoms found in the padding mask."
        )
        assert not f_input.atom.pad_mask[:, num_atoms:].any(), (
            "Padding atoms found in the non-padding region of the padding mask."
        )
        sample_coords = sample_coords[0, :, :num_atoms, :]  # [Nsample, Natom, 3]

        # Compute validation metrics
        ref_struct_aligned: list[RefStructure] = []
        sample_metrics: list[dict[str, Any]] = []
        with torch.autocast("cuda", torch.float32):
            # Permute predicted and true coordinates to align
            if val_config.symmetry_correction:
                assert "symmetry" in struct_info, (
                    "symmetry_dict must be provided in struct_info "
                    "for symmetry correction during validation."
                )
            symmetry_dict = struct_info.get("symmetry", None)
            for i in range(num_diffusion_samples):
                pred_coords_i = sample_coords[i]  # [Natom, 3]
                struct_i = validation_metrics.get_aligned_structure(
                    ref_struct,
                    pred_coords_i,
                    find_best_permutation=val_config.symmetry_correction,
                    symmetry_dict=symmetry_dict,
                )
                metric_i = validation_metrics.compute_validation_metric(
                    struct_i, pred_coords_i, align=False
                )
                ref_struct_aligned.append(struct_i)
                sample_metrics.append(metric_i)

        # Aggregate metrics
        aggr_metrics = validation_metrics.aggregate_validation_metrics(sample_metrics)

        # Update validation metrics
        metrics: MetricCollection = self.val_metrics[dataloader_idx]
        for prefix in ["top1", "top5"]:
            _m = aggr_metrics[prefix]
            for k in validation_metrics.main_metric_names:
                if k in _m:
                    metrics[f"{prefix}/{k}"].update(_m[k])
        for k in validation_metrics.monitor_metric_names:
            _m = aggr_metrics["monitor"]
            if k in _m:
                metrics[f"monitor/{k}"].update(_m[k])

        # Save validation predictions if needed
        if val_config.save_predictions:
            if self.trainer.log_dir is None:
                print(
                    "Warning: trainer.log_dir is None, "
                    "skipping saving validation predictions."
                )
            else:
                dataset_name = self.val_dataset_names[dataloader_idx]
                save_dir: pathlib.Path = (
                    pathlib.Path(self.trainer.log_dir)
                    / "validation_logs"
                    / dataset_name
                    / f"epoch-{self.current_epoch}_step-{self.global_step}"
                    / ref_struct.id
                )
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save ground-truth and apo structures
                name = ref_struct.id
                self.writer.write(ref_struct, save_dir / f"{name}-gt.cif")
                self.writer.write(ref_struct, save_dir / f"{name}-apo.cif", save_apo=True)

                # Save predicted structures and metrics
                for i in range(num_diffusion_samples):
                    prefix = str(save_dir / f"{name}-sample{i}")
                    self.save_structure_and_metrics(
                        ref_struct=ref_struct_aligned[i],
                        pred_coords=sample_coords[i],
                        metrics=sample_metrics[i],
                        prefix=prefix,
                    )

                # Save trajectory if available
                if traj is not None:
                    self.save_trajectory(
                        ref_struct,
                        traj[:, 0],  # [T, Natom, 3]
                        save_dir,
                        format=val_config.traj_format,
                    )

    def on_validation_epoch_start(self):
        torch.backends.cudnn.benchmark = False

    def on_validation_epoch_end(self):
        torch.backends.cudnn.benchmark = True
        for metrics in self.val_metrics:
            if not self.trainer.sanity_checking:
                avg_values = metrics.compute()
                # NOTE: do not filter out NaN values to avoid deadlock in DDP
                self.log_dict(
                    avg_values,
                    on_step=False,
                    on_epoch=True,
                    # Already synced in compute(), but keep to avoid warning...
                    sync_dist=True,
                )
            metrics.reset()

        # Clear cache after validation
        # NOTE: is this necessary?
        gc.collect()
        torch.cuda.empty_cache()

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

    def compute_interaction_loss(
        self, logits: torch.Tensor, f_input: FoldingInput
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute interaction loss."""
        loss = self.interaction_loss(logits, f_input)
        metrics = {"interaction_loss": loss.detach().mean()}
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
        L_mse_weighted = L_mse * per_sample_weights  # [B, Nsample]
        metrics["mse_loss"] = L_mse_weighted.detach().mean()

        # Equation 5
        alpha_bond = self.loss_weights["bond"]
        if alpha_bond > 0:
            L_bond = self.bond_loss(x_pred, x_true, f_input)
            L_bond_weighted = L_bond * per_sample_weights  # [B, Nsample]
            metrics["bond_loss"] = L_bond_weighted.detach().mean()
        else:
            L_bond_weighted = None

        # Algorithm 27
        alpha_smooth_lddt = self.loss_weights["smooth_lddt"]
        if alpha_smooth_lddt > 0:
            L_smooth_lddt = self.smooth_lddt_loss(x_pred, x_true, f_input)
            metrics["smooth_lddt_loss"] = L_smooth_lddt.detach().mean()
        else:
            L_smooth_lddt = None

        # Equation 6
        # NOTE: per-sample weights are already applied in L_mse and L_bond
        # L_diff = loss_weights(L_mse + α_bond * L_bond) + L_smooth_lddt
        L_diffusion_per_sample = L_mse_weighted
        if L_bond_weighted is not None:
            L_diffusion_per_sample = L_diffusion_per_sample + alpha_bond * L_bond_weighted
        if L_smooth_lddt is not None:
            L_diffusion_per_sample = L_diffusion_per_sample + L_smooth_lddt

        # Mean over diffusion samples
        L_diffusion = L_diffusion_per_sample.mean(-1)  # [B, Nsample] -> [B,]
        metrics["diffusion_loss"] = L_diffusion.detach().mean()

        if self._binned_cache_enabled and self.train_structure_module:
            payload: dict[str, torch.Tensor] = {
                "mse_loss": L_mse_weighted.detach(),
                "diffusion_loss": L_diffusion_per_sample.detach(),
            }
            if L_bond_weighted is not None:
                payload["bond_loss"] = L_bond_weighted.detach()
            if L_smooth_lddt is not None:
                payload["smooth_lddt_loss"] = L_smooth_lddt.detach()
            self._timebin_last_diffusion_per_sample = payload

        return L_diffusion, metrics

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        out: dict[str, torch.Tensor] = {}
        out |= self.time_binned_logger.flush()
        out |= self.entity_binned_logger.flush()
        if out:
            self.log_dict(out)

    def compute_confidence_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError("Confidence loss not implemented yet.")

    # === Training logs === #
    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 10 == 0:
            self.log_model_state()

    def log_model_state(self):
        """Log model parameter and gradient norms."""

        model = self.model
        self.log("monitor/grad_norm", gradient_norm(model), prog_bar=False)
        self.log("monitor/param_norm", parameter_norm(model), prog_bar=False)

        if self.train_structure_module:
            self.log(
                "monitor/grad_norm_trunk",
                gradient_norm(model.trunk),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_trunk",
                parameter_norm(model.trunk),
                sync_dist=False,
                prog_bar=False,
            )

            self.log(
                "monitor/grad_norm_score_model",
                gradient_norm(model.score_model),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_score_model",
                parameter_norm(model.score_model),
                sync_dist=False,
                prog_bar=False,
            )

        if self.train_confidence_head:
            raise NotImplementedError(
                "Logging for confidence module not implemented yet."
            )
            # self.log(
            #     "monitor/grad_norm_confidence_head",
            #     gradient_norm(model.confidence_head),
            #     sync_dist=False,
            #     prog_bar=False,
            # )
            # self.log(
            #     "monitor/param_norm_confidence_head",
            #     parameter_norm(model.confidence_head),
            #     sync_dist=False,
            #     prog_bar=False,
            # )

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
        self._ema: ExponentialMovingAverage = value

    def on_train_start(self) -> None:
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.optimizer_config.ema_decay
                self.ema = ExponentialMovingAverage(self, decay=ema_decay)
            self.ema.to(self.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore
        optimizer.step(closure=optimizer_closure)
        if self.use_ema:
            self.ema.update(self)

    def prepare_train(self) -> None:
        if self.use_ema:
            self.ema.restore(self)

    def prepare_eval(self) -> None:
        if self.use_ema:
            if not self.is_ema_initialized:
                ema_decay = self.optimizer_config.ema_decay
                self.ema = ExponentialMovingAverage(self, decay=ema_decay)
            self.ema.store(self)
            self.ema.copy_to(self)

    def on_validation_start(self):
        self.prepare_eval()

    def on_validation_end(self) -> None:
        self.prepare_train()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema and "ema" in checkpoint:
            ema_decay = self.optimizer_config.ema_decay
            self.ema = ExponentialMovingAverage(self, decay=ema_decay)
            if self.ema.compatible(checkpoint["ema"]):
                self.ema.load_state_dict(checkpoint["ema"], device=torch.device("cpu"))
                self.ema.to(self.device)
            else:
                print(
                    "Warning: EMA state not loaded due to incompatible model parameters."
                )
                self.use_ema = False  # Disable EMA if not compatible
                del self._ema

    # === Helper functions === #
    def save_structure_and_metrics(
        self,
        ref_struct: RefStructure,
        pred_coords: torch.Tensor,
        metrics: dict[str, Any],
        prefix: str,
    ):
        """Save predicted and ground-truth structures as mmCIF files."""
        num_atoms = ref_struct.num_atoms
        assert pred_coords.shape == (num_atoms, 3), (
            "pred_coords must have shape (Natoms, 3)."
        )
        # Save metrics
        with open(f"{prefix}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save aligned ground-truth structure
        aligned_gt_path = f"{prefix}-gt_aligned.cif"
        self.writer.write(ref_struct, aligned_gt_path)

        # Save predicted structure
        rmsd = metrics["metrics"]["rmsd"]
        lddt = metrics["metrics"]["lddt"] * 100  # scale to [0, 100]
        pred_path = f"{prefix}-rmsd{rmsd:.2f}-lddt{lddt:.2f}.cif"
        self.writer.write_new_coords(ref_struct, pred_coords.cpu().numpy(), pred_path)

    def save_trajectory(
        self,
        ref_struct: RefStructure,
        traj: torch.Tensor,
        save_dir: pathlib.Path,
        format: str = "cif",
    ):
        """Save predicted and ground-truth structures as mmCIF files."""
        name: str = ref_struct.id

        assert traj.ndim == 4, "Trajectory must be of shape (Nframe, Nsample, Natom, 3)"
        num_samples: int = traj.shape[1]

        # Remove padding atoms
        num_atoms: int = ref_struct.num_atoms
        traj: np.ndarray = traj[:, :, :num_atoms, :].detach().cpu().numpy()

        # Compute structure metrics
        for i in range(num_samples):
            # Save trajectory
            traj_i = traj[:, i, :, :]  # [Nframe, Natom, 3]
            save_path = save_dir / f"{name}-sample-{i}-traj.{format}"
            self.writer.write_trajectory(ref_struct, traj_i, save_path)
