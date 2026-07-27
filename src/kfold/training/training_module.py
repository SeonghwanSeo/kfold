"""Define training modules for k-fold"""

import dataclasses
import gc
import json
import pathlib
from typing import Any, Self

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from kfold.config import to_dict
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model.model_train import KFoldConfig, KFoldForTrain
from kfold.training.utils.binned_loss_logging import (
    EntityBinConfig,
    EntityBinnedLossLogger,
    TimeBinConfig,
    TimeBinnedLossLogger,
)
from kfold.training.utils.gradient_logging import gradient_norm, parameter_norm
from kfold.utils import confidence_metrics
from kfold.utils.geometry.rigid_align import compute_rmsd

from . import loss as loss_fn
from .metrics import structure_metrics as validation_metrics
from .optim.ema import ExponentialMovingAverage
from .optim.lr_scheduler import AF3LRScheduler

_PARCAE_RECURRENCE_BASE_SEED = 42
_PARCAE_RECURRENCE_SCHEDULE_SIZE = 100_000
_PARCAE_RECURRENCE_SAMPLING_MODES = {"shared", "rank_independent"}


class _Config:
    @classmethod
    def from_dict(cls, config) -> Self:
        merged = OmegaConf.merge(OmegaConf.create(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
    """Configuration for training and validation steps."""

    name: str
    out_dir: str
    seed: int
    compile: "CompileConfig"
    training: "TrainingConfig"
    validation: "ValidationConfig"
    optimizer: "OptimizerConfig"
    loss: "LossConfig"


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig(_Config):
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
    ema_decay: float = 0.999
    validate_with_ema_after_n_steps: int = 10000
    # multi-phase training
    load_opt_state: bool = True
    final_training_stage: bool = False


@dataclasses.dataclass(kw_only=True)
class ParcaeTrainConfig(_Config):
    """Training-time Parcae recycle-count sampling configuration."""

    max_recycles: int = 5
    min_recycles: int = 0
    poisson_mean: float = 2.0
    grad_recurrence_steps: int = 2
    recurrence_sampling_mode: str = "shared"


def _build_clamped_poisson_recycle_schedule(
    config: ParcaeTrainConfig,
    seed: int,
    size: int = _PARCAE_RECURRENCE_SCHEDULE_SIZE,
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    sampled_recycles = rng.poisson(
        lam=config.poisson_mean,
        size=size,
    )
    return np.clip(
        sampled_recycles,
        config.min_recycles,
        config.max_recycles,
    ).astype(np.int64)


def _select_recycle_count(schedule: np.ndarray, global_step: int) -> int:
    idx = global_step % len(schedule)
    return int(schedule[idx])


@dataclasses.dataclass(kw_only=True)
class TrainingConfig(_Config):
    """Training step configuration."""

    # Whether to train each submodules
    train_trunk: bool = True
    train_diffusion_head: bool = True
    train_confidence_head: bool = False

    # trunk recycling; Parcae training-time sampling below owns the active
    # recycle schedule, and this value is kept for config/checkpoint compatibility.
    num_recycles: int = 3
    parcae: ParcaeTrainConfig = dataclasses.field(default_factory=ParcaeTrainConfig)
    # for structure model training
    diffusion_batch_size: int = 48
    # for confidence module training
    num_mini_rollout_steps: int = 20
    num_mini_rollout_samples: int = 1

    # Logging: time-binned train losses (epoch-level)
    # If enabled, logs
    # `train_bin/uXX_YY/{loss,mse_loss,bond_loss,smooth_lddt_loss,diffusion_loss}
    # where u is normalized diffusion time in [0, 1] with bins of width `time_bin_width`.
    log_time_binned_losses: bool = False
    time_bin_width: float = 0.1

    # Logging: entity-count binned train losses (epoch-level)
    # Entity count is computed as unique(token.asym_id) among valid tokens.
    # Logs `train/{metric}_entity_interval{1..10}` where interval10 means >=10.
    log_entity_binned_losses: bool = False


def _get_diffusion_time_for_binning(
    diffusion_out: dict[str, Any],
) -> torch.Tensor | None:
    """Return the diffusion time tensor used by generic time-bin logging."""
    t_hat = diffusion_out.get("t_hat")
    if torch.is_tensor(t_hat):
        return t_hat
    t = diffusion_out.get("t")
    if torch.is_tensor(t):
        return t
    return None


def _get_structure_module_for_binning(model: Any) -> Any:
    """Return the structure module that defines time/sigma bounds for binning."""
    diffusion_head = getattr(model, "diffusion_head", None)
    if diffusion_head is not None:
        return diffusion_head

    structure_module = getattr(model, "structure_module", None)
    if structure_module is not None:
        return structure_module

    raise AttributeError(
        "Model must expose either diffusion_head or structure_module for "
        "time-binned loss logging."
    )


def _binary_average_precision(
    score: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    score = score[mask]
    target = target[mask]
    if score.numel() == 0 or not target.any() or target.all():
        return None

    order = torch.argsort(score, descending=True)
    sorted_target = target[order].to(score.dtype)
    rank = torch.arange(
        1, sorted_target.numel() + 1, device=score.device, dtype=score.dtype
    )
    precision = sorted_target.cumsum(dim=0) / rank
    return (precision * sorted_target).sum() / sorted_target.sum().clamp(min=1.0)


@dataclasses.dataclass(kw_only=True)
class ValidationConfig(_Config):
    """Validation step configuration."""

    num_recycles: int = 3
    num_steps: int = 20
    num_diffusion_samples: int = 5
    return_traj: bool = False
    traj_format: str = "cif"
    # Validation output logging
    save_predictions: bool = False


@dataclasses.dataclass(kw_only=True)
class LossConfig(_Config):
    """Loss configuration."""

    weights: dict[str, float]
    distogram_loss: Any
    diffusion_loss: Any
    confidence_loss: Any
    patch_geometry_loss: Any = dataclasses.field(default_factory=dict)
    interface_contact_loss: Any = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class CompileConfig(_Config):
    enabled: bool = False
    mode: str = "default"
    dynamic: bool = False


class KFoldTrainingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.global_config: DictConfig = config
        self.config: TrainConfig = config.train
        self.training_config: TrainingConfig = TrainingConfig.from_dict(
            self.config.training
        )
        self.parcae_train_config: ParcaeTrainConfig = ParcaeTrainConfig.from_dict(
            self.training_config.parcae
        )
        self.validation_config: ValidationConfig = ValidationConfig.from_dict(
            self.config.validation
        )
        self.optimizer_config: OptimizerConfig = OptimizerConfig.from_dict(
            self.config.optimizer
        )
        self.loss_config: LossConfig = LossConfig.from_dict(self.config.loss)
        self.compile_config: CompileConfig = CompileConfig.from_dict(self.config.compile)

        # Save hyperparameters
        self.save_hyperparameters(to_dict(self.global_config))

        # Whether to train structure and confidence modules
        self.train_trunk: bool = self.training_config.train_trunk
        self.train_diffusion_head: bool = self.training_config.train_diffusion_head
        self.train_confidence_head: bool = self.training_config.train_confidence_head

        # Initialize model here
        model_config: KFoldConfig = self.global_config.model
        self.model = KFoldForTrain(model_config)

        # Compile
        if self.compile_config.enabled:
            self.model.do_compile(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup EMA
        self.submodules_to_ignore_for_ema = (
            "prot_seq_encoder",
            "rna_seq_encoder",
            "prot_struct_encoder",
        )
        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            model=self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.submodules_to_ignore_for_ema,
        )
        self.stored_weights: dict[str, torch.Tensor] | None = None
        self.last_lr_step = -1

        # Setup losses and metrics
        self.setup_losses()
        self.setup_metrics()

        # Create writer
        self.writer: KFoldWriter = KFoldWriter()

        # Parcae methodology: pre-sample a clamped-Poisson recycle schedule for
        # training. The trunk still runs num_recycles + 1 loops, and the number
        # of recurrent trunk steps saved for backprop is controlled by
        # parcae.grad_recurrence_steps. "shared" preserves the previous
        # cross-rank lockstep schedule; "rank_independent" lazily builds a
        # deterministic rank-specific schedule once Lightning rank is known.
        self._shared_recycles_per_step: np.ndarray = (
            _build_clamped_poisson_recycle_schedule(
                self.parcae_train_config,
                seed=_PARCAE_RECURRENCE_BASE_SEED,
            )
        )
        self._rank_independent_recycles_per_step: np.ndarray | None = None
        self._rank_independent_recycles_rank: int | None = None
        # Backward-compatible alias for code/tests that read the existing attr.
        self.recycles_per_step: np.ndarray = self._shared_recycles_per_step

        # Time-binned logging state (populated only when enabled)
        self._timebin_enabled: bool = bool(self.training_config.log_time_binned_losses)

        # These are set inside loss computation to avoid recomputation
        self._timebin_last_distogram_loss_per_batch: torch.Tensor | None = None
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

    def _get_recurrence_schedule_rank(self) -> int:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return 0
        return int(getattr(trainer, "global_rank", 0))

    def _get_active_recycles_per_step(self) -> np.ndarray:
        if self.parcae_train_config.recurrence_sampling_mode == "shared":
            self.recycles_per_step = self._shared_recycles_per_step
            return self.recycles_per_step

        rank = self._get_recurrence_schedule_rank()
        if (
            self._rank_independent_recycles_per_step is None
            or self._rank_independent_recycles_rank != rank
        ):
            self._rank_independent_recycles_per_step = (
                _build_clamped_poisson_recycle_schedule(
                    self.parcae_train_config,
                    seed=_PARCAE_RECURRENCE_BASE_SEED + rank,
                )
            )
            self._rank_independent_recycles_rank = rank

        self.recycles_per_step = self._rank_independent_recycles_per_step
        return self.recycles_per_step

    def _get_num_recycles_for_current_step(self) -> int:
        schedule = self._get_active_recycles_per_step()
        return _select_recycle_count(schedule, int(self.global_step))

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when we train the confidence module only (Final-training-stage)

        self.frozen_modules = []
        self.frozen_modules += self.model.get_pretrained_module_names()

        if self.train_trunk is False:
            self.frozen_modules += self.model.get_trunk_module_names()
            self.frozen_modules += self.model.get_distogram_head_module_names()

            # freeze trunk Parcae params directly, as they are not treated as modules
            for param_name in self.model.get_trunk_parameter_names():
                param = getattr(self.model, param_name)
                param.requires_grad_(False)

        if self.train_diffusion_head is False:
            self.frozen_modules += self.model.get_diffusion_head_module_names()

        if self.train_confidence_head is False:
            self.frozen_modules += self.model.get_confidence_head_module_names()

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
        self.loss_weights: dict[str, float] = loss_config.weights

        # Distogram loss
        self.distogram_loss = loss_fn.distogram.DistogramLoss(
            **loss_config.distogram_loss
        )
        self.patch_geometry_loss = loss_fn.patch_geometry.PatchPairGeometryLoss(
            **loss_config.patch_geometry_loss
        )
        self.interface_contact_loss = (
            loss_fn.interface_contact.InterfaceContactBalancedLoss(
                **loss_config.interface_contact_loss
            )
        )

        # Diffusion loss
        diffusion_loss_config = loss_config.diffusion_loss
        self.weighted_mse_loss = loss_fn.diffusion.WeightedMSELoss(
            **diffusion_loss_config["mse_loss"]
        )
        # Only used in fine-tuning stage
        self.bond_loss = loss_fn.diffusion.BondLoss(**diffusion_loss_config["bond_loss"])
        # Only used in regular training stage
        self.smooth_lddt_loss = loss_fn.diffusion.SmoothLDDTLoss(
            **diffusion_loss_config["smooth_lddt_loss"]
        )

        confidence_loss_config = loss_config.confidence_loss
        # pLDDT loss
        self.plddt_loss = loss_fn.confidence.PLDDTLoss(
            **confidence_loss_config["plddt_loss"]
        )

        # PDE loss
        self.pde_loss = loss_fn.confidence.PDELoss(**confidence_loss_config["pde_loss"])

        # Experimentally resolved loss
        self.exp_res_loss = loss_fn.confidence.ExperimentallyResolvedPredictionLoss(
            **confidence_loss_config["experimentally_resolved_loss"]
        )

        # PAE loss
        self.pae_loss = loss_fn.confidence.PAELoss(**confidence_loss_config["pae_loss"])

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
            dataset_metrics["monitor/distogram_loss"] = MeanMetric()
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

        if self.last_lr_step != -1:
            for param_group in optimizer.param_groups:
                param_group.setdefault("initial_lr", config.base_lr)

        if config.lr_scheduler == "af3":
            scheduler = AF3LRScheduler(
                optimizer,
                last_epoch=self.last_lr_step,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                warmup_no_steps=config.lr_warmup_no_steps,
                start_decay_after_n_steps=config.lr_start_decay_after_n_steps,
                decay_every_n_steps=config.lr_decay_every_n_steps,
                decay_factor=config.lr_decay_factor,
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented yet."
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(
        self, f_input: FoldingInput, mode: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        if mode == "train":
            training_config = self.training_config
            num_recycles = self._get_num_recycles_for_current_step()
            return self.model.forward_train(
                f_input,
                num_recycles=num_recycles,
                grad_recurrence_steps=self.parcae_train_config.grad_recurrence_steps,
                num_mini_rollout_steps=training_config.num_mini_rollout_steps,
                num_mini_rollout_samples=training_config.num_mini_rollout_samples,
                diffusion_batch_size=training_config.diffusion_batch_size,
                train_trunk=self.train_trunk,
                train_diffusion_head=self.train_diffusion_head,
                train_confidence_module=self.train_confidence_head,
            )
        elif mode == "validation":
            val_config = self.validation_config
            dict_out = self.model.sample_validation(
                f_input,
                num_recycles=val_config.num_recycles,
                num_steps=val_config.num_steps,
                num_samples=val_config.num_diffusion_samples,
                return_traj=val_config.return_traj,
            )
            return dict_out
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(
        self,
        batch: tuple[FoldingInput, list[dict]],
        batch_idx: int,
    ) -> torch.Tensor:
        f_input, _ = batch  # second one is full_structure_dict, not used in training step

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(f_input=f_input, mode="train")
        with torch.autocast("cuda", dtype=torch.float32):
            loss, metrics = self.compute_losses(batch, out)

        if self._binned_cache_enabled and self.train_diffusion_head:
            diffusion_out = out.get("diffusion", {})
            t_for_bins = _get_diffusion_time_for_binning(diffusion_out)
            diffusion_per_sample = self._timebin_last_diffusion_per_sample
            distogram_loss_per_batch = self._timebin_last_distogram_loss_per_batch

            # These caches are populated inside compute_losses/compute_diffusion_loss.
            # Skip if anything is missing for this batch.
            if diffusion_per_sample is not None and distogram_loss_per_batch is not None:
                if self._timebin_enabled and torch.is_tensor(t_for_bins):
                    self.time_binned_logger.update(
                        t_hat=t_for_bins,
                        structure_module=_get_structure_module_for_binning(self.model),
                        diffusion_per_sample=diffusion_per_sample,
                        distogram_loss_per_batch=distogram_loss_per_batch,
                        loss_weights=self.loss_weights,
                    )
                if self._entitybin_enabled:
                    self.entity_binned_logger.update(
                        f_input=f_input,
                        diffusion_per_sample=diffusion_per_sample,
                        distogram_loss_per_batch=distogram_loss_per_batch,
                        loss_weights=self.loss_weights,
                    )

        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), sync_dist=False)

        return loss

    def compute_losses(
        self, batch: tuple[FoldingInput, list[dict]], model_output: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses of given the model output."""
        f_input, struct_info = batch

        # NOTE: Compute the losses in float32 for better numerical stability
        # Compute losses
        if self.train_trunk:
            distogram_loss, distogram_metrics = self.compute_distogram_loss(
                logits=model_output["distogram"]["logits"],
                f_input=f_input,
            )
            interface_contact_weight = self.loss_weights.get(
                "interface_contact",
                0.0,
            )
            if interface_contact_weight > 0:
                interface_contact_loss, interface_contact_metrics = (
                    self.interface_contact_loss(
                        logits=model_output["distogram"]["logits"],
                        f_input=f_input,
                    )
                )
            else:
                interface_contact_loss, interface_contact_metrics = 0.0, {}
            patch_weight = self.loss_weights.get("patch_geometry", 0.0)
            if patch_weight > 0 and "patch_geometry" in model_output:
                patch_geometry_loss, patch_geometry_metrics = self.patch_geometry_loss(
                    model_output["patch_geometry"]
                )
            else:
                patch_geometry_loss, patch_geometry_metrics = 0.0, {}
        else:
            distogram_loss, distogram_metrics = 0.0, {}
            interface_contact_loss, interface_contact_metrics = 0.0, {}
            patch_geometry_loss, patch_geometry_metrics = 0.0, {}

        if self.train_diffusion_head:
            diffusion_out = model_output["diffusion"]
            diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                x_pred=diffusion_out["x_0_hat"],
                x_true=diffusion_out["x_gt"],
                f_input=f_input,
                per_sample_weights=model_output["diffusion"]["loss_weights"],
            )

        else:
            diffusion_loss, diffusion_metrics = 0.0, {}

        if self.train_confidence_head:
            x_pred = model_output["sample"]["coordinates"]
            confidence_loss_mask = torch.tensor(
                [info["train_confidence_head"] for info in struct_info],
                device=x_pred.device,
                dtype=torch.bool,
            )
            x_gt, mask_gt = loss_fn.confidence.get_aligned_gt_structure(
                x_pred=x_pred,
                f_input=f_input,
                struct_info=struct_info,
            )  # [B, Nsample, Latom, 3]
            confidence_loss, confidence_metrics = self.compute_confidence_loss(
                logits=model_output["confidence"],
                x_pred=x_pred,
                x_gt=x_gt,
                mask=mask_gt,
                f_input=f_input,
                loss_mask=confidence_loss_mask,
            )
            # Log the rmsd between mini-rollout sample and GT.
            rmsd = compute_rmsd(x_pred, x_gt, mask_gt, align=True)  # [B, Nsample]
            sample_metrics = {"mini_rollout_rmsd": rmsd.mean()}

        else:
            confidence_loss, confidence_metrics = 0.0, {}
            sample_metrics = {}

        # Aggregate losses
        # See Section 5.3 Equation 15
        loss_weights = self.loss_weights
        loss = (
            loss_weights["diffusion"] * diffusion_loss
            + loss_weights["distogram"] * distogram_loss
            + loss_weights["confidence"] * confidence_loss
            + loss_weights.get("patch_geometry", 0.0) * patch_geometry_loss
            + loss_weights.get("interface_contact", 0.0) * interface_contact_loss
        )  # [B,]
        assert torch.is_tensor(loss), "Loss must be a torch.Tensor."

        # Log loss and metrics
        all_metrics = (
            distogram_metrics
            | interface_contact_metrics
            | patch_geometry_metrics
            | diffusion_metrics
            | confidence_metrics
            | sample_metrics
        )
        all_metrics["loss"] = loss.detach()

        return loss, all_metrics

    def validation_step(
        self,
        batch: tuple[FoldingInput, dict],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        val_config = self.validation_config
        num_samples = val_config.num_diffusion_samples

        f_input, struct_info = batch
        assert not f_input.is_batched, "Validation input should not be batched."
        ref_struct: RefStructure = struct_info["structure"]
        symmetry_dict: dict = struct_info["symmetry"]

        try:
            model_out: dict[str, dict[str, torch.Tensor]] = self(
                f_input=f_input, mode="validation"
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                torch.cuda.empty_cache()
                return
            else:
                raise e
        diffusion_out = model_out["diffusion"]
        distogram_out = model_out["distogram"]
        confidence_out = model_out["confidence"]

        token_mask = f_input.token.pad_mask  # [L,]
        atom_mask = f_input.atom.pad_mask  # [Natom,]
        n_tokens: int = int(token_mask.sum().item())
        n_atoms: int = int(atom_mask.sum().item())
        assert n_atoms == ref_struct.num_atoms

        # Compute validation metrics
        ref_struct_aligned: list[RefStructure] = []
        sample_metrics: list[dict[str, Any]] = []
        with torch.autocast("cuda", torch.float32):
            distogram_loss = self.compute_validation_distogram_loss(
                distogram_out, f_input
            )

            # Permute predicted and true coordinates to align
            for i in range(num_samples):
                pred_coords_i = diffusion_out["coordinates"][i, :n_atoms]
                struct_i = validation_metrics.get_aligned_gt_structure(
                    ref_struct,
                    pred_coords_i,
                    symmetry_dict=symmetry_dict,
                )
                metric_i = validation_metrics.compute_validation_metric(
                    struct_i, pred_coords_i
                )
                ref_struct_aligned.append(struct_i)
                sample_metrics.append(metric_i)

            # Select the best sample based on global PDE score.
            top1_index = None  # Use oracle sample.
            if self.train_confidence_head:
                pde = confidence_metrics.compute_pde(
                    confidence_out["pde_logits"],
                    confidence_out["pde_bin_centers"],
                    mask=token_mask,
                )  # [Nsample, Ntoken, Ntoken]
                prob_contact = distogram_out["prob_contact"]
                gpde: torch.Tensor = validation_metrics.compute_global_pde(
                    pde[:, :n_tokens, :n_tokens],
                    prob_contact[:n_tokens, :n_tokens],
                )  # [Nsample,]
                assert gpde.shape == (num_samples,)
                top1_index = int(gpde.argmin().item())

            # Aggregate metrics
            aggr_metrics = validation_metrics.aggregate_validation_metrics(
                sample_metrics, top1_index
            )

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
        metrics["monitor/distogram_loss"].update(distogram_loss)

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

                # Save predicted structures and metrics
                for i in range(num_samples):
                    prefix = str(save_dir / f"{name}-sample{i}")
                    self.save_structure_and_metrics(
                        ref_struct=ref_struct_aligned[i],
                        pred_coords=diffusion_out["coordinates"][i, :n_atoms],
                        metrics=sample_metrics[i],
                        prefix=prefix,
                    )

                # Save trajectory if available
                if "traj" in diffusion_out:
                    traj = diffusion_out["traj"][0]  # remove batch dim
                    self.save_trajectory(
                        ref_struct,
                        traj,
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
    def compute_validation_distogram_loss(
        self,
        distogram_out: dict[str, torch.Tensor],
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Compute validation distogram loss from inference outputs."""
        logits = distogram_out.get("logits")
        if logits is None:
            logits = distogram_out["distogram"]
        if not f_input.is_batched:
            f_input = f_input.add_batch_dim()
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)

        loss_per_batch = self.distogram_loss(logits, f_input)
        return loss_per_batch.mean().detach()

    def compute_distogram_loss(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
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
            The computed distogram loss (scalar).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        loss_per_batch = self.distogram_loss(logits, f_input)
        loss = loss_per_batch.mean()
        metrics = {"distogram_loss": loss.detach()}
        if (
            hasattr(self.distogram_loss, "boundaries")
            and hasattr(f_input, "token")
            and self.global_step % 10 == 0
        ):
            metrics |= self.compute_distogram_diagnostic_metrics(logits, f_input)
        if self._binned_cache_enabled and self.train_diffusion_head:
            self._timebin_last_distogram_loss_per_batch = loss_per_batch.detach()
        return loss, metrics

    def compute_distogram_diagnostic_metrics(
        self,
        logits: torch.Tensor,
        f_input: FoldingInput,
        near_cutoff: float = 12.0,
        far_cutoff: float = 22.0,
    ) -> dict[str, torch.Tensor]:
        """Log inter-chain near-ranking and false-positive pressure diagnostics."""
        with torch.no_grad():
            boundaries = self.distogram_loss.boundaries.to(logits.device)
            gt_coords = f_input.token.repr_coords
            diff = gt_coords[..., None, :, :] - gt_coords[..., :, None, :]
            d_repr = diff.norm(dim=-1)

            repr_mask = f_input.token.repr_mask
            pair_mask = repr_mask[..., None, :] & repr_mask[..., :, None]
            asym_id = f_input.token.asym_id
            inter_chain = asym_id[..., None, :] != asym_id[..., :, None]
            upper_tri = torch.ones(
                logits.shape[-3],
                logits.shape[-2],
                dtype=torch.bool,
                device=logits.device,
            ).triu(diagonal=1)
            valid = pair_mask & inter_chain & upper_tri

            if not valid.any():
                zero = logits.sum().detach() * 0.0
                return {
                    "distogram_inter_chain_valid_pairs": zero,
                    "distogram_inter_chain_near_pairs": zero,
                    "distogram_inter_chain_far_pairs": zero,
                }

            bin_size = float(boundaries[1].item() - boundaries[0].item())
            near_bin = int((near_cutoff - float(boundaries[0].item())) / bin_size)
            near_bin = max(0, min(near_bin, logits.shape[-1] - 1))
            p_near = torch.softmax(logits.float(), dim=-1)[..., : near_bin + 1].sum(
                dim=-1
            )

            true_near = (d_repr < near_cutoff) & valid
            true_far = (d_repr > far_cutoff) & valid
            ap = _binary_average_precision(p_near, true_near, valid)

            valid_count = valid.float().sum().clamp(min=1.0)
            metrics = {
                "distogram_inter_chain_valid_pairs": valid.float().sum().detach(),
                "distogram_inter_chain_near_pairs": true_near.float().sum().detach(),
                "distogram_inter_chain_far_pairs": true_far.float().sum().detach(),
                "distogram_inter_chain_target_near_rate": (
                    true_near.float().sum() / valid_count
                ).detach(),
                "distogram_inter_chain_pred_near_mass": p_near[valid].mean().detach(),
            }
            if ap is not None:
                metrics["distogram_inter_chain_near_ap"] = ap.detach()
            if true_near.any():
                metrics["distogram_inter_chain_p_near_true_near"] = (
                    p_near[true_near].mean().detach()
                )
            if true_far.any():
                metrics["distogram_inter_chain_false_positive_near_mass"] = (
                    p_near[true_far].mean().detach()
                )
            return metrics

    def compute_diffusion_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_input: FoldingInput,
        per_sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute diffusion structure loss.
        See Section 3.7.1 Diffusion Training.

        Parameters
        ----------
        x_pred : torch.Tensor
            The predicted atom coordinates of shape (B, Nsample, Latom, 3).
        x_true : torch.Tensor
            The ground truth atom coordinates of shape (B, Nsample, Latom, 3).
        f_input : FoldingInput
            The input features containing the target distogram and masks.
        per_sample_weights : torch.Tensor
            The per-sample loss weights of shape (B, Nsample),
            which is computed from the diffusion noise scale.

        Returns
        -------
        diffusion_loss : torch.Tensor
            The computed diffusion loss (scalar).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}
        alpha_chain_com = self.loss_weights["chain_com"]
        alpha_bond = self.loss_weights["bond"]
        alpha_smooth_lddt = self.loss_weights["smooth_lddt"]

        # Equations 3-4
        L_mse, L_chain_com = self.weighted_mse_loss(
            x_pred, x_true, f_input, compute_chain_com_loss=alpha_chain_com > 0
        )
        L_mse_weighted = L_mse * per_sample_weights  # [B, Nsample]
        metrics["mse_loss"] = L_mse_weighted.detach().mean()

        if alpha_chain_com > 0:
            assert L_chain_com is not None
            L_chain_com_weighted = L_chain_com * per_sample_weights  # [B, Nsample]
            metrics["chain_com_loss"] = L_chain_com_weighted.detach().mean()
        else:
            L_chain_com_weighted = None

        # Equation 5
        if alpha_bond > 0:
            L_bond = self.bond_loss(x_pred, x_true, f_input)
            L_bond_weighted = L_bond * per_sample_weights  # [B, Nsample]
            metrics["bond_loss"] = L_bond_weighted.detach().mean()
        else:
            L_bond_weighted = None

        # Algorithm 27
        if alpha_smooth_lddt > 0:
            L_smooth_lddt = self.smooth_lddt_loss(x_pred, x_true, f_input)  # [B, Nsample]
            metrics["smooth_lddt_loss"] = L_smooth_lddt.detach().mean()
        else:
            L_smooth_lddt = None

        # Equation 6
        # NOTE: per-sample weights are already applied in L_mse and L_bond
        # L_diff = loss_weights(L_mse + α_com * L_com + α_bond * L_bond) + L_smooth_lddt
        L_diffusion_per_sample = L_mse_weighted
        if L_chain_com_weighted is not None:
            L_diffusion_per_sample = (
                L_diffusion_per_sample + alpha_chain_com * L_chain_com_weighted
            )
        if L_bond_weighted is not None:
            L_diffusion_per_sample = L_diffusion_per_sample + alpha_bond * L_bond_weighted
        if L_smooth_lddt is not None:
            L_diffusion_per_sample = (
                L_diffusion_per_sample + alpha_smooth_lddt * L_smooth_lddt
            )

        # Mean over diffusion samples
        L_diffusion = L_diffusion_per_sample.mean()
        metrics["diffusion_loss"] = L_diffusion.detach()

        if self._binned_cache_enabled and self.train_diffusion_head:
            payload: dict[str, torch.Tensor] = {
                "mse_loss": L_mse_weighted.detach(),
                "diffusion_loss": L_diffusion_per_sample.detach(),
            }
            if L_chain_com_weighted is not None:
                payload["chain_com_loss"] = L_chain_com_weighted.detach()
            if L_bond_weighted is not None:
                payload["bond_loss"] = L_bond_weighted.detach()
            if L_smooth_lddt is not None:
                payload["smooth_lddt_loss"] = L_smooth_lddt.detach()
            self._timebin_last_diffusion_per_sample = payload

        return L_diffusion, metrics

    def compute_confidence_loss(
        self,
        logits: dict[str, torch.Tensor],
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        mask: torch.Tensor,
        f_input: FoldingInput,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute confidence head loss.

        Parameters
        ----------
        logits : dict[str, torch.Tensor]
            A dictionary containing the logits for different confidence predictions.
        x_pred : torch.Tensor
            The mini-rollout sample coordinates of shape (B, Nsample, Latom, 3).
        x_gt : torch.Tensor
            The GT coordinates aligned to x_pred of shape (B, Nsample, Latom, 3).
        mask : torch.Tensor
            The mask indicating which residues to include in the loss computation,
            of shape (B, Nsample, Latom).
        f_input : FoldingInput
            The input features containing the target distogram and masks.
        loss_mask : torch.Tensor
            A boolean tensor of shape (B,) indicating which samples in the batch should
            contribute to the confidence loss.

        Returns
        -------
        confidence_loss : torch.Tensor
            The computed confidence loss (scalar).
        metrics : dict[str, torch.Tensor]
            A dictionary containing loss metrics.
        """
        metrics: dict[str, torch.Tensor] = {}
        num_samples = x_pred.shape[1]
        loss_mask = loss_mask.float()[:, None]  # [B, 1]
        num_valid_samples = (loss_mask.sum() * num_samples).clamp(1)

        L_pde = self.pde_loss(logits["pde_logits"], x_pred, x_gt, mask, f_input)
        L_pde = L_pde * loss_mask  # [B, Nsample]
        metrics["pde_loss"] = L_pde.detach().sum() / num_valid_samples

        L_plddt = self.plddt_loss(logits["plddt_logits"], x_pred, x_gt, mask, f_input)
        L_plddt = L_plddt * loss_mask  # [B, Nsample]
        metrics["plddt_loss"] = L_plddt.detach().sum() / num_valid_samples

        is_resolved = mask
        pad_mask = f_input.atom.pad_mask
        L_resolved = self.exp_res_loss(logits["resolved_logits"], is_resolved, pad_mask)
        L_resolved = L_resolved * loss_mask  # [B, Nsample]
        metrics["resolved_loss"] = L_resolved.detach().sum() / num_valid_samples

        # NOTE: PAE loss return 0.0 when alpha_pae is 0.
        L_pae = self.pae_loss(logits["pae_logits"], x_pred, x_gt, mask, f_input)
        L_pae = L_pae * loss_mask  # [B, Nsample]
        metrics["pae_loss"] = L_pae.detach().sum() / num_valid_samples

        L_confidence_per_sample = L_pde + L_plddt + L_resolved + L_pae
        L_confidence = L_confidence_per_sample.sum() / num_valid_samples

        metrics["confidence_loss"] = L_confidence.detach()

        return L_confidence, metrics

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        out: dict[str, torch.Tensor] = {}
        out |= self.time_binned_logger.flush()
        out |= self.entity_binned_logger.flush()
        if out:
            self.log_dict(out)

    # === Training logs === #
    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % 10 == 0:
            self.log_model_state()

    def log_model_state(self):
        """Log model parameter and gradient norms."""

        model = self.model
        self.log("monitor/grad_norm", gradient_norm(model), prog_bar=False)
        self.log("monitor/param_norm", parameter_norm(model), prog_bar=False)

        if self.train_trunk:
            self.log(
                "monitor/grad_norm_lm_stack",
                gradient_norm(model.lm_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_lm_stack",
                parameter_norm(model.lm_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/grad_norm_main_stack",
                gradient_norm(model.main_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_refine_stack",
                parameter_norm(model.refine_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/grad_norm_refine_stack",
                gradient_norm(model.refine_stack),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_main_stack",
                parameter_norm(model.main_stack),
                sync_dist=False,
                prog_bar=False,
            )

        if self.train_diffusion_head:
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
            self.log(
                "monitor/grad_norm_confidence_head",
                gradient_norm(model.confidence_head),
                sync_dist=False,
                prog_bar=False,
            )
            self.log(
                "monitor/param_norm_confidence_head",
                parameter_norm(model.confidence_head),
                sync_dist=False,
                prog_bar=False,
            )

        pass

    def _remove_orig_mod_from_state_dict(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove '._orig_mod.' from state dict keys if present."""
        return {
            k.replace("._orig_mod.", ".") if "._orig_mod." in k else k: v
            for k, v in state_dict.items()
        }

    def _add_orig_mod_to_state_dict(
        self, state_dict: dict[str, Any], model_state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Add '._orig_mod.' to state dict keys if required"""
        model_keys = set(model_state_dict.keys())
        state_keys = set(state_dict.keys())

        # Keys expected by the compiled model but missing in the checkpoint
        remaining_keys = model_keys - state_keys
        if len(remaining_keys) == 0:
            return state_dict  # No modification needed

        new_state_dict = dict(state_dict)
        for rk in remaining_keys:
            k = rk.replace("._orig_mod.", ".")
            if k in state_dict:
                new_state_dict[rk] = new_state_dict.pop(k)
        return new_state_dict

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # remove pretrained model keys
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if "prot_seq_encoder." not in k
            and "rna_seq_encoder." not in k
            and "prot_struct_encoder." not in k
        }

        # Remove '._orig_mod.' from checkpoint keys
        checkpoint["state_dict"] = self._remove_orig_mod_from_state_dict(
            checkpoint["state_dict"]
        )

        # Add EMA state dict
        # Remove '._orig_mod.' from EMA state dict keys
        ema_state_dict = self.ema.state_dict()
        ema_state_dict["shadow_params"] = self._remove_orig_mod_from_state_dict(
            ema_state_dict["shadow_params"]
        )
        checkpoint["ema"] = ema_state_dict

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Load EMA state dict
        self.load_ema_state_dict(checkpoint["ema"])

        if self.config.optimizer.final_training_stage:
            # Confidence-only training, so replace the structure-related
            # parameters to EMA's parameters.
            ema_state_dict = self.ema.state_dict()
            override_prefixes = tuple(self.frozen_modules)
            n = 0
            for k, v in ema_state_dict["shadow_params"].items():
                if k.startswith(override_prefixes):
                    n += 1
                    self.model.state_dict()[k].copy_(v)
            print(
                f"Override {n} parameters from EMA for final training stage "
                f"with prefixes {override_prefixes}."
            )

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):  # type: ignore
        """Override load_state_dict to handle EMA state dict."""
        # Remove '._orig_mod.' from state dict keys if present
        state_dict = self._remove_orig_mod_from_state_dict(state_dict)
        # Then, add '._orig_mod.' to state dict keys if required by the model
        state_dict = self._add_orig_mod_to_state_dict(state_dict, self.state_dict())
        # Remove 'model.' prefix from state dict keys if present
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        out = self.model.load_state_dict(state_dict, strict=strict)
        return out

    # === EMA === #
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):  # type: ignore
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        if self.ema.device != self.device:
            self.ema.to(self.device)
        self.ema.update(self.model)

    def _replace_ema_weights(self):
        if self.stored_weights is not None:
            # NOTE: To avoid accidentally replacing the weights multiple times,
            # we force only one replacement.
            raise ValueError("EMA weights have already been replaced.")
        self.stored_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if not name.startswith(self.submodules_to_ignore_for_ema)
        }
        ema_params = self.ema.shadow_params
        for name, param in self.model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name].data)

    def _restore_weights(self):
        if self.stored_weights is None:
            raise ValueError("No stored weights to restore.")
        for name, param in self.model.named_parameters():
            if name in self.stored_weights:
                param.data.copy_(self.stored_weights[name].data)
        self.stored_weights = None

    def on_validation_start(self):
        if self.ema.device != self.device:
            self.ema.to(self.device)
        if self.global_step >= self.config.optimizer.validate_with_ema_after_n_steps:
            self._replace_ema_weights()

    def on_validation_end(self) -> None:
        if self.stored_weights is not None:
            self._restore_weights()

    def load_ema_state_dict(self, state_dict: dict[str, Any]):
        """Load EMA state dict."""
        # Remove 'model.' prefix from EMA state dict keys if present.
        state_dict["shadow_params"] = {
            k.removeprefix("model."): v for k, v in state_dict["shadow_params"].items()
        }
        # Remove '._orig_mod.' from EMA state dict keys if present.
        state_dict["shadow_params"] = self._remove_orig_mod_from_state_dict(
            state_dict["shadow_params"]
        )
        # Add '._orig_mod.' to EMA state dict keys if required by the model.
        state_dict["shadow_params"] = self._add_orig_mod_to_state_dict(
            state_dict["shadow_params"], self.ema.shadow_params
        )
        assert self.ema.compatible(state_dict), (
            "EMA state dict is not compatible with the model."
        )
        self.ema.load_state_dict(state_dict, device=torch.device("cpu"))

    # === Helper functions === #
    def save_structure_and_metrics(
        self,
        ref_struct: RefStructure,
        pred_coords: torch.Tensor,
        metrics: dict[str, Any],
        prefix: str,
    ):
        """Save predicted and ground-truth structures as mmCIF files."""
        # TODO: save confidence too.
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
        self.writer.write_new_coords(ref_struct, pred_path, pred_coords.cpu().numpy())

    def save_trajectory(
        self,
        ref_struct: RefStructure,
        traj: torch.Tensor,
        save_dir: pathlib.Path,
        format: str = "cif",
    ):
        """Save predicted and ground-truth structures as mmCIF files."""
        name: str = ref_struct.id

        assert traj.ndim == 4, "Trajectory must be of shape (Nsample, Nframe, Natom, 3)"
        num_samples: int = traj.shape[0]

        # Remove padding atoms
        num_atoms: int = ref_struct.num_atoms
        traj: np.ndarray = traj[:, :, :num_atoms, :].detach().cpu().numpy()

        # Compute structure metrics
        for i in range(num_samples):
            # Save trajectory
            traj_i = traj[i]
            save_path = save_dir / f"{name}-sample-{i}-traj.{format}"
            self.writer.write_trajectory(ref_struct, traj_i, save_path, align=True)
