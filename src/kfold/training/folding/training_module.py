"""Define training modules for k-fold"""

import gc
import random
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import MeanMetric
from omegaconf import DictConfig

from kfold import constants as const
from kfold.config import to_dict
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold

from . import loss as loss_fn
from .loss import validation as val_fn
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

    # Whether to train structure and confidence modules
    train_structure_module: bool = True
    train_confidence_module: bool = False

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
        self.train_structure_module: bool = self.training_config.train_structure_module
        self.train_confidence_module: bool = self.training_config.train_confidence_module

        # Initialize model here
        self.model: KFold = KFold(self.global_config)

        # Freeze parts of the model if needed
        self.freeze_submodules()

        # Setup losses
        self.setup_losses()

        self.save_hyperparameters(to_dict(self.global_config))

        # (MingyeongShin) validation ----------------------------------
        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        
        for m in const.chain.OutType:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()

        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()
        # -------------------------------------------------------------

    def freeze_submodules(self):
        """Freeze submodules based on the training configuration."""
        # FIXME: (SeonghwanSeo) I did not test this function yet.
        # This is required when we train the confidence module only (Final-training-stage)
        if self.train_structure_module is False:
            self.model.trunk.eval()
            self.model.score_model.eval()
            self.model.trunk.requires_grad_(False)
            self.model.score_model.requires_grad_(False)
        if self.train_confidence_module is False:
            # TODO: freeze confidence module after they are implemented
            pass

    def setup_losses(self):
        """Setup loss functions for training"""
        loss_config = self.loss_config
        self.loss_weights = loss_config.weights

        if self.train_structure_module:
            # Distogram loss
            self.distogram_loss = loss_fn.distogram.DistogramLoss(
                **loss_config.distogram_loss
            )

            diffusion_loss_config = loss_config.diffusion_loss

            # Diffusion loss
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

        if self.train_confidence_module:
            raise NotImplementedError("Confidence loss not implemented yet.")

    def setup_metrics(self):
        """Setup metrics for validation"""
        pass

    def configure_optimizers(self):
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
                train_confidence_module=self.train_confidence_module,
                sample_structures=self.train_confidence_module,
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

    def training_step(self, batch: FoldingInput, batch_idx: int) -> torch.Tensor:
        training_config = self.training_config

        # Sample recycling steps
        num_cycles = random.randint(1, training_config.num_cycles)

        # Compute the forward pass
        out: dict[str, torch.Tensor] = self(
            f_input=batch,
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

        metrics = {f"train/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics)

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

            if self.train_confidence_module:
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
        try:
            out = self(
                f_input=batch,
                num_cycles=val_config.num_cycles,
                num_steps=val_config.num_steps,
                num_diffusion_samples=num_diffusion_samples,
                mode="validation",
            )

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        try:
            # Compute distogram LDDT --------------------------------
            boundaries = torch.linspace(2, 22.0, 63)
            lower = torch.tensor([1.0])
            upper = torch.tensor([22.0 + 5.0])
            exp_boundaries = torch.cat((lower, boundaries, upper))
            mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
                out["sample"]["distogram_logits"]
            )

            # Compute predicted dists
            preds = out["sample"]["distogram_logits"] # (B, T, T, num_bins)
            pred_softmax = torch.softmax(preds, dim=-1)
            pred_softmax = pred_softmax.argmax(dim=-1) # why argmax? # TODO: delete
            pred_softmax = torch.nn.functional.one_hot( 
                pred_softmax, num_classes=preds.shape[-1]
            ) # why argmax? # TODO: delete
            pred_dist = (pred_softmax * mid_points).sum(dim=-1)
            true_center = batch["disto_coords"]
            true_dists = torch.cdist(true_center, true_center)

            # Compute lddt's
            # batch["disto_mask"] = batch["disto_mask"] #? # TODO: delete
            disto_lddt_dict, disto_total_dict = val_fn.factored_token_lddt_dist_loss(
                f_input=batch,
                true_d=true_dists,
                pred_d=pred_dist,
            )
            # -------------------------------------------------------------

            # symmetry correction
            # TODO: fix get_true_coordinates function to use symmetry correction
            true_coords, rmsds, best_rmsds, true_coords_resolved_mask = (
                val_fn.get_true_coordinates(
                    batch=batch,
                    out=out,
                    num_diffusion_samples=num_diffusion_samples,
                    symmetry_correction=val_config.symmetry_correction,
                )
            )
            all_lddt_dict, all_total_dict = val_fn.factored_lddt_loss(
                f_input=batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords,
                pred_atom_coords=out["sample"]["coordinates"],
                num_diffusion_samples=num_diffusion_samples,
            )

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        B = true_coords.shape[0] // num_diffusion_samples

        if num_diffusion_samples > 1:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            
            for key in all_lddt_dict.keys():
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            
            complex_lddt /= complex_total + 1e-7
            best_complex_idx = complex_lddt.reshape(-1, num_diffusion_samples).argmax(dim=1) # (B, M)
            
            for key in all_lddt_dict:
                best_idx = all_lddt_dict[key].reshape(-1, num_diffusion_samples).argmax(dim=1)
                best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, num_diffusion_samples)[
                    torch.arange(B), best_idx
                ]
                best_total_dict[key] = all_total_dict[key].reshape(-1, num_diffusion_samples)[
                    torch.arange(B), best_idx
                ]
                best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, num_diffusion_samples)[
                    torch.arange(B), best_complex_idx
                ]
                best_complex_total_dict[key] = all_total_dict[key].reshape(
                    -1, num_diffusion_samples
                )[torch.arange(B), best_complex_idx]
        
        else:
            best_lddt_dict = all_lddt_dict # (B*M,)
            best_total_dict = all_total_dict
            best_complex_lddt_dict = all_lddt_dict
            best_complex_total_dict = all_total_dict

        # -------------------------------------------------------------
        # TODO: confidence module validation loss
        # -------------------------------------------------------------

        for m in const.chain.OutType:
            # 기존 코드 -------------------------------------------------------------
            # ligand_protein interface lddt의 경우 pocket feature가 2(POCKET)로 지정된 원자가 하나라도 있으면 pocket_ligand_protein이라는 특별 category에 기록    
            # boltz only (AF3 X)
            # TODO: Remain it. There would be pocket information somewhere in the future...
            # if m == "ligand_protein":
            #     if torch.any(
            #         batch["pocket_contact_type"][
            #             :, :, const.pocket.PocketContactType.POCKET
            #         ].bool()
            #     ):
            #         self.lddt["pocket_ligand_protein"].update(
            #             best_lddt_dict[m], best_total_dict[m]
            #      ㅣ   )
            #         self.disto_lddt["pocket_ligand_protein"].update(
            #             disto_lddt_dict[m], disto_total_dict[m]
            #         )
            #         self.complex_lddt["pocket_ligand_protein"].update(
            #             best_complex_lddt_dict[m], best_complex_total_dict[m]
            #         )
            #     else:
            #         self.lddt["ligand_protein"].update(
            #             best_lddt_dict[m], best_total_dict[m]
            #         )
            #         self.disto_lddt["ligand_protein"].update(
            #             disto_lddt_dict[m], disto_total_dict[m]
            #         )
            #         self.complex_lddt["ligand_protein"].update(
            #             best_complex_lddt_dict[m], best_complex_total_dict[m]
            #         )
            # else:
            #     self.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
            #     self.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
            #     self.complex_lddt[m].update(
            #         best_complex_lddt_dict[m], best_complex_total_dict[m]
            #     )
            # -------------------------------------------------------------
        
            self.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
            self.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
            self.complex_lddt[m].update(
                best_complex_lddt_dict[m], best_complex_total_dict[m]
            )
        
        self.rmsd.update(rmsds)
        self.best_rmsd.update(best_rmsds)

    def on_validation_epoch_end(self):
        # TODO: confidence module loss
        avg_lddt = {}
        avg_disto_lddt = {}
        avg_complex_lddt = {}

        # for m in const.out_types + ["pocket_ligand_protein"]: # when use "pocket_ligand_protein"
        for m in const.chain.OutType:   
            avg_lddt[m] = self.lddt[m].compute()
            avg_lddt[m] = 0.0 if torch.isnan(avg_lddt[m]) else avg_lddt[m].item()
            self.lddt[m].reset()
            self.log(f"val/lddt_{m}", avg_lddt[m], prog_bar=False, sync_dist=True)

            avg_disto_lddt[m] = self.disto_lddt[m].compute()
            avg_disto_lddt[m] = (
                0.0 if torch.isnan(avg_disto_lddt[m]) else avg_disto_lddt[m].item()
            )
            self.disto_lddt[m].reset()
            self.log(
                f"val/disto_lddt_{m}", avg_disto_lddt[m], prog_bar=False, sync_dist=True
            )

            avg_complex_lddt[m] = self.complex_lddt[m].compute()
            avg_complex_lddt[m] = (
                0.0 if torch.isnan(avg_complex_lddt[m]) else avg_complex_lddt[m].item()
            )
            self.complex_lddt[m].reset()
            self.log(
                f"val/complex_lddt_{m}",
                avg_complex_lddt[m],
                prog_bar=False,
                sync_dist=True,
            )

        # NOTE: 3 options for weights (boltz, AF3_Initial, AF3_Finetune)
        overall_disto_lddt = sum(
        avg_disto_lddt[m] * w for (m, w) in const.chain.OutTypeWeightsBoltz.items() 
        ) / sum(const.chain.OutTypeWeightsBoltz.values())
        self.log("val/disto_lddt", overall_disto_lddt, prog_bar=True, sync_dist=True)

        overall_lddt = sum(
            avg_lddt[m] * w for (m, w) in const.chain.OutTypeWeightsBoltz.items()
        ) / sum(const.chain.OutTypeWeightsBoltz.values())
        self.log("val/lddt", overall_lddt, prog_bar=True, sync_dist=True)

        overall_complex_lddt = sum(
            avg_complex_lddt[m] * w for (m, w) in const.chain.OutTypeWeightsBoltz.items()
        ) / sum(const.chain.OutTypeWeightsBoltz.values())
        self.log(
            "val/complex_lddt", overall_complex_lddt, prog_bar=True, sync_dist=True
        )

        # RMSD
        self.log("val/rmsd", self.rmsd.compute(), prog_bar=True, sync_dist=True)
        self.rmsd.reset()

        self.log(
            "val/best_rmsd", self.best_rmsd.compute(), prog_bar=True, sync_dist=True
        )
        self.best_rmsd.reset()


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

        if self.train_confidence_module:
            raise NotImplementedError(
                "Logging for confidence module not implemented yet."
            )
            # self.log(
            #     "train/grad_norm_confidence_module",
            #     gradient_norm(model.confidence_module),
            #     prog_bar=False,
            # )
            # self.log(
            #     "train/param_norm_confidence_module",
            #     parameter_norm(model.confidence_module),
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
