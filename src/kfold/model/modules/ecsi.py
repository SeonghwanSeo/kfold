"""Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI) for biomolecular
structure prediction.

Reference: "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553).

# Diffusion Path Design

Models the transition from source (apo) to target (holo) conformations.
- t=1 (Prior): Apo chain structures perturbed with 50 Å translational noise.
- t=0 (Data): Ground-truth assembled holo complexes.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives.utils import expand_dim
from kfold.utils.config import configurable
from kfold.utils.geometry.random_augment import (
    CenterRandomAugmentation,
    do_centering,
    random_rotations_torch,
)
from kfold.utils.geometry.rigid_align import get_rigid_transform_torch

from .score_model import DiffusionModule

_T = TypeVar("_T", float, torch.Tensor)


class BaseStructureModule(ABC):
    """High-level framework for structure generation."""

    @dataclasses.dataclass(kw_only=True)
    class Config: ...

    def __init__(self, cfg: Config, score_model: DiffusionModule):
        self.cfg = cfg
        self.score_model = score_model

    @abstractmethod
    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        """Get the sampling schedule."""

    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("training_step must be implemented in subclass")

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("forward_train must be implemented in subclass")

    @abstractmethod
    def sample_noise_level(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample training noise levels."""

    @abstractmethod
    def sample_train_input(
        self,
        f_input: FoldingInput,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample structure-module training inputs."""


# === Utility functions with type flexibility and numerical stability handling === #
def _clip(t: _T, eps: float = 1e-10) -> _T:
    return t.clip(min=eps) if isinstance(t, torch.Tensor) else max(t, eps)  # type: ignore


def _sqrt(t: _T) -> _T:
    return _clip(t, eps=0) ** 0.5  # type: ignore


def _log(t: _T) -> _T:
    log = torch.log if isinstance(t, torch.Tensor) else math.log
    return log(_clip(t, eps=1e-10))  # type: ignore


# === Custom rigid align function === #
def custom_rigid_align(
    coords: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    rotation_only: bool = False,
    output_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Torch implementation of weighted rigid alignment.

    `mask` selects the atoms that drive the fit; `output_mask` selects the atoms
    that survive in the result, and defaults to `mask`.
    """
    output_mask_was_none = output_mask is None
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)
    if not mask.any():
        if output_mask_was_none:
            return coords
        keep_mask = output_mask.bool().unsqueeze(-1)
        return coords.masked_fill(~keep_mask, 0.0)
    if output_mask is None:
        output_mask = mask

    original_dtype = coords.dtype
    fit_mask = mask.bool().unsqueeze(-1)
    keep_mask = output_mask.bool().unsqueeze(-1)

    with torch.autocast(device_type=coords.device.type, enabled=False):
        coords, target = coords.float(), target.float()

        # Fit on the alignment overlap only.
        fit_coords = coords.masked_fill(~fit_mask, 0.0)
        fit_target = target.masked_fill(~fit_mask, 0.0)
        weights = mask.to(dtype=fit_coords.dtype)
        RT, T = get_rigid_transform_torch(fit_coords, fit_target, weights)

        # Apply it to every atom the caller considers valid.
        aligned_coords = coords.masked_fill(~keep_mask, 0.0) @ RT
        if not rotation_only:
            aligned_coords = (aligned_coords + T.unsqueeze(-2)).masked_fill(
                ~keep_mask, 0.0
            )

    return aligned_coords.to(original_dtype)


# === Main ECSI module implementation === #
@dataclasses.dataclass
class SICoeffs:
    """Stochastic interpolant coefficient helper for ECSI."""

    gamma_max: float
    gamma_power: float
    eta: float

    def alpha(self, t: _T) -> _T:
        return 1.0 - t  # type: ignore

    def alpha_deriv(self, t: _T) -> _T:
        return -1.0  # type: ignore

    def beta(self, t: _T) -> _T:
        return t

    def beta_deriv(self, t: _T) -> _T:
        return 1.0  # type: ignore

    def gamma(self, t: _T) -> _T:
        t_pow = t**self.gamma_power
        return 0.5 * self.gamma_max * _sqrt(t_pow * (1 - t_pow))  # type: ignore

    def gamma_deriv(self, t: _T) -> _T:
        t_pow = t**self.gamma_power
        coeff = self.gamma_power * t ** (self.gamma_power - 1)
        denom = _sqrt(t_pow * (1 - t_pow))  # type: ignore
        return (self.gamma_max / 4) * coeff * (1 - 2 * t_pow) / _clip(denom)  # type: ignore

    def sigma_eff(self, t: _T) -> _T:
        r"""Return the analytical effective noise scale ``gamma(t) / alpha(t)``.

        This value selects a sigma-matched churn time only. It does not transform
        the coordinates supplied to the score model.
        """
        return self.gamma(t) / _clip(self.alpha(t))  # type: ignore

    # Compute \epsilon = \eta (\gamma \dot{\gamma} - \dot{\alpha}/\alpha \gamma^2)
    def eps(self, t: _T) -> _T:
        alpha, alpha_dot = self.alpha(t), self.alpha_deriv(t)
        gamma, gamma_dot = self.gamma(t), self.gamma_deriv(t)
        return self.eta * (  # type: ignore
            gamma * gamma_dot - alpha_dot / _clip(alpha) * gamma**2
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ECSISOARConfig:
    """Configuration for sampler-matched Exact-Markov ECSI SOAR training."""

    mode: str = "disabled"
    root_time_policy: str = "mirror_base_training_time"
    apply_rollout_churn: bool = False
    rollout_schedule_num_steps: int = 100
    rollout_schedule_step_count: float = 1.0
    auxiliary_samples_per_root: int = 4
    forward_retention_min: float = 0.5
    lambda_aux: float = 1.0
    mid_time_lower: float = 0.2
    high_time_split: float = 0.8
    mid_time_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("disabled", "model_sampler"):
            raise ValueError(
                f"Unknown ECSI SOAR mode {self.mode!r}; "
                "expected 'disabled' or 'model_sampler'."
            )
        if self.root_time_policy not in (
            "mirror_base_training_time",
            "mid_high_schedule_stratified",
        ):
            raise ValueError(
                f"Unknown ECSI SOAR root-time policy {self.root_time_policy!r}; "
                "expected 'mirror_base_training_time' or "
                "'mid_high_schedule_stratified'."
            )
        if self.rollout_schedule_num_steps <= 0:
            raise ValueError("ECSI SOAR rollout_schedule_num_steps must be positive.")
        if not 0.0 < self.rollout_schedule_step_count <= self.rollout_schedule_num_steps:
            raise ValueError(
                "ECSI SOAR rollout_schedule_step_count must be in "
                "(0, rollout_schedule_num_steps]."
            )
        if self.auxiliary_samples_per_root < 0:
            raise ValueError("ECSI SOAR auxiliary_samples_per_root must be non-negative.")
        if not 0.0 <= self.forward_retention_min <= 1.0:
            raise ValueError("ECSI SOAR forward_retention_min must lie in [0, 1].")
        if self.lambda_aux < 0.0:
            raise ValueError("ECSI SOAR lambda_aux must be non-negative.")
        if not 0.0 <= self.mid_time_lower < self.high_time_split <= 1.0:
            raise ValueError(
                "ECSI SOAR times must satisfy 0 <= mid_time_lower < high_time_split <= 1."
            )
        if not 0.0 <= self.mid_time_probability <= 1.0:
            raise ValueError("ECSI SOAR mid_time_probability must lie in [0, 1].")
        if self.mode != "disabled":
            if self.auxiliary_samples_per_root == 0:
                raise ValueError(
                    "Active ECSI SOAR requires auxiliary_samples_per_root > 0."
                )
            if self.lambda_aux == 0.0:
                raise ValueError("Active ECSI SOAR requires lambda_aux > 0.")

    @classmethod
    def from_mapping(
        cls,
        config: "Mapping[str, object] | ECSISOARConfig | None",
    ) -> "ECSISOARConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        known_fields = {field.name for field in dataclasses.fields(cls)}
        unknown_fields = set(config) - known_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown ECSI SOAR config fields: {sorted(unknown_fields)}."
            )
        return cls(**dict(config))  # type: ignore[arg-type]

    def num_auxiliary_samples(self, num_roots: int) -> int:
        if num_roots < 0:
            raise ValueError("ECSI SOAR num_roots must be non-negative.")
        return num_roots * self.auxiliary_samples_per_root


@configurable
class KFoldECSI(BaseStructureModule):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Linear route:
      \alpha_t=1-t and \beta_t=t
    - Tunable base gamma:
      \gamma_t^2=\gamma_{\max}^2/4 \cdot u(1-u), where u=t^{gamma_power}
    - Stochasticity control via \eta during sampling
    - Preconditioning adapted from DDBM using the shared base gamma

    Reference:
    - ECSI: Zhang et al., "Exploring the Design Space of Diffusion Bridge Models"

    NOTE: K-Fold uses a hybrid churn and ODE sampler specialized for structure
    prediction.
    """

    @dataclasses.dataclass(kw_only=True)
    class Config(BaseStructureModule.Config):
        """Configuration for the ECSI structure module.

        Parameters
        ----------
        # ECSI preconditioning parameters
        sigma_data : float
            Effective target-coordinate scale used in ECSI preconditioning.
        sigma_data_end : float
            Effective source-coordinate scale used in ECSI preconditioning.
        cov_xy : float
            Cross-covariance term between source and target coordinates used by
            the bridge preconditioning formulas.

        # ECSI coefficients
        time_min: float
            Minimum time value for training/inference.
        time_max: float
            Maximum time value for training/inference.
        gamma_max : float
            Shared base bridge maximum used by `gamma(t)`.
        gamma_power : float
            Exponent controlling the base gamma schedule while route coefficients
            remain fixed as `alpha_t = 1 - t` and `beta_t = t`.
        eta : float
            Stochasticity control parameter for ECSI sampling.
        # Inference sampling parameters
        align_x_0_hat_to_x_t : bool
            Whether to rigidly align the predicted x_0_hat to x_t at each sampling step.
        sampler_mode : str
            Update mode: ``sde`` for the default hybrid or ``ode`` for rollback.
        sampler_ode_type : str
            High-time ECSI/SI update family. The default is ``ecsi``.
        sampler_step_scale : float
            Multiplier for deterministic ODE update displacement.
        sampler_switch_time : float | None
            Reverse-time boundary for the fixed low-time SI-ODE phase. Values
            at or below the boundary use SI ODE; ``None`` disables the phase.
        sampler_sde_atom_classes : tuple[str, ...]
            Explicit classes that receive the sampler profile. ``("all",)``
            preserves the global sampler, while ``()`` makes every atom use SI
            ODE. Molecular classes select their matching token atoms only; they
            do not expand over covalent bonds. ``protein`` includes peptide
            chains; ``peptide`` selects protein chains with fewer than 16
            residues.
        churn_factor : float
            Fractional effective-noise inflation ``chi`` before the fixed
            high-time ramp. This is the only numerical sampler knob.
        churn_max_multiplier : float
            Maximum high-time multiplier applied to ``churn_factor``.
        churn_end_time : float
            End of the forward-pinned churn window.
        churn_max_time : float | None
            Exclusive upper churn bound. ``None`` resolves to ``time_max``.
        churn_step_fraction : float
            Fraction of sampling steps in the churn phase.
        churn_step_power : float
            Churn-phase schedule exponent.
        ode_step_power : float
            ODE-phase schedule exponent.
        stepwarp_power : float
            High-churn knot redistribution exponent.

        # Training time scheduling
        train_time_schedule : str
            Training-time sampling schedule. Supported values are "logistic" and
            "uniform".
        train_time_schedule_params : tuple[float, float]
            A tuple of (mu, std) for the logistic time sampling schedule.
            Time values are sampled from sigmoid(N(mu, std)), then scaled to
            [time_min, time_max].
        train_x_0_perturb_time_min : float
            Apply chain-rigid x0 perturbation only strictly above this time.
        train_x_0_perturb_prob : float
            Probability of perturbing an eligible high-time training sample.
        train_x_0_perturb_translation_std : float
            Per-axis standard deviation of each chain's Gaussian translation,
            in Angstrom. Applied chains also receive a random SO(3) rotation.
        """

        sigma_data: float = 16.0
        sigma_data_end: float = 48.0  # 48 translations
        cov_xy: float = 300.0

        time_min: float = 1e-8
        time_max: float = 0.9999
        gamma_max: float = 24.0
        gamma_power: float = 1.0
        eta: float = 1.0

        # Inference sampling
        align_x_0_hat_to_x_t: bool = True
        sampler_mode: str = "sde"
        sampler_ode_type: str = "ecsi"
        sampler_step_scale: float = 1.0
        sampler_switch_time: float | None = 0.1
        sampler_sde_atom_classes: tuple[str, ...] = ("all",)
        churn_factor: float = 0.1
        churn_max_multiplier: float = 4.0
        churn_end_time: float = 0.5
        churn_max_time: float | None = None
        churn_step_fraction: float = 0.4
        churn_step_power: float = 1.0
        ode_step_power: float = 2.0
        stepwarp_power: float = 0.5

        # Train time scheduling
        train_time_schedule: str = "logistic"
        train_time_schedule_params: tuple[float, float] = (-2.15, 2.25)

        # Optional high-time chain-wise x0 perturbation for base bridge samples.
        train_x_0_perturb_time_min: float = 0.7
        train_x_0_perturb_prob: float = 0.0
        train_x_0_perturb_translation_std: float = 0.0

    def __init__(self, cfg: Config, score_model: DiffusionModule):
        """Initialize the ECSI module.

        The constructor copies the high-level config fields onto runtime
        attributes, then immediately validates and normalizes the nested
        sampling config so downstream code can assume a runtime-ready ECSI
        configuration.
        """
        super().__init__(cfg, score_model)
        self.cfg = cfg
        self.score_model: DiffusionModule = score_model

        # ECSI preconditioning parameters
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy

        # ECSI coefficients
        self.coeff = SICoeffs(cfg.gamma_max, cfg.gamma_power, cfg.eta)
        self.time_min: float = cfg.time_min
        self.time_max: float = cfg.time_max

        # Train time scheduling
        if cfg.train_time_schedule not in {"logistic", "uniform"}:
            raise ValueError(
                "Unknown ECSI train_time_schedule: "
                f"{cfg.train_time_schedule!r}. Expected 'logistic' or 'uniform'."
            )
        self.train_time_schedule: str = cfg.train_time_schedule
        self.train_time_schedule_params: tuple[float, float] = (
            cfg.train_time_schedule_params
        )
        if not self.time_min <= cfg.train_x_0_perturb_time_min <= self.time_max:
            raise ValueError(
                "ECSI train_x_0_perturb_time_min must lie in the trained time support."
            )
        if not 0.0 <= cfg.train_x_0_perturb_prob <= 1.0:
            raise ValueError("ECSI train_x_0_perturb_prob must lie in [0, 1].")
        if cfg.train_x_0_perturb_translation_std < 0.0:
            raise ValueError(
                "ECSI train_x_0_perturb_translation_std must be non-negative."
            )
        self.train_x_0_perturb_time_min = cfg.train_x_0_perturb_time_min
        self.train_x_0_perturb_prob = cfg.train_x_0_perturb_prob
        self.train_x_0_perturb_translation_std = cfg.train_x_0_perturb_translation_std

        # Inference time sampling
        self.align_x_0_hat_to_x_t: bool = cfg.align_x_0_hat_to_x_t
        if cfg.sampler_mode not in {"ode", "sde"}:
            raise ValueError(f"Unknown ECSI sampler_mode: {cfg.sampler_mode}")
        if cfg.sampler_ode_type not in {"si", "ecsi"}:
            raise ValueError(f"Unknown ECSI sampler_ode_type: {cfg.sampler_ode_type}")
        if cfg.sampler_step_scale <= 0:
            raise ValueError("ECSI sampler_step_scale must be positive.")
        if cfg.sampler_switch_time is not None and not (
            cfg.time_min <= cfg.sampler_switch_time <= cfg.time_max
        ):
            raise ValueError(
                "ECSI sampler_switch_time must lie within the trained time support."
            )
        if cfg.gamma_power <= 0:
            raise ValueError("ECSI gamma_power must be positive.")
        if not 0.0 <= cfg.churn_end_time <= cfg.time_max:
            raise ValueError(
                "ECSI churn_end_time must lie within the trained time support."
            )
        churn_max_time = (
            cfg.time_max if cfg.churn_max_time is None else cfg.churn_max_time
        )
        if churn_max_time > cfg.time_max:
            raise ValueError(
                f"ECSI churn_max_time ({churn_max_time}) must not exceed time_max "
                f"({cfg.time_max}); the score model is untrained beyond time_max."
            )
        if churn_max_time < cfg.churn_end_time:
            raise ValueError(
                f"ECSI churn_max_time ({churn_max_time}) must not be below "
                f"churn_end_time ({cfg.churn_end_time})."
            )
        if any(
            value < 0
            for value in (
                cfg.eta,
                cfg.churn_factor,
                cfg.churn_max_multiplier,
                cfg.stepwarp_power,
            )
        ):
            raise ValueError("ECSI sampler parameters must be non-negative.")
        self.sampler_mode: str = cfg.sampler_mode
        self.sampler_ode_type: str = cfg.sampler_ode_type
        self.sampler_step_scale: float = cfg.sampler_step_scale
        self.sampler_switch_time: float | None = cfg.sampler_switch_time
        self.sampler_sde_atom_classes = self._normalize_sde_atom_classes(
            cfg.sampler_sde_atom_classes
        )
        self.churn_factor: float = cfg.churn_factor
        self.churn_max_multiplier: float = cfg.churn_max_multiplier
        self.churn_end_time: float = cfg.churn_end_time
        self.churn_max_time: float = churn_max_time
        self.churn_step_fraction: float = cfg.churn_step_fraction
        self.churn_step_power: float = cfg.churn_step_power
        self.ode_step_power: float = cfg.ode_step_power
        self.stepwarp_power: float = cfg.stepwarp_power

        # NOTE: centering should be disabled.
        self.random_augmentation = CenterRandomAugmentation()

    @staticmethod
    def _normalize_sde_atom_classes(
        atom_classes: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate the explicit classes for class-routed sampler updates."""
        if atom_classes is None:
            raise ValueError(
                "ECSI sampler_sde_atom_classes must be explicit. Use ('all',) "
                "for the global sampler or () for SI ODE."
            )
        if isinstance(atom_classes, str):
            raise ValueError(
                "ECSI sampler_sde_atom_classes must be a sequence, not a string."
            )

        normalized = tuple(atom_classes)
        if any(not isinstance(atom_class, str) for atom_class in normalized):
            raise ValueError("ECSI sampler_sde_atom_classes must contain strings.")

        valid_classes = {"all", "protein", "ligand", "peptide", "rna", "dna"}
        unknown = sorted(set(normalized) - valid_classes)
        if unknown:
            raise ValueError(
                "ECSI sampler_sde_atom_classes contains unsupported classes: "
                f"{unknown}. Expected a subset of {sorted(valid_classes)}."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("ECSI sampler_sde_atom_classes must not contain duplicates.")
        if "all" in normalized and len(normalized) != 1:
            raise ValueError("ECSI sampler_sde_atom_classes 'all' must be used alone.")
        return normalized

    # === Bridge Preconditioning Coefficients === #
    def _get_bridge_scalings(self, t: _T) -> tuple[_T, _T, _T]:
        """Compute bridge diffusion scalings for ECSI.

        Adapted from DDBM formulation using ECSI's decoupled parameterization.

        Parameters
        ----------
        t : float | torch.Tensor
            Time values, in range [0, 1].

        Returns
        -------
        c_in : float | torch.Tensor
            Input scaling coefficient.
        c_skip : float | torch.Tensor
            Skip connection coefficient.
        c_out : float | torch.Tensor
            Output scaling coefficient.
        """
        C = self.coeff
        alpha_t, beta_t, gamma_t = C.alpha(t), C.beta(t), C.gamma(t)
        sigma_0, sigma_T, sigma_0T = self.sigma_data, self.sigma_data_end, self.cov_xy

        c_in = 1 / _clip(
            _sqrt(
                (alpha_t * sigma_0) ** 2
                + (beta_t * sigma_T) ** 2
                + 2 * alpha_t * beta_t * sigma_0T
                + gamma_t**2
            )
        )
        c_skip = (alpha_t * sigma_0**2 + beta_t * sigma_0T) * (c_in**2)
        c_out = (
            _sqrt(
                (beta_t * sigma_0 * sigma_T) ** 2
                - (beta_t * sigma_0T) ** 2
                + gamma_t**2 * sigma_0**2
            )
            * c_in
        )
        if isinstance(c_out, torch.Tensor):
            c_in, c_skip, c_out = c_in.float(), c_skip.float(), c_out.float()
        return c_in, c_skip, c_out

    def c_in(self, t: _T) -> _T:
        """Input scaling coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[0]

    def c_skip(self, t: _T) -> _T:
        """Skip connection coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[1]

    def c_out(self, t: _T) -> _T:
        """Output scaling coefficient for ECSI preconditioning."""
        return self._get_bridge_scalings(t)[2]

    def c_noise(self, t: _T) -> _T:
        """Noise level conditioning coefficient."""
        return 0.25 * _log(t)

    def loss_weights(self, t: torch.Tensor) -> torch.Tensor:
        """ECSI training loss weights"""
        return 1 / self.c_out(t).pow(2).clamp(min=1e-8)

    # ============================================================
    # For training
    # ============================================================
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
        soar_config: Mapping[str, object] | ECSISOARConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform base ECSI training plus optional Exact-Markov SOAR."""
        soar = ECSISOARConfig.from_mapping(soar_config)
        with torch.autocast(f_input.device.type, enabled=False):
            train_input = self.sample_train_input(f_input, diffusion_batch_size)

        t = train_input["t"]  # [B, N]
        x_0 = train_input["x_0"]  # [B, N, Natom, 3]
        x_t = train_input["x_t"]  # [B, N, Natom, 3]
        x_T = train_input["x_T"]  # [B, N, Natom, 3]
        atom_mask = train_input["atom_mask"]

        x_0_hat = self._forward_train(
            x_t=x_t,  # [B, N, Natom, 3]
            t=t,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            z=z,  # [B, Lt, Lt, c_z]
            x_T=x_T,  # [B, N, Natom, 3]
            atom_mask=atom_mask,  # [B, Natom]
        )  # [B, N, Natom, 3]

        loss_weights = self.loss_weights(t)  # [B, N]

        output = {
            "t": t,
            "x_t": x_t,
            "x_T": x_T,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }
        for name in (
            "time_eligible_mask",
            "eligible_mask",
            "requested_mask",
            "applied_mask",
            "x_0_rmsd",
            "x_t_rmsd",
            "resolved_chain_count",
        ):
            output[f"x_0_perturb_{name}"] = train_input[f"x_0_perturb_{name}"]
        if soar.mode == "disabled":
            return output

        auxiliary = self._build_soar_training_batch(
            f_input=f_input,
            s_inputs=s_inputs,
            z=z,
            config=soar,
            x_0=x_0,
            x_T=x_T,
            atom_mask=atom_mask,
            bridge_noise=train_input["noise"],
            base_t0=t,
        )
        output["t"] = torch.cat((t, auxiliary["t_aux"]), dim=1)
        output["x_t"] = torch.cat((x_t, auxiliary["x_aux"]), dim=1)
        output["x_T"] = torch.cat((x_T, auxiliary["x_T"]), dim=1)
        output["x_0_hat"] = torch.cat((x_0_hat, auxiliary["x_0_hat_aux"]), dim=1)
        output["x_gt"] = torch.cat((x_0, auxiliary["x_0"]), dim=1)
        output["loss_weights"] = torch.cat(
            (loss_weights, self.loss_weights(auxiliary["t_aux"])), dim=1
        )

        batch_size = t.shape[0]
        auxiliary_mask = torch.ones(
            (batch_size, auxiliary["t_aux"].shape[1]),
            dtype=torch.bool,
            device=t.device,
        )
        output["soar_auxiliary_mask"] = torch.cat(
            (torch.zeros_like(t, dtype=torch.bool), auxiliary_mask), dim=1
        )
        output["soar_supervision_weights"] = torch.cat(
            (torch.ones_like(t), auxiliary["supervision_weights"]), dim=1
        )
        for name in (
            "base_t0",
            "t0",
            "t_call",
            "t1",
            "t2",
            "root_branch_count",
            "forward_retention",
            "forward_variance",
            "endpoint_error_rmsd",
            "model_to_oracle_sampler_rmsd",
            "oracle_sampler_to_exact_ecsi_rmsd",
            "model_sampler_to_exact_ecsi_rmsd",
        ):
            output[f"soar_{name}"] = auxiliary[name]
        return output

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        x_T: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the score model with ECSI preconditioning.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t : torch.Tensor
            Time values. Shape (B, N).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, L, c_s).
        z : torch.Tensor
            Trunk pairwise embeddings. Shape (B, L, L, c_z).
        x_T : torch.Tensor | None
            Source (apo) coordinates x_T. Shape (B, N, L, 3).
        atom_mask : torch.Tensor | None
            Atoms the coordinate stack may attend to. Shape (B, L).

        Returns
        -------
        x_0_hat : torch.Tensor
            Denoised atom coordinates. Shape (B, N, Natom, 3).
        """
        assert x_T is not None, "x_T must be provided for ECSI"
        c_in, c_skip, c_out = self._get_bridge_scalings(t)  # [B, N]
        c_noise = self.c_noise(t)  # [B, N]

        # Input preconditioning: r_noisy = c_in * x_t, r_T = x_T / sigma_data_end
        r_noisy = c_in[:, :, None, None] * x_t  # [B, N, Natom, 3]

        # End-point conditioning: r_T = x_T / sigma_data_end
        r_T = x_T / self.sigma_data_end  # [B, N, Natom, 3]
        r_noisy = torch.cat([r_noisy, r_T], dim=-1)

        # Call score model
        r_update = self.score_model.train_step(
            f_input=f_input,
            r_noisy=r_noisy,  # [B, N, Natom, 6]
            c_noise=c_noise,  # [B, N]
            s_inputs=s_inputs,  # [B, Lt, c_s]
            z=z,  # [B, Lt, Lt, c_z]
            atom_mask=atom_mask,  # [B, Natom]
        )

        # Output preconditioning: \hat{x}_0 = c_{skip} * x_t + c_{out} * F_\theta
        x_0_hat = c_skip[..., None, None] * x_t + c_out[..., None, None] * r_update
        return x_0_hat

    def sample_noise_level(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        r"""Sample time values for training.

        Returns samples in [sampling_time_min, sampling_time_max] which represents
        the time interval [t_{min}, t_{max}] \subset [0, 1].

        Returns
        -------
        t : torch.Tensor
            Time values. Shape (B, N).
        """
        if self.train_time_schedule == "uniform":
            t = torch.rand(shape, device=device)
        else:
            mu, std = self.train_time_schedule_params
            z = torch.randn(shape, device=device)
            x = mu + std * z
            t = torch.sigmoid(x)

        # Scale to [sampling_time_min, sampling_time_max]
        t = self.time_min + (self.time_max - self.time_min) * t
        return t

    def _sample_x_0_perturb_masks(
        self,
        *,
        t: torch.Tensor,
        resolved_chain_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Select high-time perturbations without touching disabled RNG."""
        time_eligible = t > self.train_x_0_perturb_time_min
        multichain = resolved_chain_count[:, None] > 1
        eligible = time_eligible & multichain
        if self.train_x_0_perturb_prob == 0.0:
            requested = torch.zeros_like(time_eligible)
        else:
            requested = time_eligible & (torch.rand_like(t) < self.train_x_0_perturb_prob)
        applied = requested & multichain
        return {
            "time_eligible": time_eligible,
            "eligible": eligible,
            "requested": requested,
            "applied": applied,
        }

    def _perturb_x_0(
        self,
        *,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        x_0_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the original high-time chain-rigid x0 corruption contract."""
        batch_size, num_samples, num_atoms, _ = x_0.shape
        if self.train_x_0_perturb_prob == 0.0:
            time_eligible = t > self.train_x_0_perturb_time_min
            false_mask = torch.zeros_like(time_eligible)
            zero = torch.zeros_like(t)
            return {
                "x_0_bridge": x_0,
                "time_eligible_mask": time_eligible,
                "eligible_mask": false_mask,
                "requested_mask": false_mask,
                "applied_mask": false_mask,
                "x_0_rmsd": zero,
                "resolved_chain_count": torch.zeros_like(t, dtype=torch.long),
            }
        num_chains = f_input.chain.asym_id.shape[1]
        token_index = f_input.atom.token_index.clamp(
            min=0, max=f_input.token.asym_id.shape[-1] - 1
        )
        atom_asym_id = torch.gather(f_input.token.asym_id, 1, token_index)
        resolved_mask = x_0_mask[:, 0].bool()
        atom_in_chain = atom_asym_id[:, :, None] == f_input.chain.asym_id[:, None, :]
        atom_in_chain &= f_input.chain.pad_mask[:, None, :]
        atom_chain_index = atom_in_chain.to(torch.int64).argmax(dim=-1)
        atom_is_valid = resolved_mask & atom_in_chain.any(dim=-1)
        chain_has_atoms = (atom_in_chain & resolved_mask[:, :, None]).any(dim=1)
        resolved_chain_count = chain_has_atoms.sum(dim=-1)
        selection = self._sample_x_0_perturb_masks(
            t=t, resolved_chain_count=resolved_chain_count
        )
        applied = selection["applied"]
        chain_index_xyz = atom_chain_index[:, None, :, None].expand(
            batch_size, num_samples, num_atoms, 3
        )
        chain_sums = torch.zeros(
            (batch_size, num_samples, num_chains, 3),
            dtype=x_0.dtype,
            device=x_0.device,
        )
        chain_sums.scatter_add_(
            dim=2,
            index=chain_index_xyz,
            src=x_0 * atom_is_valid[:, None, :, None],
        )
        chain_counts = (atom_in_chain & resolved_mask[:, :, None]).sum(dim=1)
        chain_centers = chain_sums / chain_counts[:, None, :, None].clamp_min(1)
        chain_rotations = random_rotations_torch(
            (batch_size, num_samples, num_chains),
            dtype=x_0.dtype,
            device=x_0.device,
        )
        chain_translations = torch.randn_like(chain_centers)
        chain_translations *= self.train_x_0_perturb_translation_std
        atom_centers = torch.gather(chain_centers, dim=2, index=chain_index_xyz)
        atom_translations = torch.gather(chain_translations, dim=2, index=chain_index_xyz)
        rotation_index = atom_chain_index[:, None, :, None, None].expand(
            batch_size, num_samples, num_atoms, 3, 3
        )
        atom_rotations = torch.gather(chain_rotations, dim=2, index=rotation_index)
        transformed = (
            torch.einsum("bnad,bnads->bnas", x_0 - atom_centers, atom_rotations)
            + atom_centers
            + atom_translations
        )
        apply_atom_mask = applied[..., None, None] & atom_is_valid[:, None, :, None]
        x_0_perturbed = torch.where(apply_atom_mask, transformed, x_0)
        expanded_mask = x_0_mask.expand(-1, num_samples, -1)
        aligned = custom_rigid_align(
            x_0_perturbed,
            x_T,
            expanded_mask,
            rotation_only=True,
        )
        aligned = do_centering(aligned, mask=expanded_mask)
        apply_mask = applied[..., None, None] & expanded_mask[..., None]
        x_0_bridge = torch.where(apply_mask, aligned, x_0)
        x_0_rmsd = self._masked_coordinate_rmsd(x_0_bridge - x_0, expanded_mask)
        return {
            "x_0_bridge": x_0_bridge,
            "time_eligible_mask": selection["time_eligible"],
            "eligible_mask": selection["eligible"],
            "requested_mask": selection["requested"],
            "applied_mask": applied,
            "x_0_rmsd": x_0_rmsd,
            "resolved_chain_count": resolved_chain_count[:, None].expand_as(t),
        }

    def sample_train_input(
        self,
        f_input: FoldingInput,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample training inputs for the structure module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        diffusion_batch_size : int
            The number of samples to generate for training.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the x_0, x_t, and related representations.
        """
        batch_size = f_input.batch_size
        num_samples = diffusion_batch_size
        device = f_input.device
        t = self.sample_noise_level((batch_size, num_samples), device)

        # === Prepare x_0 and x_T === #
        x_holo = f_input.atom.label_coords  # [B, Natom, 3]
        holo_mask = f_input.atom.resolved_mask  # [B, Natom]
        x_apo = f_input.atom.prior_coords.permute(0, 2, 1, 3)  # [B, Nprior, Natom, 3]
        apo_mask = f_input.atom.pad_mask  # [B, Natom]

        # Repeat holo coords
        x_0 = expand_dim(x_holo, num_samples, dim=-3).clone()  # [B, N, Natom, 3]
        x_0_mask = holo_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        num_prior = x_apo.shape[-3]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = x_apo[:, idx, :, :]  # [B, N, Natom, 3]

        # Atoms the coordinate stack may see. Unresolved atoms have no holo target,
        # so x_0 is 0 there and any interpolant through it is meaningless. Hiding
        # them keeps that meaningless coordinate out of every geometric operation
        # below and out of the score model's attention. Inference has no
        # `resolved_mask`, so it keeps using the full `pad_mask`.
        train_mask = apo_mask & holo_mask  # [B, Natom]
        x_T_mask = train_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Apply centering/coordinate augmentation
        x_0 = self.random_augmentation(x_0, mask=x_0_mask)

        # Rotate x_T toward x_0 while preserving the prior translation distribution.
        x_T = custom_rigid_align(
            x_T, x_0, x_0_mask, rotation_only=True, output_mask=x_T_mask
        )

        # Perturb only the endpoint used by the base bridge. SOAR reconstructs
        # its roots from the clean x_0 returned below.
        perturb = self._perturb_x_0(
            x_0=x_0,
            x_T=x_T,
            t=t,
            f_input=f_input,
            x_0_mask=x_0_mask,
        )

        # ECSI interpolation with atom-wise shared bridge noise.
        noise = torch.randn_like(x_0).masked_fill_(~x_T_mask[..., None], 0.0)
        x_t_clean = self._interpolate_bridge(x_0, x_T, noise, t)
        x_t = self._interpolate_bridge(perturb["x_0_bridge"], x_T, noise, t)
        expanded_mask = x_T_mask.expand(-1, num_samples, -1)
        x_t_rmsd = self._masked_coordinate_rmsd(x_t - x_t_clean, expanded_mask)

        # Zero the hidden atoms as well as the padding, so that a code path which
        # forgets `train_mask` sees the origin rather than a prior-scale offset.
        x_0.masked_fill_(~x_0_mask[..., None], 0.0)
        x_T.masked_fill_(~x_T_mask[..., None], 0.0)
        x_t.masked_fill_(~x_T_mask[..., None], 0.0)

        return {
            "t": t,
            "x_0": x_0,
            "x_t": x_t,
            "x_T": x_T,
            "atom_mask": train_mask,
            "noise": noise,
            "x_0_perturb_time_eligible_mask": perturb["time_eligible_mask"],
            "x_0_perturb_eligible_mask": perturb["eligible_mask"],
            "x_0_perturb_requested_mask": perturb["requested_mask"],
            "x_0_perturb_applied_mask": perturb["applied_mask"],
            "x_0_perturb_x_0_rmsd": perturb["x_0_rmsd"],
            "x_0_perturb_x_t_rmsd": x_t_rmsd,
            "x_0_perturb_resolved_chain_count": perturb["resolved_chain_count"],
        }

    def _interpolate_bridge(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the ECSI bridge with caller-supplied shared noise."""
        expanded_t = t[..., None, None]
        return (
            self.coeff.alpha(expanded_t) * x_0
            + self.coeff.beta(expanded_t) * x_T
            + self.coeff.gamma(expanded_t) * noise
        )

    def _build_soar_training_batch(
        self,
        *,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        config: ECSISOARConfig,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        atom_mask: torch.Tensor,
        bridge_noise: torch.Tensor,
        base_t0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build detached Exact-Markov auxiliaries from model-generated states."""
        if x_0.shape != x_T.shape or x_0.shape != bridge_noise.shape:
            raise ValueError("SOAR endpoint and bridge-noise shapes must match.")
        if base_t0.shape != x_0.shape[:2]:
            raise ValueError("SOAR base times must match the base sample dimensions.")
        if atom_mask.shape != (x_0.shape[0], x_0.shape[-2]):
            raise ValueError("SOAR atom_mask must have shape [B, Natom].")

        batch_size, num_roots = base_t0.shape
        if num_roots == 0:
            raise ValueError("Active ECSI SOAR requires at least one base root.")
        num_auxiliary = config.num_auxiliary_samples(num_roots)
        root_mask = atom_mask.unsqueeze(1).expand(-1, num_roots, -1)

        with torch.no_grad(), torch.autocast(f_input.device.type, enabled=False):
            t0 = self.construct_soar_root_times(base_t0=base_t0, config=config)
            t1, t2 = self.sample_soar_auxiliary_times(t0=t0, config=config)
            x_t0 = self._interpolate_bridge(x_0, x_T, bridge_noise, t0)
            x_t0 = x_t0.masked_fill(~root_mask[..., None], 0.0)
            x_t1_exact = self._interpolate_bridge(x_0, x_T, bridge_noise, t1)
            x_t1_exact = x_t1_exact.masked_fill(~root_mask[..., None], 0.0)
            x_call, t_call = self._prepare_soar_rollout_call(
                config=config,
                x_t=x_t0,
                x_T=x_T,
                mask=root_mask,
                t=t0,
            )

        with torch.no_grad():
            x_0_hat_t0 = self._forward_train(
                x_t=x_call,
                t=t_call,
                f_input=f_input,
                s_inputs=s_inputs,
                z=z,
                x_T=x_T,
                atom_mask=atom_mask,
            ).detach()

        with torch.no_grad(), torch.autocast(f_input.device.type, enabled=False):
            model_endpoint = self._postprocess_soar_endpoint(
                endpoint=x_0_hat_t0,
                x_t=x_call,
                mask=root_mask,
            )
            oracle_endpoint = self._postprocess_soar_endpoint(
                endpoint=x_0,
                x_t=x_call,
                mask=root_mask,
            )
            shared_update_noise = torch.randn_like(x_call).masked_fill_(
                ~root_mask[..., None], 0.0
            )
            x_t1_model = self._apply_soar_sampler_update(
                f_input=f_input,
                x_t=x_call,
                x_0_hat=model_endpoint,
                x_T=x_T,
                mask=root_mask,
                t=t_call,
                t_next=t1,
                noise=shared_update_noise,
            )
            x_t1_oracle = self._apply_soar_sampler_update(
                f_input=f_input,
                x_t=x_call,
                x_0_hat=oracle_endpoint,
                x_T=x_T,
                mask=root_mask,
                t=t_call,
                t_next=t1,
                noise=shared_update_noise,
            )
            model_transition = self._exact_ecsi_forward_transition(
                x_t1=x_t1_model,
                x_T=x_T,
                mask=root_mask,
                t1=t1,
                t2=t2,
            )
            x_aux_branched = model_transition["x_t2"]
            x_aux = x_aux_branched.reshape(
                batch_size, num_auxiliary, *x_0.shape[-2:]
            ).detach()
            x_T_aux = (
                x_T.unsqueeze(2)
                .expand(-1, -1, config.auxiliary_samples_per_root, -1, -1)
                .reshape_as(x_aux)
            )
            x_0_aux = (
                x_0.unsqueeze(2)
                .expand(-1, -1, config.auxiliary_samples_per_root, -1, -1)
                .reshape_as(x_aux)
            )
            t_aux = t2.reshape(batch_size, num_auxiliary)

        x_0_hat_aux = self._forward_train(
            x_t=x_aux,
            t=t_aux,
            f_input=f_input,
            s_inputs=s_inputs,
            z=z,
            x_T=x_T_aux,
            atom_mask=atom_mask,
        )

        endpoint_error_rmsd = self._masked_coordinate_rmsd(
            model_endpoint - oracle_endpoint, root_mask
        )
        model_to_oracle_rmsd = self._masked_coordinate_rmsd(
            x_t1_model - x_t1_oracle, root_mask
        )
        oracle_to_exact_rmsd = self._masked_coordinate_rmsd(
            x_t1_oracle - x_t1_exact, root_mask
        )
        model_to_exact_rmsd = self._masked_coordinate_rmsd(
            x_t1_model - x_t1_exact, root_mask
        )
        supervision_weights = torch.full(
            (batch_size, num_auxiliary),
            config.lambda_aux,
            dtype=t_aux.dtype,
            device=t_aux.device,
        )
        root_branch_count = torch.full(
            (batch_size, num_roots),
            config.auxiliary_samples_per_root,
            dtype=torch.long,
            device=t_aux.device,
        )
        return {
            "base_t0": base_t0.detach(),
            "t0": t0.detach(),
            "t_call": t_call.detach(),
            "t1": t1.detach(),
            "t2": t2.detach(),
            "root_branch_count": root_branch_count,
            "supervision_weights": supervision_weights,
            "t_aux": t_aux,
            "x_0": x_0_aux,
            "x_T": x_T_aux,
            "x_aux": x_aux,
            "x_0_hat_aux": x_0_hat_aux,
            "forward_retention": model_transition["retention"].detach(),
            "forward_variance": model_transition["variance"].detach(),
            "endpoint_error_rmsd": endpoint_error_rmsd.detach(),
            "model_to_oracle_sampler_rmsd": model_to_oracle_rmsd.detach(),
            "oracle_sampler_to_exact_ecsi_rmsd": oracle_to_exact_rmsd.detach(),
            "model_sampler_to_exact_ecsi_rmsd": model_to_exact_rmsd.detach(),
        }

    def construct_soar_root_times(
        self, *, base_t0: torch.Tensor, config: ECSISOARConfig
    ) -> torch.Tensor:
        """Construct independently controlled SOAR rollout root times."""
        if base_t0.ndim != 2:
            raise ValueError("SOAR base time must have shape [B, Nroot].")
        if config.root_time_policy == "mirror_base_training_time":
            return (self.time_min + self.time_max - base_t0).clamp(
                min=self.time_min, max=self.time_max
            )

        schedule = torch.tensor(
            self.get_effective_sampling_schedule(config.rollout_schedule_num_steps),
            dtype=base_t0.dtype,
            device=base_t0.device,
        )
        mid_roots = self._sample_soar_schedule_cell_band(
            shape=base_t0.shape,
            schedule=schedule,
            lower=max(self.time_min, config.mid_time_lower),
            upper=min(self.time_max, config.high_time_split),
        )
        high_roots = self._sample_soar_schedule_cell_band(
            shape=base_t0.shape,
            schedule=schedule,
            lower=max(self.time_min, config.high_time_split),
            upper=self.time_max,
        )
        choose_mid = torch.rand_like(base_t0) < config.mid_time_probability
        return torch.where(choose_mid, mid_roots, high_roots)

    @staticmethod
    def _sample_soar_schedule_cell_band(
        *,
        shape: torch.Size | tuple[int, ...],
        schedule: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        """Sample uniformly over schedule cells intersecting one time band."""
        cell_high = torch.minimum(schedule[:-1], torch.full_like(schedule[:-1], upper))
        cell_low = torch.maximum(schedule[1:], torch.full_like(schedule[1:], lower))
        valid_indices = torch.nonzero(cell_high > cell_low, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError(
                f"No production-schedule cell intersects SOAR band [{lower}, {upper}]."
            )
        sampled_offset = torch.randint(
            valid_indices.numel(), shape, device=schedule.device
        )
        sampled_index = valid_indices[sampled_offset]
        sampled_low = cell_low[sampled_index]
        sampled_high = cell_high[sampled_index]
        return sampled_low + (sampled_high - sampled_low) * torch.rand(
            shape, dtype=schedule.dtype, device=schedule.device
        )

    def sample_soar_auxiliary_times(
        self, *, t0: torch.Tensor, config: ECSISOARConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move one production-schedule step, then sample later forward times."""
        schedule_values = self.get_effective_sampling_schedule(
            config.rollout_schedule_num_steps
        )
        work_dtype = torch.float64 if t0.dtype == torch.float64 else torch.float32
        schedule = torch.tensor(schedule_values, dtype=work_dtype, device=t0.device)
        t0_work = t0.to(dtype=work_dtype)
        cell_index = ((schedule[:-1] >= t0_work.unsqueeze(-1)).sum(dim=-1) - 1).clamp(
            min=0, max=config.rollout_schedule_num_steps - 1
        )
        cell_start = schedule[cell_index]
        cell_end = schedule[cell_index + 1]
        cell_fraction = ((cell_start - t0_work) / (cell_start - cell_end)).clamp(0, 1)
        u0 = cell_index.to(dtype=work_dtype) + cell_fraction
        u1 = (u0 + config.rollout_schedule_step_count).clamp_max(
            config.rollout_schedule_num_steps
        )
        next_index = (
            torch.floor(u1)
            .to(dtype=torch.long)
            .clamp_max(config.rollout_schedule_num_steps - 1)
        )
        next_fraction = u1 - next_index.to(dtype=work_dtype)
        t1 = torch.lerp(
            schedule[next_index], schedule[next_index + 1], next_fraction
        ).clamp_min(self.time_min)
        t1 = t1.to(dtype=t0.dtype)

        uniform = torch.rand(
            (*t1.shape, config.auxiliary_samples_per_root),
            dtype=t1.dtype,
            device=t1.device,
        )
        alpha_t1 = self.coeff.alpha(t1)
        feasible_min = self.coeff.alpha(torch.full_like(t1, self.time_max)) / (
            alpha_t1.clamp_min(torch.finfo(alpha_t1.dtype).eps)
        )
        retention_min = torch.maximum(
            torch.full_like(t1, config.forward_retention_min), feasible_min
        ).clamp_max(1.0)
        retention = (
            retention_min.unsqueeze(-1) + (1.0 - retention_min.unsqueeze(-1)) * uniform
        )
        t2 = 1.0 - retention * alpha_t1.unsqueeze(-1)
        t2 = torch.maximum(t2, t1.unsqueeze(-1))
        t2 = torch.minimum(t2, torch.full_like(t2, self.time_max))
        return t1, t2

    def _prepare_soar_rollout_call(
        self,
        *,
        config: ECSISOARConfig,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally apply production churn before the detached rollout call."""
        if not config.apply_rollout_churn:
            return x_t, t
        output = torch.empty_like(x_t)
        call_times = torch.empty_like(t)
        for batch_index in range(t.shape[0]):
            for root_index in range(t.shape[1]):
                state, call_time = self._apply_forward_pinned_churn(
                    x_t[batch_index : batch_index + 1, root_index : root_index + 1],
                    x_T[batch_index : batch_index + 1, root_index : root_index + 1],
                    mask[batch_index : batch_index + 1, root_index : root_index + 1],
                    float(t[batch_index, root_index]),
                )
                output[batch_index, root_index] = state[0, 0]
                call_times[batch_index, root_index] = call_time
        return output, call_times

    def _apply_soar_sampler_update(
        self,
        *,
        f_input: FoldingInput,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the class-routed production update to independently timed roots."""
        output = torch.empty_like(x_t)
        sde_atom_mask = self._get_sde_atom_mask(f_input)
        for batch_index in range(t.shape[0]):
            selected_atoms = sde_atom_mask[batch_index : batch_index + 1]
            has_sde_atoms = self.sampler_sde_atom_classes == ("all",) or bool(
                selected_atoms.any()
            )
            for root_index in range(t.shape[1]):
                state = self._apply_class_selective_update(
                    x_t[batch_index : batch_index + 1, root_index : root_index + 1],
                    x_0_hat[batch_index : batch_index + 1, root_index : root_index + 1],
                    x_T[batch_index : batch_index + 1, root_index : root_index + 1],
                    mask[batch_index : batch_index + 1, root_index : root_index + 1],
                    selected_atoms,
                    has_sde_atoms,
                    float(t[batch_index, root_index]),
                    float(t_next[batch_index, root_index]),
                    noise=noise[
                        batch_index : batch_index + 1, root_index : root_index + 1
                    ],
                )
                output[batch_index, root_index] = state[0, 0]
        return output.detach()

    @staticmethod
    def _masked_coordinate_rmsd(
        displacement: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if displacement.shape[:-1] != mask.shape:
            raise ValueError("Coordinate displacement and atom-mask shapes must match.")
        weights = mask.to(displacement.dtype)
        squared = displacement.square().sum(dim=-1)
        return torch.sqrt(
            (squared * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
        )

    def _exact_ecsi_forward_transition(
        self,
        *,
        x_t1: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample the exact ECSI bridge transition from t1 to later t2."""
        alpha_t1 = self.coeff.alpha(t1).unsqueeze(-1)
        retention = self.coeff.alpha(t2) / alpha_t1.clamp_min(torch.finfo(t1.dtype).eps)
        gamma_t1 = self.coeff.gamma(t1).unsqueeze(-1)
        gamma_t2 = self.coeff.gamma(t2)
        variance = self.coeff.eta * (
            gamma_t2.square() - retention.square() * gamma_t1.square()
        )
        tolerance = (
            32.0
            * torch.finfo(variance.dtype).eps
            * (gamma_t2.square() + retention.square() * gamma_t1.square())
            .abs()
            .clamp_min(1.0)
        )
        if torch.any(variance < -tolerance):
            raise RuntimeError("Exact ECSI forward-transition variance is negative.")
        variance = variance.clamp_min(0.0)
        branch_shape = (*t2.shape, *x_t1.shape[-2:])
        x_t1_branches = x_t1.unsqueeze(-3).expand(branch_shape)
        x_T_branches = x_T.unsqueeze(-3).expand(branch_shape)
        branch_mask = mask.unsqueeze(-2).expand(*t2.shape, mask.shape[-1])
        mean = (
            retention[..., None, None] * x_t1_branches
            + (self.coeff.beta(t2) - retention * self.coeff.beta(t1).unsqueeze(-1))[
                ..., None, None
            ]
            * x_T_branches
        )
        if noise is None:
            noise = torch.randn_like(mean)
        elif noise.shape != mean.shape:
            raise ValueError("ECSI forward-transition noise shape is incompatible.")
        noise = noise.masked_fill(~branch_mask[..., None], 0.0)
        x_t2 = mean + torch.sqrt(variance)[..., None, None] * noise
        x_t2 = x_t2.masked_fill(~branch_mask[..., None], 0.0)
        return {
            "x_t2": x_t2,
            "variance": variance,
            "retention": retention,
            "noise": noise,
        }

    def _postprocess_soar_endpoint(
        self,
        *,
        endpoint: torch.Tensor,
        x_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the endpoint transform used by production ECSI sampling."""
        if self.align_x_0_hat_to_x_t:
            endpoint = custom_rigid_align(
                endpoint, x_t, mask, rotation_only=True, output_mask=mask
            )
        return do_centering(endpoint, mask=mask)

    def _get_sde_atom_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Return selected atoms for class-routed SDE updates.

        Global churn is deliberately outside this selector: every atom reaches
        the same post-churn time before the shared score-model call.
        """
        if self.sampler_sde_atom_classes == ("all",):
            return f_input.atom.pad_mask

        selected_token = torch.zeros_like(f_input.token.pad_mask)
        if "protein" in self.sampler_sde_atom_classes:
            selected_token |= f_input.token.is_protein
        if "ligand" in self.sampler_sde_atom_classes:
            selected_token |= f_input.token.is_ligand
        if "rna" in self.sampler_sde_atom_classes:
            selected_token |= f_input.token.is_rna
        if "dna" in self.sampler_sde_atom_classes:
            selected_token |= f_input.token.is_dna
        if "peptide" in self.sampler_sde_atom_classes:
            peptide_chain = f_input.chain.is_protein & (f_input.chain.num_residues < 16)
            peptide_chain &= f_input.chain.pad_mask
            token_matches_peptide = (
                f_input.token.asym_id[:, :, None] == f_input.chain.asym_id[:, None, :]
            )
            selected_token |= (token_matches_peptide & peptide_chain[:, None, :]).any(
                dim=-1
            )
        selected_token &= f_input.token.pad_mask
        selected_atom = torch.gather(
            selected_token,
            1,
            f_input.atom.token_index,
        )
        return selected_atom & f_input.atom.pad_mask

    # ============================================================
    # For inference
    # ============================================================
    def get_effective_sampling_schedule(self, num_steps: int) -> list[float]:
        """Return the production schedule after high-churn knot warping."""
        times = list(self.get_sampling_schedule(num_steps))
        t_hi = float(times[0])
        t_lo = self.churn_end_time
        band = [
            index
            for index, time_value in enumerate(times)
            if t_lo < float(time_value) < t_hi
        ]
        if band and self.stepwarp_power != 1.0:
            lo_index, hi_index = band[0], band[-1]
            count = hi_index - lo_index + 1
            for offset, index in enumerate(range(lo_index, hi_index + 1)):
                fraction = (offset + 1) / (count + 1)
                warped_fraction = 1.0 - (1.0 - fraction) ** self.stepwarp_power
                times[index] = t_hi - (t_hi - t_lo) * warped_fraction
        if len(times) != num_steps + 1:
            raise RuntimeError(
                f"Expected {num_steps + 1} ECSI time points, got {len(times)}."
            )
        if any(left <= right for left, right in zip(times[:-1], times[1:], strict=True)):
            raise RuntimeError("Effective ECSI sampling schedule must decrease.")
        return [float(time_value) for time_value in times]

    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        r"""Sample structures via ECSI sampling.

        Implements Algorithm 1 from the paper with Euler discretization.
        Uses stochasticity control via `sampling_eta`-style parameters.

        The sampling SDE is:
        dX_t = b(t, X_t, x_T) dt + \sqrt{2\epsilon_t} dW_t

        where:
        b(t, x_t, x_T) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
                       + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
        \epsilon_t = \eta (\gamma_t \dot{\gamma}_t - \dot{\alpha}_t/\alpha_t \gamma_t^2)
        """
        model = self.score_model

        # Get the exact production schedule (from t_max toward t_min).
        times = self.get_effective_sampling_schedule(num_steps)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)
        sde_atom_mask = self._get_sde_atom_mask(f_input)
        has_sde_atoms = self.sampler_sde_atom_classes == ("all",) or bool(
            sde_atom_mask.any()
        )

        # Compute time-independent variables
        z = model.get_pair_conditioning(f_input, z)
        q, c, p = model.get_atom_embeddings(f_input, z)
        pair_bias = model.get_pair_bias(z)

        def run_step(x_t: torch.Tensor, t: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, c_noise.view(1, 1))
            return self.inference_step(
                f_input, x_t, x_T, t, q, c, p, s, pair_bias, chunk_size
            )

        traj: list[torch.Tensor] = []

        def append_traj(x_t: torch.Tensor):
            if return_traj:
                traj.append(x_t.cpu())

        # Sampling loop
        append_traj(x_t)
        for step_idx in range(num_steps):
            # Apply random augmentation without centering to preserve x_T/x_t translation.
            x_t, x_T = self.random_augmentation(x_t, x_T, mask=mask, centering=False)

            t = times[step_idx]
            t_next = times[step_idx + 1]

            x_noisy, t = self._apply_forward_pinned_churn(x_t, x_T, mask, t)

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_noisy, t)

            if self.align_x_0_hat_to_x_t:
                # Rotate x_0_hat toward x_t before centering.
                x_0_hat = custom_rigid_align(x_0_hat, x_noisy, mask, rotation_only=True)

            # Centering the predicted x_0_hat
            x_0_hat = do_centering(x_0_hat, mask=mask)

            x_t = self._apply_class_selective_update(
                x_noisy,
                x_0_hat,
                x_T,
                mask,
                sde_atom_mask,
                has_sde_atoms,
                t,
                t_next,
            )
            append_traj(x_t)

        sample_out: dict[str, torch.Tensor] = {}
        sample_out["init_coordinates"] = x_T
        sample_out["coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Natom, 3)

        return sample_out

    def sample_prior(self, f_input: FoldingInput, num_samples: int) -> torch.Tensor:
        """Sample xT (prior) coordinates for ECSI sampling.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_samples : int
            Number of diffusion samples

        Returns
        -------
        x_T : torch.Tensor
            prior coordinates. Shape (B, N, Natom, 3).
        """
        x_apo = f_input.atom.prior_coords.permute(0, 2, 1, 3)  # [B, Nprior, L, 3]
        apo_mask = f_input.atom.pad_mask  # [B, L]

        # If num_diffusion_samples > num_prior, cycle through prior coords
        num_prior = x_apo.shape[-3]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = x_apo[:, idx, :, :]  # [B, N, L, 3]
        x_T_mask = apo_mask.unsqueeze(-2)  # [B, 1, L]

        # Apply random augmentation without centering to preserve the prior distribution.
        x_T = self.random_augmentation(x_T, mask=x_T_mask, centering=False)
        return x_T

    def inference_step(
        self,
        f_input: FoldingInput,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        t: float,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
        pair_bias: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        x_T : torch.Tensor
            Prior (apo) coordinates. Shape (B, N, L, 3).
        t : float
            Diffusion time value for the current step, in range [0, 1].
        q : torch.Tensor
            The atom single representation, shape [B, Natom, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, Natom, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, Natom, Natom, c_atompair].
        s : torch.Tensor
            Single conditioning. Shape (B, 1, L, c_s), broadcast to (B, N, L, c_s).
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 3).
        """
        token_index = f_input.atom.token_index  # [B, Natom]
        atom_mask = f_input.atom.pad_mask  # [B, Natom]
        token_mask = f_input.token.pad_mask  # [B, L]

        c_in, c_skip, c_out = self._get_bridge_scalings(t)  # [B, N]

        # Input preconditioning: r_noisy = c_in * x_t
        r_noisy = c_in * x_t

        # End-point conditioning: r_T = x_T / sigma_data_end
        r_T = x_T / self.sigma_data_end  # [B, N, L, 3]
        r_noisy = torch.cat([r_noisy, r_T], dim=-1)

        def _step(r: torch.Tensor) -> torch.Tensor:
            return self.score_model.step(
                r,  # [B, N, L, 6]
                q,  # [B, Natom, c_atom]
                c,  # [B, Natom, c_atom]
                p,  # [B, Natom, Natom, c_atompair]
                token_index,  # [B, Natom]
                atom_mask,  # [B, Natom]
                s,  # [B, 1, Ntoken, c_s]
                pair_bias,  # [B, Nblock, H, Lt, Lt]
                token_mask,  # [B, L]
            )

        if chunk_size is None:
            r_update = _step(r_noisy)
        else:
            r_update = torch.zeros_like(x_t)
            for st in range(0, x_t.shape[1], chunk_size):
                end = st + chunk_size
                r_update[:, st:end] = _step(r_noisy[:, st:end])

        # Output preconditioning: \hat{x}_0 = c_{skip} * x_t + c_{out} * F_\theta
        x_out = c_skip * x_t + c_out * r_update
        return x_out

    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        r"""Get the time schedule for diffusion sampling.

        Uses the configured phase-power schedule adapted for t \in [0, 1].

        Parameters
        ----------
        num_steps : int, optional
            Number of sampling steps. If None, uses `self.sampling.steps`.

        Returns
        -------
        times : list[float]
            Time schedule. Shape (num_steps + 1,), from t_max to 0.
        """
        time_max = self.time_max
        time_min = self.time_min
        ode_st = self.churn_end_time
        total_scale = time_max - time_min

        ode_power = self.ode_step_power
        churn_power = self.churn_step_power
        churn_fraction = self.churn_step_fraction
        ode_fraction = 1.0 - churn_fraction

        s = np.linspace(0, 1, num_steps)
        churn_mask = s <= churn_fraction
        ode_mask = s > churn_fraction

        ode_u = (ode_st - time_min) / total_scale

        u = np.zeros_like(s)
        if churn_mask.any():
            progress = s[churn_mask] / _clip(churn_fraction)
            _u = 1.0 - (1.0 - ode_u) * progress**churn_power
            u[churn_mask] = _u
        if ode_mask.any():
            progress = (1.0 - s[ode_mask]) / _clip(ode_fraction)
            _u = ode_u * progress**ode_power
            u[ode_mask] = _u
        t_unit = u.clip(0.0, 1.0)

        times = time_min + total_scale * t_unit
        times = times.tolist()
        # Append time=0.0 at the end to ensure the final step reaches t=0.
        times.append(0.0)
        return times

    def _sigma_eff_to_time(self, target: float, lo: float, hi: float) -> float:
        r"""Return a churn time in ``[lo, hi]`` for an effective-noise target.

        ``sigma_eff`` is monotonic over the sampling interval. The upper endpoint
        is returned when the requested inflation cannot fit inside the configured
        churn band, preserving the trained-time support instead of extrapolating.
        """
        if hi <= lo:
            return lo

        coeff = self.coeff
        lo_sigma = float(coeff.sigma_eff(lo))
        hi_sigma = float(coeff.sigma_eff(hi))
        if target <= lo_sigma:
            return lo
        if target >= hi_sigma:
            return hi

        low, high = lo, hi
        for _ in range(60):
            mid = 0.5 * (low + high)
            if float(coeff.sigma_eff(mid)) < target:
                low = mid
            else:
                high = mid
        return high

    def _apply_sigma_matched_churn(
        self,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        *,
        chi: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        r"""Apply a bridge conditional with ``sigma_eff`` inflated by ``chi``.

        The model still receives x-space coordinates. ``sigma_eff`` only chooses
        ``t_hat`` such that ``sigma_eff(t_hat) = (1 + chi) * sigma_eff(t)``, when
        that target lies inside the configured churn band.
        """
        if chi <= 0.0:
            return x_t, t

        coeff = self.coeff
        target = (1.0 + chi) * float(coeff.sigma_eff(t))
        t_hat = self._sigma_eff_to_time(target, t, self.churn_max_time)
        if t_hat <= t:
            return x_t, t

        alpha_ratio = float(coeff.alpha(t_hat)) / _clip(float(coeff.alpha(t)))
        variance = (
            float(coeff.gamma(t_hat)) ** 2 - (alpha_ratio**2) * float(coeff.gamma(t)) ** 2
        )
        if variance <= 0.0:
            return x_t, t

        if noise is None:
            noise = torch.randn_like(x_t)
        elif noise.shape != x_t.shape:
            raise ValueError("Sigma-matched churn noise must match x_t shape.")
        noise = noise.masked_fill(~mask[..., None], 0.0)
        x_hat = (
            alpha_ratio * x_t
            + (float(coeff.beta(t_hat)) - alpha_ratio * float(coeff.beta(t))) * x_T
            + math.sqrt(variance) * noise
        )
        return x_hat, t_hat

    def _churn_factor_at_time(self, t: float) -> float:
        span = self.churn_max_time - self.churn_end_time
        weight = (t - self.churn_end_time) / span if span > 0.0 else 0.0
        weight = min(max(weight, 0.0), 1.0) ** 2
        return self.churn_factor * (1.0 + (self.churn_max_multiplier - 1.0) * weight)

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        # The schedule starts at ``time_max``. Keep both churn bounds open so
        # that the initial state is never perturbed or consumes RNG.
        if not self.churn_end_time < t < self.churn_max_time:
            return x_t, t

        return self._apply_sigma_matched_churn(
            x_t,
            x_T,
            mask,
            t,
            chi=self._churn_factor_at_time(t),
            noise=noise,
        )

    def _apply_class_selective_update(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        sde_atom_mask: torch.Tensor,
        has_sde_atoms: bool,
        t: float,
        t_next: float,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Preserve [all] or overlay selected updates on an SI-ODE base."""
        selected_mode, selected_ode_type = self._select_update_method(t)
        if self.sampler_sde_atom_classes == ("all",):
            return self._update_step(
                x_t,
                x_0_hat,
                x_T,
                mask,
                t,
                t_next,
                mode=selected_mode,
                ode_type=selected_ode_type,
                step_scale=self.sampler_step_scale,
                noise=noise,
            )

        x_ode = self._update_step(
            x_t,
            x_0_hat,
            x_T,
            mask,
            t,
            t_next,
            mode="ode",
            ode_type="si",
            step_scale=self.sampler_step_scale,
            noise=noise,
        )
        if not has_sde_atoms:
            return x_ode

        if selected_mode == "ode" and selected_ode_type == "si":
            return x_ode

        sde_mask = mask & sde_atom_mask[:, None, :]
        x_sde = self._update_step(
            x_t,
            x_0_hat,
            x_T,
            sde_mask,
            t,
            t_next,
            mode=selected_mode,
            ode_type=selected_ode_type,
            step_scale=self.sampler_step_scale,
            noise=noise,
        )
        return torch.where(sde_atom_mask[:, None, :, None], x_sde, x_ode)

    def _select_update_method(self, t: float) -> tuple[str, str]:
        """Use the high-time profile or the fixed low-time SI-ODE phase."""
        if self.sampler_switch_time is not None and t <= self.sampler_switch_time:
            return "ode", "si"
        return self.sampler_mode, self.sampler_ode_type

    def _update_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        t_next: float,
        mode: str = "ode",  # 'ode' or 'sde'
        ode_type: str = "si",  # 'si' or 'ecsi'
        step_scale: float = 1.0,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SDE step for ECSI sampling.
        See Algorithm 1 of ECSI paper.

        Parameters
        ----------
        x_t : torch.Tensor
            Current coordinates at time t. Shape (*, Natom, 3).
        x_0_hat : torch.Tensor
            Denoised coordinates predicted by the score model. Shape (*, Natom, 3).
        x_T : torch.Tensor
            Prior (apo) coordinates. Shape (*, Natom, 3).
        mask : torch.Tensor
            Atom mask. Shape (B, Natom).
        t : float
            Current time value.
        t_next : float
            Next time value after the update step.
        mode : str, optional
            Update mode: 'sde' or 'ode'
        ode_type : str, optional
            Type of update: 'si' or 'ecsi'.
        step_scale : float, optional
            Multiplier for the deterministic ODE displacement.
        """
        C = self.coeff
        alpha_t, alpha_dot = C.alpha(t), C.alpha_deriv(t)
        beta_t, beta_dot = C.beta(t), C.beta_deriv(t)

        if mode == "sde":
            # SDE update
            gamma_t = C.gamma(t)
            eps: float = C.eps(t)
            if noise is None:
                noise = torch.randn_like(x_t)
            elif noise.shape != x_t.shape:
                raise ValueError("Sampler update noise must match x_t shape.")
            noise = noise.masked_fill(~mask[..., None], 0.0)

            if ode_type == "si":
                # SI SDE drift pinned to the denoised endpoint:
                # b = (beta_dot / beta) x_t
                #     + (alpha_dot - alpha * beta_dot / beta) x_0_hat
                f_t = beta_dot / _clip(beta_t)
                s_t = alpha_dot - alpha_t * beta_dot / _clip(beta_t)
                drift = f_t * x_t + s_t * x_0_hat
            else:
                gamma_dot = C.gamma_deriv(t)
                x_N = x_T  # For clarity with the paper's notation.

                # Line 5
                # \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / gamma
                z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_N) / _clip(gamma_t)

                # Line 8: Compute drift term
                # d = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_N
                #     + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
                drift = (
                    alpha_dot * x_0_hat
                    + beta_dot * x_N
                    + (gamma_dot + eps / _clip(gamma_t)) * z_hat
                )

            # Euler-Maruyama update
            dt = t - t_next
            x_upd = x_t - drift * dt + _sqrt(2 * eps * dt) * noise
        else:
            # ODE update
            if ode_type == "si":
                # SI ODE update
                alpha_tm, beta_tm = C.alpha(t_next), C.beta(t_next)
                c_skip = beta_tm / beta_t
                c_update = alpha_tm - alpha_t * c_skip
                x_target = c_skip * x_t + c_update * x_0_hat
            else:
                # ECSI ODE update
                alpha_tm, beta_tm = C.alpha(t_next), C.beta(t_next)
                gamma_t, gamma_tm = C.gamma(t), C.gamma(t_next)
                z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_T) / _clip(gamma_t)
                x_target = alpha_tm * x_0_hat + beta_tm * x_T + gamma_tm * z_hat
            x_upd = x_t + step_scale * (x_target - x_t)
        return x_upd
