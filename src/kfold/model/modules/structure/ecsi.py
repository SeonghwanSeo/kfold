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
from typing import TypeVar

import numpy as np
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives.utils import expand_dim
from kfold.utils.geometry.random_augment import CenterRandomAugmentation, do_centering
from kfold.utils.geometry.rigid_align import get_rigid_transform_torch
from kfold.utils.registry import STRUCTURE_MODULE

from .sample_diffusion import BaseStructureModule
from .score_model import DiffusionModule

RIGID_ALIGN = 0  # conduct centering ; kabsch align
NO_ALIGN = 1  # no centering; no kabsch align

_T = TypeVar("_T", float, torch.Tensor)


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
) -> torch.Tensor:
    """
    Torch implementation of weighted rigid alignment.
    """
    if mask is None:
        mask = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)
    if not mask.any():
        return coords

    original_dtype = coords.dtype
    mask_bool = mask.bool().unsqueeze(-1)
    coords = coords.masked_fill(~mask_bool, 0.0)
    target = target.masked_fill(~mask_bool, 0.0)

    with torch.autocast(device_type=coords.device.type, enabled=False):
        coords, target = coords.float(), target.float()
        weights = mask.to(dtype=coords.dtype)
        RT, T = get_rigid_transform_torch(coords, target, weights)
        aligned_coords = coords @ RT
        if not rotation_only:
            aligned_coords += T.unsqueeze(-2)

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

    # Compute \epsilon = \eta (\gamma \dot{\gamma} - \dot{\alpha}/\alpha \gamma^2)
    def eps(self, t: _T) -> _T:
        alpha, alpha_dot = self.alpha(t), self.alpha_deriv(t)
        gamma, gamma_dot = self.gamma(t), self.gamma_deriv(t)
        return self.eta * (  # type: ignore
            gamma * gamma_dot - alpha_dot / _clip(alpha) * gamma**2
        )


@STRUCTURE_MODULE.register()
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

    NOTE: We design our own stochastic sampler for structure prediction.
    While ECSI uses a stochastic sampler with SDE formulation, we design similar sampler
    to EDM with churn & ODE formulation.
    """

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
        sampler_step_scale : float
            Multiplier for deterministic ODE update displacement. A value of 1.5
            mirrors the step scale convention used by AF3/EDM samplers.
        churn_factor : float
            The factor controlling the magnitude of forward-pinned churn noise.
        churn_max_multiplier : float
            Maximum multiplier applied to churn_factor in the high-time ramp.
        churn_end_time : float
            The time value at which to end forward-pinned churn.

        # Inference time scheduling parameters
        churn_step_fraction : float
            The fraction of the total sampling steps to apply churn.
        churn_step_power : float
            The exponent controlling the time schedule for churn steps.
        ode_step_power : float
            The exponent controlling the time schedule for ODE steps.
        stepwarp_power : float
            Exponent used to redistribute high-churn schedule knots. A value of
            1.0 preserves the native schedule without applying the warp.
        svgd_step : float
            Multiplier for the SVGD repulsion displacement.
        svgd_cap_frac : float
            Maximum SVGD displacement relative to the ensemble spread.
        svgd_num_eps : float
            Numerical floor used by the SVGD bandwidth and normalization.
        svgd_skip_rms : float
            Spread threshold below which SVGD is skipped.

        # Training time scheduling
        train_time_schedule : str
            Training-time sampling schedule. Supported values are "logistic" and
            "uniform".
        train_time_schedule_params : tuple[float, float]
            A tuple of (mu, std) for the logistic time sampling schedule.
            Time values are sampled from sigmoid(N(mu, std)), then scaled to
            [time_min, time_max].
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
        sampler_mode: str = "ode"
        sampler_ode_type: str = "si"
        sampler_step_scale: float = 1.0
        sampler_switch_gamma: float | None = None
        sampler_after_switch_mode: str = "ode"
        sampler_after_switch_ode_type: str = "si"
        churn_factor: float = 0.1
        churn_max_multiplier: float = 4.0
        churn_end_time: float = 0.5
        churn_step_fraction: float = 0.4
        churn_step_power: float = 1.0
        ode_step_power: float = 2.0
        stepwarp_power: float = 0.5

        # SVGD sample spreading
        svgd_step: float = 1.0
        svgd_cap_frac: float = 0.05
        svgd_num_eps: float = 1e-8
        svgd_skip_rms: float = 1e-6

        # Train time scheduling
        train_time_schedule: str = "logistic"
        train_time_schedule_params: tuple[float, float] = (-2.15, 2.25)

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

        # Inference time sampling
        self.align_x_0_hat_to_x_t: bool = cfg.align_x_0_hat_to_x_t
        if cfg.sampler_mode not in {"ode", "sde"}:
            raise ValueError(f"Unknown ECSI sampler_mode: {cfg.sampler_mode}")
        if cfg.sampler_ode_type not in {"si", "ecsi"}:
            raise ValueError(f"Unknown ECSI sampler_ode_type: {cfg.sampler_ode_type}")
        if cfg.sampler_after_switch_mode not in {"ode", "sde"}:
            raise ValueError(
                f"Unknown ECSI sampler_after_switch_mode: {cfg.sampler_after_switch_mode}"
            )
        if cfg.sampler_after_switch_ode_type not in {"si", "ecsi"}:
            raise ValueError(
                f"Unknown ECSI sampler_after_switch_ode_type: "
                f"{cfg.sampler_after_switch_ode_type}"
            )
        if cfg.sampler_step_scale <= 0:
            raise ValueError("ECSI sampler_step_scale must be positive.")
        if cfg.sampler_switch_gamma is not None and cfg.sampler_switch_gamma < 0:
            raise ValueError("ECSI sampler_switch_gamma must be non-negative.")
        if cfg.svgd_num_eps <= 0:
            raise ValueError("ECSI svgd_num_eps must be positive.")
        if any(
            value < 0
            for value in (
                cfg.churn_max_multiplier,
                cfg.stepwarp_power,
                cfg.svgd_step,
                cfg.svgd_cap_frac,
                cfg.svgd_skip_rms,
            )
        ):
            raise ValueError("ECSI sampler parameters must be non-negative.")
        self.sampler_mode: str = cfg.sampler_mode
        self.sampler_ode_type: str = cfg.sampler_ode_type
        self.sampler_step_scale: float = cfg.sampler_step_scale
        self.sampler_switch_gamma: float | None = cfg.sampler_switch_gamma
        self.sampler_after_switch_mode: str = cfg.sampler_after_switch_mode
        self.sampler_after_switch_ode_type: str = cfg.sampler_after_switch_ode_type
        self.churn_factor: float = cfg.churn_factor
        self.churn_max_multiplier: float = cfg.churn_max_multiplier
        self.churn_end_time: float = cfg.churn_end_time
        self.churn_step_fraction: float = cfg.churn_step_fraction
        self.churn_step_power: float = cfg.churn_step_power
        self.ode_step_power: float = cfg.ode_step_power
        self.stepwarp_power: float = cfg.stepwarp_power
        self.svgd_step: float = cfg.svgd_step
        self.svgd_cap_frac: float = cfg.svgd_cap_frac
        self.svgd_num_eps: float = cfg.svgd_num_eps
        self.svgd_skip_rms: float = cfg.svgd_skip_rms

        # NOTE: centering should be disabled.
        self.random_augmentation = CenterRandomAugmentation()

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
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        # Sample x
        with torch.autocast(f_input.device.type, enabled=False):
            train_input = self.sample_train_input(f_input, diffusion_batch_size)

        t = train_input["t"]  # [B, N]
        x_0 = train_input["x_0"]  # [B, N, Natom, 3]
        x_t = train_input["x_t"]  # [B, N, Natom, 3]
        x_T = train_input["x_T"]  # [B, N, Natom, 3]

        x_0_hat = self._forward_train(
            x_t=x_t,  # [B, N, Natom, 3]
            t=t,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            z=z,  # [B, Lt, Lt, c_z]
            x_T=x_T,  # [B, N, Natom, 3]
        )  # [B, N, Natom, 3]

        loss_weights = self.loss_weights(t)  # [B, N]

        return {
            "t": t,
            "x_t": x_t,
            "x_T": x_T,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        x_T: torch.Tensor | None = None,
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
        x_0 = expand_dim(x_holo, num_samples, dim=-3)  # [B, N, Natom, 3]
        x_0_mask = holo_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        num_prior = x_apo.shape[-3]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = x_apo[:, idx, :, :]  # [B, N, Natom, 3]
        x_T_mask = apo_mask.unsqueeze(-2)  # [B, 1, Natom]

        # Apply centering/coordinate augmentation
        x_0 = self.random_augmentation(x_0, mask=x_0_mask)

        # Rotate x_T toward x_0 while preserving the prior translation distribution.
        x_T = custom_rigid_align(x_T, x_0, x_0_mask, rotation_only=True)

        # === Interpolate to get x_t === #
        C = self.coeff
        _t = t[:, :, None, None]
        alpha_t, beta_t, gamma_t = C.alpha(_t), C.beta(_t), C.gamma(_t)

        # Sample noise
        noise = torch.randn_like(x_0)

        # ECSI interpolation with atom-wise noise.
        x_t = alpha_t * x_0 + beta_t * x_T + gamma_t * noise

        # Mask out unresolved/pad atoms.
        x_0.masked_fill_(~x_0_mask[..., None], 0.0)
        x_T.masked_fill_(~x_T_mask[..., None], 0.0)
        x_t.masked_fill_(~x_T_mask[..., None], 0.0)

        return {
            "t": t,
            "x_0": x_0,
            "x_t": x_t,
            "x_T": x_T,
        }

    # ============================================================
    # For inference
    # ============================================================
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

        # Get time schedule (from t_max toward t_min)
        times = self.get_sampling_schedule(num_steps)

        # Concentrate knots in the early high-churn band without changing NFE.
        t_hi = float(times[0])
        t_lo = self.churn_end_time
        times = list(times)
        band = [i for i, time in enumerate(times) if t_lo < float(time) < t_hi]
        if band and self.stepwarp_power != 1.0:
            lo_i, hi_i = band[0], band[-1]
            n = hi_i - lo_i + 1
            for k, i in enumerate(range(lo_i, hi_i + 1)):
                fraction = (k + 1) / (n + 1)
                warped_fraction = 1.0 - (1.0 - fraction) ** self.stepwarp_power
                times[i] = t_hi - (t_hi - t_lo) * warped_fraction

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)

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

            # Early-stage forward-pinned churn.
            span = float(times[0]) - self.churn_end_time
            weight = (float(t) - self.churn_end_time) / span if span > 0.0 else 0.0
            weight = min(max(weight, 0.0), 1.0) ** 2
            effective_churn_factor = self.churn_factor * (
                1.0 + (self.churn_max_multiplier - 1.0) * weight
            )
            x_noisy, t = self._apply_forward_pinned_churn(
                x_t,
                x_T,
                mask,
                t,
                churn_factor=effective_churn_factor,
            )

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_noisy, t)

            if self.align_x_0_hat_to_x_t:
                # Rotate x_0_hat toward x_t before centering.
                x_0_hat = custom_rigid_align(x_0_hat, x_noisy, mask, rotation_only=True)

            # Centering the predicted x_0_hat
            x_0_hat = do_centering(x_0_hat, mask=mask)

            # Redistribute endpoint estimates without another score evaluation.
            x_0_hat = self._apply_svgd_spread(x_0_hat, mask)

            sampler_mode, sampler_ode_type = self._select_update_method(t)

            # Update x_t
            x_t = self._update_step(
                x_noisy,
                x_0_hat,
                x_T,
                mask,
                t,
                t_next,
                mode=sampler_mode,
                ode_type=sampler_ode_type,
                step_scale=self.sampler_step_scale,
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

    def _apply_svgd_spread(
        self,
        x: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Spread endpoint samples without moving their masked centroid."""
        num_samples = x.shape[1]
        if num_samples < 2:
            return x

        mask = atom_mask.to(dtype=x.dtype)
        mask_4d = mask.unsqueeze(-1)

        if x.dtype in (torch.float16, torch.bfloat16):
            deviation = x.float()
            deviation.sub_(deviation.mean(dim=1, keepdim=True))
            deviation.mul_(mask_4d)
        else:
            deviation = (x - x.mean(dim=1, keepdim=True)) * mask_4d
        deviation_flat = deviation.flatten(start_dim=2)
        distance_flat = deviation_flat

        num_real = mask.sum(dim=-1, dtype=distance_flat.dtype).clamp_min(1.0)
        denominator = (3.0 * num_samples * num_real).clamp_min(1.0)
        spread_rms = torch.sqrt(
            (distance_flat * distance_flat).sum(dim=(1, 2)).clamp_min(0.0)
            / denominator.squeeze(-1)
        )
        if float(spread_rms.max()) <= self.svgd_skip_rms:
            return x

        # Avoid materializing pairwise atom-coordinate differences.
        distance_sq = torch.bmm(distance_flat, distance_flat.transpose(1, 2))
        norm_sq = distance_sq.diagonal(dim1=1, dim2=2).clone()
        distance_sq.mul_(-2.0)
        distance_sq.add_(norm_sq.unsqueeze(2))
        distance_sq.add_(norm_sq.unsqueeze(1))
        distance_sq.clamp_min_(0.0)
        distance_sq.div_(3.0 * num_real.unsqueeze(-1))

        off_diagonal = ~torch.eye(
            num_samples,
            dtype=torch.bool,
            device=x.device,
        )

        # NOTE: This is same to median(dim=...), but deterministic.
        pairwise_distance_sq = distance_sq[:, off_diagonal]
        pairwise_distance_sq = pairwise_distance_sq.sort(dim=-1).values
        median_index = (pairwise_distance_sq.shape[-1] - 1) // 2
        bandwidth = pairwise_distance_sq[:, median_index]

        bandwidth = bandwidth.clamp_min(self.svgd_num_eps)
        bandwidth = bandwidth.view(-1, 1, 1)

        kernel = distance_sq.div_(bandwidth).neg_().exp_()
        coefficient = kernel.square_()
        coefficient.mul_(2.0 / bandwidth)
        coefficient.masked_fill_(~off_diagonal.unsqueeze(0), 0.0)

        row_sum = coefficient.sum(dim=-1, keepdim=True)
        displacement_flat = torch.bmm(coefficient, distance_flat).neg_()
        displacement_flat.addcmul_(distance_flat, row_sum)
        displacement_flat.div_(num_samples)
        del coefficient, distance_flat, deviation_flat, deviation
        del norm_sq, distance_sq, kernel, bandwidth, row_sum
        displacement = displacement_flat.reshape_as(x)
        del displacement_flat
        displacement.mul_(mask_4d)
        displacement.sub_(displacement.mean(dim=1, keepdim=True))
        displacement.mul_(mask_4d)

        displacement_rms = torch.sqrt(
            (displacement * displacement).sum(dim=(1, 2, 3)) / denominator.squeeze(-1)
            + self.svgd_num_eps
        )
        cap = self.svgd_cap_frac * spread_rms
        scale = (cap / displacement_rms.clamp_min(self.svgd_num_eps)).clamp_max(1.0)
        displacement.mul_(scale.view(-1, 1, 1, 1))
        if displacement.dtype != x.dtype:
            displacement = displacement.to(dtype=x.dtype)
            displacement.sub_(displacement.mean(dim=1, keepdim=True))
            displacement.mul_(mask_4d)

        output = torch.add(x, displacement, alpha=self.svgd_step)
        if not torch.isfinite(output).all():
            raise RuntimeError("svgd-spread-t24: non-finite repulsion output")
        return output

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

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        *,
        churn_factor: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        if t <= self.churn_end_time:
            # No churn applied before or at churn_end
            return x_t, t

        effective_churn_factor = (
            self.churn_factor if churn_factor is None else churn_factor
        )
        dt = (1 - t) * effective_churn_factor

        C = self.coeff
        alpha_t, beta_t = C.alpha(t), C.beta(t)
        alpha_dot, beta_dot = C.alpha_deriv(t), C.beta_deriv(t)
        eps: float = C.eps(t)

        # Forward-pinned churn step
        noise = torch.randn_like(x_t).masked_fill_(~mask[..., None], 0.0)

        f_t = alpha_dot / alpha_t
        s_t = beta_dot - f_t * beta_t
        drift = f_t * x_t + s_t * x_T

        x_tm = x_t + drift * dt + _sqrt(2 * eps * dt) * noise
        tm = t + dt

        return x_tm, tm

    def _select_update_method(self, t: float) -> tuple[str, str]:
        if self.sampler_switch_gamma is None:
            return self.sampler_mode, self.sampler_ode_type

        # Reverse sampling starts near t=1 where gamma is also small. The switch is
        # intended for the late low-gamma phase after the gamma envelope has peaked.
        gamma_peak_time = 0.5 ** (1.0 / self.coeff.gamma_power)
        if t <= gamma_peak_time and self.coeff.gamma(t) <= self.sampler_switch_gamma:
            return self.sampler_after_switch_mode, self.sampler_after_switch_ode_type
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
            Multiplier for the deterministic ODE displacement, analogous to the
            step scale used in AF3/EDM samplers.

        """
        C = self.coeff
        alpha_t, alpha_dot = C.alpha(t), C.alpha_deriv(t)
        beta_t, beta_dot = C.beta(t), C.beta_deriv(t)

        if mode == "sde":
            # SDE update
            gamma_t = C.gamma(t)
            eps: float = C.eps(t)
            noise = torch.randn_like(x_t).masked_fill_(~mask[..., None], 0.0)

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
