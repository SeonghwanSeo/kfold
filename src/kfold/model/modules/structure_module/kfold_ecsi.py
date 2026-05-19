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
from kfold.model.modules.score_model.base import AF3StyleDiffusionModule
from kfold.utils.geometry.random_augment import CenterRandomAugmentation, do_centering
from kfold.utils.geometry.rigid_align import get_rigid_transform_torch
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig
from kfold.utils.torch import expand_dim

from .base import BaseStructureModule

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

    class Config(BaseConfig):
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
        churn_factor : float
            The factor controlling the magnitude of forward-pinned churn noise.
        churn_end_time : float
            The time value at which to end forward-pinned churn.

        # Inference time scheduling parameters
        churn_step_fraction : float
            The fraction of the total sampling steps to apply churn.
        churn_step_power : float
            The exponent controlling the time schedule for churn steps.
        ode_step_power : float
            The exponent controlling the time schedule for ODE steps.

        # Training time scheduling
        train_time_schedule_params : tuple[float, float]
            A tuple of (mu, std) for the time sampling schedule during training.
            Time values are sampled from:
                t ~ logistic(mu, std), then scaled to [time_min, time_max].
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
        churn_factor: float = 0.1
        churn_end_time: float = 0.5
        churn_step_fraction: float = 0.4
        churn_step_power: float = 1.0
        ode_step_power: float = 4.0

        # Train time scheduling (mu, std)
        train_time_schedule_params: tuple[float, float] = (-2.15, 2.25)

    def __init__(self, cfg: Config, score_model: AF3StyleDiffusionModule):
        """Initialize the ECSI module.

        The constructor copies the high-level config fields onto runtime
        attributes, then immediately validates and normalizes the nested
        sampling config so downstream code can assume a runtime-ready ECSI
        configuration.
        """
        super().__init__(cfg, score_model)
        self.cfg = cfg
        self.score_model: AF3StyleDiffusionModule = score_model

        # ECSI preconditioning parameters
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy

        # ECSI coefficients
        self.coeff = SICoeffs(cfg.gamma_max, cfg.gamma_power, cfg.eta)
        self.time_min: float = cfg.time_min
        self.time_max: float = cfg.time_max

        # Train time scheduling
        self.train_time_schedule_params: tuple[float, float] = (
            cfg.train_time_schedule_params
        )

        # Inference time sampling
        self.align_x_0_hat_to_x_t: bool = cfg.align_x_0_hat_to_x_t
        self.churn_factor: float = cfg.churn_factor
        self.churn_end_time: float = cfg.churn_end_time
        self.churn_step_fraction: float = cfg.churn_step_fraction
        self.churn_step_power: float = cfg.churn_step_power
        self.ode_step_power: float = cfg.ode_step_power

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
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
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
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
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
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
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
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, L, c_s).
        z_trunk : torch.Tensor
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
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
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

        # Rigidly align x_T to x_0
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
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_steps: int = 200,
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
        model: AF3StyleDiffusionModule = self.score_model

        # Get time schedule (from t_max toward t_min)
        times = self.get_sampling_schedule(num_steps)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)

        # Compute time-independent variables
        z = model.get_pair_conditioning(f_input, z_trunk)
        q, c, p = model.get_atom_embeddings(f_input, s_trunk, z)
        pair_bias = model.get_pair_bias(z)
        del z_trunk, z  # Free up memory for large LxL tensors

        def run_step(x_t: torch.Tensor, t: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, s_trunk, c_noise.view(1, 1))
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
            # Apply random augmentation
            x_t, x_T = self.random_augmentation(x_t, x_T, mask=mask, centering=False)

            t = times[step_idx]
            t_next = times[step_idx + 1]

            # Early-stage forward-pinned churn.
            x_noisy, t = self._apply_forward_pinned_churn(x_t, x_T, mask, t)

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_noisy, t)

            if self.align_x_0_hat_to_x_t:
                # Rigidly align x_0_hat to x_t before centering.
                x_0_hat = custom_rigid_align(x_0_hat, x_noisy, mask)

            # Centering the predicted x_0_hat
            x_0_hat = do_centering(x_0_hat, mask=mask)

            # Update x_t
            x_t = self._update_step(x_noisy, x_0_hat, x_T, mask, t, t_next)
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

        # Apply random augmentation to prior coords
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

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        mask: torch.Tensor,
        t: float,
    ) -> tuple[torch.Tensor, float]:
        if t <= self.churn_end_time:
            # No churn applied before or at churn_end
            return x_t, t

        dt = (1 - t) * self.churn_factor

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
            Type of ODE update: 'si' or 'ecsi'.

        """
        C = self.coeff
        alpha_t, alpha_dot = C.alpha(t), C.alpha_deriv(t)
        beta_t, beta_dot = C.beta(t), C.beta_deriv(t)

        if mode == "sde":
            # SDE update
            gamma_t, gamma_dot = C.gamma(t), C.gamma_deriv(t)
            eps: float = C.eps(t)

            x_N = x_T  # For clarity with the paper's notation.

            # Line 5
            # \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / gamma
            z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_N) / _clip(gamma_t)

            # Line 7: Sample noise for SDE step
            # \bar{z} ~ N(0, I)
            noise = torch.randn_like(x_t).masked_fill_(~mask[..., None], 0.0)

            # Line 8: Compute drift term
            # d = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_N
            #     + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
            drift = (
                alpha_dot * x_0_hat
                + beta_dot * x_N
                + (gamma_dot + eps / _clip(gamma_t)) * z_hat
            )

            # Line 9: Euler-Maruyama update
            dt = t - t_next
            x_upd = x_t - drift * dt + _sqrt(2 * eps * dt) * noise
        else:
            # ODE update
            if ode_type == "si":
                # SI ODE update
                alpha_tm, beta_tm = C.alpha(t_next), C.beta(t_next)
                c_skip = beta_tm / beta_t
                c_update = alpha_tm - alpha_t * c_skip
                x_upd = c_skip * x_t + c_update * x_0_hat
            else:
                # ECSI ODE update
                alpha_tm, beta_tm = C.alpha(t_next), C.beta(t_next)
                gamma_t, gamma_tm = C.gamma(t), C.gamma(t_next)
                z_hat = (x_t - alpha_t * x_0_hat - beta_t * x_T) / _clip(gamma_t)
                x_upd = alpha_tm * x_0_hat + beta_tm * x_T + gamma_tm * z_hat
        return x_upd
