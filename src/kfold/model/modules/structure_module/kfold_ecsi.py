"""Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI) for biomolecular
structure prediction.

Reference: "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553).

# Diffusion Path Design

Models the transition from source (apo) to target (holo) conformations.
- t=1 (Prior): Apo chain structures perturbed with 50 Å translational noise.
- t=0 (Data): Ground-truth assembled holo complexes.

# Decoupling COM and Intra-chain Dynamics

In standard diffusion formulations, the variance of the Center of Mass (COM) scales
proportionally to 1/sqrt(N_atom). This naturally suppresses global motion, leading to
overly deterministic COM trajectories and suboptimal sampling. To resolve this, our
ECSI bridge formulation decouples COM and intra-chain dynamics, enabling independent
noise scheduling for rigid-body translations versus local conformational changes.

# Decoupling Radial and Tangential COM Noise
To further enhance sampling diversity, we decompose the COM noise into radial and
tangential components. This allows separate control over the magnitude of noise that
pushes the structure directly towards/away from the target (radial) versus noise that
induces lateral "tentacle-like" exploration around the direct path (tangential).
"""

import dataclasses
import math
from typing import TypeVar

import numpy as np
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.ecsi_diffusion import ECSIDiffusionModule
from kfold.utils.geometry.random_augment import CenterRandomAugmentation, do_centering
from kfold.utils.geometry.rigid_align import rigid_align
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseECSI

_T = TypeVar("_T", float, torch.Tensor)


# === Utility functions with type flexibility and numerical stability handling === #
def _clip(t: _T, eps: float = 1e-10) -> _T:
    return t.clip(min=eps) if isinstance(t, torch.Tensor) else max(t, eps)  # type: ignore


def _sqrt(t: _T) -> _T:
    return _clip(t, eps=0) ** 0.5  # type: ignore


def _log(t: _T) -> _T:
    log = torch.log if isinstance(t, torch.Tensor) else math.log
    return log(_clip(t, eps=1e-10))  # type: ignore


# === Helper class/functions for ECSI bridge decomposition and noise handling === #
class ChainDecomposition:
    def __init__(self, f_input: FoldingInput):
        """Helper class for chain-aware decomposition and composition of coordinates.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs, used to extract chain_id.
        """
        num_chains = f_input.num_chains
        atom_mask = f_input.atom.pad_mask  # [B, Natom]
        token_mask = f_input.token.pad_mask  # [B, Ntoken]
        token_index = f_input.atom.token_index  # [B, Natom]

        # Renumber asym id to chain id: 1, 2, 5, 6, ... -> 1, 2, 3, ..., 0 0 0(pad)
        # NOTE: This is necessary for training because original asym id can be very large
        asym_id = f_input.token.asym_id
        assert (asym_id[~f_input.token.pad_mask] == -1).all(), (
            "Pad tokens must have asym_id = -1"
        )
        assert (asym_id[f_input.token.pad_mask] >= 1).all(), (
            "Non-pad tokens must have asym_id >= 1"
        )
        # Token to atom mapping
        chain_id = torch.zeros_like(asym_id)
        for b_i in range(f_input.batch_size):
            asym_id_i = asym_id[b_i]
            mask_i = token_mask[b_i]
            uniq_id = torch.sort(torch.unique(asym_id_i[mask_i]))[0]
            for c_i, a_i in enumerate(uniq_id, start=1):
                chain_id[b_i, asym_id_i == a_i] = c_i  # [B, Ntoken]

        b_i = torch.arange(f_input.batch_size, device=asym_id.device)[:, None]
        chain_id = chain_id[b_i, token_index]  # [B, Natom]
        chain_id[~atom_mask] = 0  # Set pad atoms to chain_id 0

        self.num_chains = num_chains + 1  # 0 is reserved for padding
        self.chain_id = chain_id  # [B, Natom]
        self.pad_mask = atom_mask  # [B, Natom]

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose coordinates into each chain components
        (B, *, L, 3) -> (B, *, Nchain, 3), (B, *, L, 3)
        """
        assert x.dtype == torch.float32
        B, *mid, L, _ = x.shape

        # Compute chain-wise COM
        view_shape = [B] + [1] * len(mid) + [L, 1]
        chain_id = self.chain_id.view(*view_shape).expand(x.shape)
        chain_com = torch.zeros(
            (B, *mid, self.num_chains, 3), device=x.device, dtype=x.dtype
        ).scatter_reduce_(-2, chain_id, x, reduce="mean", include_self=False)
        # Zero out COM for padding chain (chain_id=0)
        chain_com[..., 0, :] = 0.0

        x_com = chain_com.gather(-2, chain_id)  # [B, *, Natom, 3]
        x_intra = x - x_com

        # Zero out padding
        pad_mask = self.pad_mask.view(*view_shape).expand(x.shape)
        x_intra.masked_fill_(~pad_mask, 0.0)
        return chain_com, x_intra

    def recompose(self, x_com: torch.Tensor, x_intra: torch.Tensor) -> torch.Tensor:
        """Recompose coordinates from chain COM and intra components
        (B, *, Nchain, 3), (B, *, L, 3) -> (B, *, L, 3)
        """
        assert x_com.dtype == torch.float32 and x_intra.dtype == torch.float32
        B, *mid, Natom, _ = x_intra.shape
        view_shape = [B] + [1] * len(mid) + [Natom, 1]
        chain_id = self.chain_id.view(*view_shape).expand(x_intra.shape)
        pad_mask = self.pad_mask.view(*view_shape).expand(x_intra.shape)
        x_com = x_com.gather(-2, chain_id)  # [B, *, Natom, 3]
        x = x_com + x_intra
        # Zero out padding
        x.masked_fill_(~pad_mask, 0.0)
        return x


def add_com_noise(
    com: torch.Tensor,
    noise: torch.Tensor,
    tentacle_scale: float | torch.Tensor,
    radial_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Add noise to coordinates with separate tentacle and radial components for COM.

    Parameters
    ----------
    com : torch.Tensor
        Input COM coordinates. Shape (B, N, Nchain, 3).
    noise : torch.Tensor
        Noise tensor. Shape (B, N, Nchain, 3).
    tentacle_scale : float | torch.Tensor
        Scale for the tangential (tentacle) component of the COM noise.
    radial_scale : float | torch.Tensor
        Scale for the radial component of the COM noise.
    Returns
    -------
    com_noisy : torch.Tensor
        Noisy COM coordinates. Shape (B, N, Nchain, 3).
    """
    # Decompose noise into radial and tangential components
    com_norm = torch.norm(com, dim=-1, keepdim=True).clamp(min=1e-8)
    radial_direction = com / com_norm
    radial_noise = (noise * radial_direction).sum(dim=-1, keepdim=True) * radial_direction
    tangential_noise = noise - radial_noise

    # Scale and combine noise components
    noisy_com = com + radial_scale * radial_noise + tentacle_scale * tangential_noise
    return noisy_com


# === Main ECSI module implementation === #
@dataclasses.dataclass(kw_only=True)
class SICoeffs:
    """Stochastic interpolant coefficient helper for ECSI."""

    gamma_max: float
    power: float
    eta: float

    def alpha(self, t: _T) -> _T:
        return 1.0 - _clip(t) ** self.power  # type: ignore

    def alpha_deriv(self, t: _T) -> _T:
        return -self.power * (_clip(t) ** (self.power - 1))  # type: ignore

    def beta(self, t: _T) -> _T:
        return _clip(t) ** self.power  # type: ignore

    def beta_deriv(self, t: _T) -> _T:
        return self.power * _clip(t) ** (self.power - 1)  # type: ignore

    def gamma(self, t: _T) -> _T:
        t_pow = _clip(t) ** self.power
        return 0.5 * self.gamma_max * (t_pow * (1 - t_pow)) ** 0.5  # type: ignore

    def gamma_deriv(self, t: _T) -> _T:
        t_pow = _clip(t) ** self.power
        denom = (t_pow * (1 - t_pow)) ** 0.5
        coeff = self.power * t ** (self.power - 1)
        return (self.gamma_max / 4) * coeff * (1 - 2 * t_pow) / _clip(denom)  # type: ignore

    # Compute \epsilon = \eta (\gamma \dot{\gamma} - \dot{\alpha}/\alpha \gamma^2)
    def eps(self, t: _T) -> _T:
        eta = self.eta
        alpha, alpha_dot = self.alpha(t), self.alpha_deriv(t)
        gamma, gamma_dot = self.gamma(t), self.gamma_deriv(t)
        return eta * (gamma * gamma_dot - alpha_dot / _clip(alpha) * gamma**2)  # type: ignore


@dataclasses.dataclass(kw_only=True)
class SamplingScheduleConfig:
    """Maps normalized solver progress to reverse-time sampling time.

    Reverse-time sampling itself runs in time-space from `t = time_max` down to `0`.
    Internally, the scheduler first parameterizes solver progress with
    `s in [0, 1]`, then maps that progress to actual sampling time `t`.

    In solver-progress space, phase-power allocates steps across two regions:

      sde   : solver progress in [0, 1 - ode_fraction]
      tail  : solver progress in (1 - ode_fraction, 1]

    Higher `sde_power` concentrates more steps near `time_max`.
    Higher `ode_power` makes the late tail flatter near `t = 0`.
    These fields decide where the solver spends steps, not which dynamics branch is used.
    """

    global_u_power: float = 1.0
    ode_fraction: float = 0.3
    sde_power: float = 1.5
    ode_power: float = 2.6


@dataclasses.dataclass(kw_only=True)
class SamplingConfig:
    """Controls how reverse-time ECSI sampling proceeds.

    Timeline in time-space (`t: time_max -> 0`):

      1. Prior initialization
         - start from sampled prior `x_T`
         - if `perturb_x_t`, add endpoint noise with `endpoint_perturb_scale`

      2. Early high-time region (`t > churn_end_time`)
         - Apply the forward-pinned churn substep

      3. Middle stochastic region (`ode_start_time < t <= churn_end_time`)
         - use the expanded ECSI SDE update
         - stochasticity is controlled by `eta`.

      4. Late deterministic region (`t <= ode_start_time`)
         - switch to the SI ODE update
         - no diffusion noise is added in this branch

    Parameter groups:
      - horizon: `steps`, `time_min`, `time_max`
      - stochasticity: `eta`, `eta_scale_*`.
      - endpoint handling: `perturb_x_t`, `endpoint_perturb_scale`
      - early churn: `churn_end_time`, `churn_factor`
      - late ODE switch: `ode_start_time`
      - time allocation across steps: `schedule`
    """

    time_min: float = 0.0001
    time_max: float = 0.9999
    eta: float = 1.0  # global stochasticity scale
    eta_scale_intra: float = 1.0
    eta_scale_com_tentacle: float = 1.0
    eta_scale_com_radial: float = 1.0
    align_x_0_hat_to_x_t: bool = True
    perturb_x_t: bool = True
    endpoint_perturb_scale: float = 0.1
    # Early-stage Pinned churn
    churn_end_time: float = 0.7  # 1.0 to disable
    churn_factor: float = 3.0
    # Late-stage ODE switch
    ode_start_time: float = 0.6
    schedule: SamplingScheduleConfig = dataclasses.field(
        default_factory=SamplingScheduleConfig
    )


@dataclasses.dataclass(kw_only=True)
class TrainTimeSamplingConfig:
    """Configuration for train-time sampling of `t_hat`.

    Parameters
    ----------
    schedule : str
        The sampling distribution for `t_hat` during training. Options:
        - 'logit_normal': sample from `sigmoid(N(mu, sigma))` distribution.
        - 'uniform': sample uniformly from [0, 1].
        - 'beta': sample from `Beta(alpha, beta)` distribution.
    logit_normal_param : tuple[float, float], optional
        Mean and standard deviation of the underlying normal distribution.
    beta_param : tuple[float, float], optional
        Alpha and Beta parameter of the Beta distribution.
    """

    schedule: str = "logit_normal"
    logit_normal_param: tuple[float, float] = (-1.2, 1.5)
    beta_param: tuple[float, float] = (0.5, 0.5)


@STRUCTURE_MODULE.register()
class KFoldECSI(BaseECSI):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Expanded bridge dynamics with separate COM and intra-coordinate noise paths
    - Linear route with shared power k:
      \alpha_t=1-t^k and \beta_t=t^k
    - Shared base gamma with component scales:
      \gamma_t^2=\gamma_{\max}^2/4 \cdot t^k(1-t^k)
      \gamma_{\mathrm{com}, t}=s_{\mathrm{com}}\gamma_t
      \gamma_{\mathrm{int}, t}=s_{\mathrm{int}}\gamma_t
    - Stochasticity control via \eta_{com}, \eta_{int} during sampling
    - Preconditioning adapted from DDBM using the shared base gamma

    Reference:
    - ECSI: Zhang et al., "Exploring the Design Space of Diffusion Bridge Models"
    """

    class Config(BaseConfig):
        """Configuration for the ECSI structure module.

        Parameters
        ----------
        gamma_max : float, optional
            Shared base bridge maximum used by `gamma(t)`.
        time_power : float, optional
            Shared exponent `k` for the route coefficients
            `alpha_t = 1 - t^k`, `beta_t = t^k`, and the base gamma schedule.
        gamma_scale_intra : float, optional
            Multiplicative scale applied to the shared base gamma for the
            intra-coordinate branch of the expanded bridge dynamics.
        gamma_scale_com_tentacle : float, optional
            Multiplicative scale applied to the shared base gamma for the
            tangential component of the COM branch of the expanded bridge dynamics.
        gamma_scale_com_radial : float, optional
            Multiplicative scale applied to the shared base gamma for the
            radial component of the COM branch of the expanded bridge dynamics.
        sigma_data : float, optional
            Effective target-coordinate scale used in ECSI preconditioning.
        sigma_data_end : float, optional
            Effective source-coordinate scale used in ECSI preconditioning.
        cov_xy : float, optional
            Cross-covariance term between source and target coordinates used by
            the bridge preconditioning formulas.
        s_trans : float, optional
            Translation scale used by `CenterRandomAugmentation`.
        sampling : SamplingConfig, optional
            Reverse-time rollout configuration including stochasticity, endpoint
            perturbation, late ODE switching, and the nested step-allocation
            schedule.
        train_time_sampling : TrainTimeSamplingConfig, optional
            Training-time sampling policy for `t_hat`, including the optional
            Uniform mixture applied to the Beta branch.
        """

        gamma_max: float = 24.0
        time_power: float = 2.0

        gamma_scale_intra: float = 1.0
        gamma_scale_com_tentacle: float = 0.5  # Tangential noise scale for COM
        gamma_scale_com_radial: float = 0.5  # Radial noise scale for COM

        sigma_data: float = 16.0
        sigma_data_end: float = 66.0  # 16 + 50 translations
        cov_xy: float = 128.0

        s_trans: float = 0.0

        sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
        train_time_sampling: TrainTimeSamplingConfig = dataclasses.field(
            default_factory=TrainTimeSamplingConfig
        )

    def __init__(self, cfg: Config, score_model: ECSIDiffusionModule):
        """Initialize the ECSI module.

        The constructor copies the high-level config fields onto runtime
        attributes, then immediately validates and normalizes the nested
        sampling config so downstream code can assume a runtime-ready ECSI
        configuration.
        """
        super().__init__(cfg, score_model)
        self.__validate_config(cfg)

        self.score_model: ECSIDiffusionModule = score_model
        self.sampling: SamplingConfig = cfg.sampling
        self.train_time_sampling: TrainTimeSamplingConfig = cfg.train_time_sampling

        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy

        self.si_coeffs = SICoeffs(
            gamma_max=cfg.gamma_max,
            power=cfg.time_power,
            eta=cfg.sampling.eta,
        )
        self.gamma_scale_intra = cfg.gamma_scale_intra
        self.gamma_scale_com_t = cfg.gamma_scale_com_tentacle
        self.gamma_scale_com_r = cfg.gamma_scale_com_radial
        self.eta_scale_intra = cfg.sampling.eta_scale_intra
        self.eta_scale_com_t = cfg.sampling.eta_scale_com_tentacle
        self.eta_scale_com_r = cfg.sampling.eta_scale_com_radial

        # NOTE: centering should be disabled.
        self.random_augmentation = CenterRandomAugmentation(
            centering=False, s_trans=cfg.s_trans
        )

    @staticmethod
    def __validate_config(config: Config) -> None:
        """Validate and normalize runtime config used by ECSI."""
        sampling_cfg = config.sampling
        if sampling_cfg.time_max <= sampling_cfg.time_min:
            raise ValueError("sampling.time_max must be > sampling.time_min")
        if not (
            sampling_cfg.time_min
            < sampling_cfg.ode_start_time
            < sampling_cfg.churn_end_time
            < sampling_cfg.time_max
        ):
            raise ValueError(
                "sampling requires time_min < ode_time < churn_until_time < time_max"
            )

        train_time_cfg = config.train_time_sampling
        if train_time_cfg.schedule not in {"logit_normal", "uniform", "beta"}:
            raise ValueError(
                "train_time_sampling.schedule must be one of logit_normal, uniform, beta"
            )

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
        c_skip : float | torch.Tensor
            Skip connection coefficient.
        c_out : float | torch.Tensor
            Output scaling coefficient.
        c_in : float | torch.Tensor
            Input scaling coefficient.
        """
        alpha_t = self.si_coeffs.alpha(t)
        beta_t = self.si_coeffs.beta(t)
        gamma_t = self.si_coeffs.gamma(t)

        sigma_data = self.sigma_data
        sigma_data_end = self.sigma_data_end
        cov_xy = self.cov_xy

        # Total variance A (adapted from DDBM Eq. 81)
        # A = \alpha_t^2 \sigma_0^2 + \beta_t^2 \sigma_T^2
        #   + 2 \alpha_t \beta_t \sigma_{0T} + \gamma_t^2
        A = _clip(
            alpha_t**2 * sigma_data**2
            + beta_t**2 * sigma_data_end**2
            + 2 * alpha_t * beta_t * cov_xy
            + gamma_t**2
        )

        # c_in: input normalization
        c_in = 1 / A**0.5

        # c_skip: skip connection weight
        c_skip = (alpha_t * sigma_data**2 + beta_t * cov_xy) / A

        # c_out: output scaling
        numerator_out = _sqrt(
            beta_t**2 * (sigma_data**2 * sigma_data_end**2 - cov_xy**2)
            + gamma_t**2 * sigma_data**2
        )

        c_out = c_in * numerator_out
        if isinstance(c_out, torch.Tensor):
            c_skip, c_out, c_in = c_skip.float(), c_out.float(), c_in.float()
        return c_skip, c_out, c_in

    def c_skip(self, t: _T) -> _T:
        r"""Skip connection coefficient for ECSI preconditioning."""
        c_skip, _, _ = self._get_bridge_scalings(t)
        return c_skip

    def c_out(self, t: _T) -> _T:
        r"""Output scaling coefficient for ECSI preconditioning."""
        _, c_out, _ = self._get_bridge_scalings(t)
        return c_out

    def c_in(self, t: _T) -> _T:
        r"""Input scaling coefficient for ECSI preconditioning."""
        _, _, c_in = self._get_bridge_scalings(t)
        return c_in

    def c_noise(self, t: _T) -> _T:
        r"""Noise level conditioning coefficient.

        Maps t to a conditioning value for the network.
        Uses log-scaling similar to EDM.
        """
        return 0.25 * _log(t)

    # ============================================================
    # For training
    # ============================================================
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        diffusion_batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        batch_size = f_input.batch_size  # =B
        num_samples = diffusion_batch_size  # =N
        mask = f_input.atom.pad_mask  # [B, Natom]
        device = f_input.device
        decomposer = ChainDecomposition(f_input)

        t_hat = self.sample_noise_level((batch_size, num_samples), device)  # [B, N]

        # sample xT from label
        x_0, x_T = self.sample_x_0_and_x_T(f_input, num_samples)

        # sample xt via interpolation
        x_t = self.interpolate(x_0, x_T, t_hat, mask, decomposer)  # [B, N, Natom, 3]

        x_0_hat = self.forward_train(
            x_t=x_t,  # [B, N, Natom, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            x_T=x_T,  # [B, N, Natom, 3]
        )  # [B, N, Natom, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "x_t": x_t,
            "x_T": x_T,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }

    def forward_train(
        self,
        x_t: torch.Tensor,
        t_hat: torch.Tensor,
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
        t_hat : torch.Tensor
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
        c_skip, c_out, c_in = self._get_bridge_scalings(t_hat)  # [B, N]
        c_noise = self.c_noise(t_hat)  # [B, N]

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

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        r"""Compute loss weights based on time t_hat.

        Uses Karras-style weighting: w(t) = 1 / c_{out}(t)^2
        """
        c_out = self.c_out(t_hat)
        weights = 1 / c_out.pow(2).clamp(min=1e-8)
        return weights

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
        schedule = self.train_time_sampling.schedule
        match schedule:
            case "logit_normal":
                mu, sigma = self.train_time_sampling.logit_normal_param
                x = torch.randn(shape, device=device)
                t = torch.sigmoid(mu + sigma * x)
            case "uniform":
                # Uniform sampling
                t = torch.rand(shape, device=device)
            case "beta":
                # Beta sampling branch.
                alpha, beta = self.train_time_sampling.beta_param
                m = torch.distributions.Beta(alpha, beta)
                t = m.sample(shape).to(device)
            case _:
                raise ValueError(f"Unsupported train_time_sampling.schedule: {schedule}")
        # Scale to [sampling_time_min, sampling_time_max]
        t = self.sampling.time_min + (self.sampling.time_max - self.sampling.time_min) * t
        return t

    def sample_x_0_and_x_T(
        self, f_input: FoldingInput, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x0 (labels) and xT (prior) coordinates for ECSI training.
        To maintain the relative alignment between x0 and xT,  use specific
        method to augment them together.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_samples : int
            Number of diffusion samples

        Returns
        -------
        x_0 : torch.Tensor
            Label coordinates. Shape (B, N, Natom, 3).
        prior_coords : torch.Tensor
            prior coordinates. Shape (B, N, Natom, 3).
        """
        # Sample from label coordinates
        x_0 = f_input.atom.label_coords[..., None, :, :]  # [B, 1, Natom, 3]
        label_mask = f_input.atom.resolved_mask[..., None, :]  # [B, 1, Natom]
        mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Natom]

        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        all_prior_coords = f_input.atom.prior_coords  # [B, Natom, Nprior, 3]
        num_prior = all_prior_coords.shape[-2]
        idx = [i % num_prior for i in range(num_samples)]
        x_T = all_prior_coords[:, :, idx, :]  # [B, Natom, N, 3]
        x_T = x_T.permute(0, 2, 1, 3)  # [B, N, Natom, 3]

        # repeat label coords
        x_0 = x_0.expand(-1, num_samples, -1, -1)  # [B, N, L, 3]

        # Apply random augmentation to label and prior coords altogether
        # to maintain their relative alignment.
        # Note: mask is not used if mask_to_zero=False, so pass any mask here.
        x_0, x_T = self.random_augmentation(
            x_0, x_T, mask=mask, centering=False, mask_to_zero=False
        )
        # Manual masking after augmentation (This is not required process)
        x_0.masked_fill_(~label_mask[..., None], 0.0)
        x_T.masked_fill_(~mask[..., None], 0.0)
        return x_0, x_T

    def interpolate(  # type: ignore
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
        decomposer: ChainDecomposition,
    ) -> torch.Tensor:
        r"""Interpolate between x_0 and x_T using ECSI bridge.

        Samples from the bridge distribution:
        x_t = \alpha_t x_0 + \beta_t x_T + \gamma_t z, where z ~ N(0, I)

        Parameters
        ----------
        x_0 : torch.Tensor
            The target (holo) coordinates x_0. Shape (B, N, Natom, 3).
        x_T : torch.Tensor
            The source (apo) coordinates x_T. Shape (B, N, Natom, 3).
        t_hat : torch.Tensor
            Time values. Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, Natom).
        decomposer : ChainDecomposition
            ChainDecomposition object for decomposing/recomposing coordinates
            into COM/intra space for ECSI interpolation.

        Returns
        -------
        x_t : torch.Tensor
            Bridge-sampled coordinates x_t. Shape (B, N, Natom, 3).
        """
        B, N, Natom, _ = x_0.shape
        t_expanded = t_hat[:, :, None, None]
        alpha_t = self.si_coeffs.alpha(t_expanded)
        beta_t = self.si_coeffs.beta(t_expanded)
        gamma = self.si_coeffs.gamma(t_expanded)

        gamma_intra = gamma * self.gamma_scale_intra
        gamma_com_t = gamma * self.gamma_scale_com_t
        gamma_com_r = gamma * self.gamma_scale_com_r

        # Interpolate in COM/intra space
        x_0_com, x_0_intra = decomposer.decompose(x_0)
        x_T_com, x_T_intra = decomposer.decompose(x_T)

        # Add noise in COM/intra space
        noise_com = torch.randn_like(x_0_com)
        _, noise_intra = decomposer.decompose(torch.randn_like(x_T))

        _mu = alpha_t * x_0_com + beta_t * x_T_com
        x_t_com = add_com_noise(_mu, noise_com, gamma_com_t, gamma_com_r)
        _mu = alpha_t * x_0_intra + beta_t * x_T_intra
        x_t_intra = _mu + gamma_intra * noise_intra

        # Recompose to Cartesian coordinates
        x_t = decomposer.recompose(x_t_com, x_t_intra)
        return x_t

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
        # Construct chain decomposer
        decomposer = ChainDecomposition(f_input)

        # Get time schedule (from t_max toward t_min)
        times = self.get_sampling_schedule(num_steps)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)

        if self.sampling.perturb_x_t:
            x_t = self._apply_endpoint_perturbation(x_t, mask)
        x_churn_target = x_t.clone()

        # Compute time-independent variables
        z = self.get_pair_conditioning(f_input, z_trunk)
        q, c, p = self.get_atom_embeddings(f_input, s_inputs, s_trunk, z)
        pair_bias = self.get_pair_bias(z)
        del z_trunk, z  # Free up memory for large LxL tensors

        def run_step(x_t: torch.Tensor, t_hat: float) -> torch.Tensor:
            s = self.get_single_conditioning(s_inputs, s_trunk, t_hat)
            return self.inference_step(
                f_input=f_input,
                x_t=x_t,
                x_T=x_T,
                t_hat=t_hat,
                q=q,
                c=c,
                p=p,
                s=s,
                pair_bias=pair_bias,
                chunk_size=chunk_size,
            )

        traj: list[torch.Tensor] = []

        def append_traj(x_t: torch.Tensor):
            if return_traj:
                traj.append(x_t.cpu())

        # Time schedule and branch control parameters
        churn_end_time: float = self.sampling.churn_end_time
        ode_start_time: float = self.sampling.ode_start_time

        # Sampling loop
        for step_idx in range(num_steps):
            append_traj(x_t)

            # Apply random augmentation
            x_t, x_T, x_churn_target = self.random_augmentation(
                x_t, x_T, x_churn_target, mask=mask
            )

            t = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t  # negative since t decreases

            if churn_end_time < t:
                # Early-stage forward-pinned churn.
                x_t, t = self._apply_forward_pinned_churn(
                    x_t, x_churn_target, decomposer, t, t_next
                )
                dt = t_next - t

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_t, t)

            if self.sampling.align_x_0_hat_to_x_t:
                # Rigidly align x_0_hat to x_t before centering.
                x_0_hat = rigid_align(x_0_hat, x_t, mask)

            # Centering c_0_hat
            x_0_hat = do_centering(x_0_hat, mask)

            if t > ode_start_time:
                # Early/Mid-stage stochastic SI SDE update.
                x_t = self._sde_step(x_t, x_0_hat, x_T, decomposer, t, dt)
            else:
                # Late-stage deterministic ODE update.
                x_t = self._ode_step(x_t, x_0_hat, mask, t, dt)

        append_traj(x_t)

        sample_out: dict[str, torch.Tensor] = {}
        sample_out["init_coordinates"] = x_T
        sample_out["sample_coordinates"] = x_t
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
        xT : torch.Tensor
            prior coordinates. Shape (B, N, Natom, 3).

        """
        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        all_prior_coords = f_input.atom.prior_coords  # [B, Natom, Nprior, 3]
        num_prior = all_prior_coords.shape[-2]
        idx = [i % num_prior for i in range(num_samples)]
        xT = all_prior_coords[:, :, idx, :]  # [B, Natom, N, 3]
        xT = xT.permute(0, 2, 1, 3)  # [B, N, Natom, 3]

        # Apply random augmentation to prior coords without centering.
        mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Natom]
        xT = self.random_augmentation(xT, mask=mask, mask_to_zero=True, centering=False)
        return xT

    def inference_step(
        self,
        f_input: FoldingInput,
        x_t: torch.Tensor,
        x_T: torch.Tensor,
        t_hat: float,
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
        t_hat : float
            Diffusion noise level (or sigmas of EDM).
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

        # Input preconditioning: r_noisy = c_in * x_t
        r_noisy = self.c_in(t_hat) * x_t

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
        x_out = self.c_skip(t_hat) * x_t + self.c_out(t_hat) * r_update
        return x_out

    def get_pair_conditioning(
        self, f_input: FoldingInput, z_trunk: torch.Tensor
    ) -> torch.Tensor:
        """Get the pair conditioning for the score model.
        See Section 3.7: Algorithm 21 of AlphaFold3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, Lt, c_z].

        Returns
        -------
        z : torch.Tensor
            The pair conditioning, shape [B, Lt, Lt, c_z].
        """
        return self.score_model.get_pair_conditioning(f_input, z_trunk)

    def get_atom_embeddings(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the inputs which are static across diffusion steps.
        # Algorithm 5 Line 1-10, 13-14.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        z : torch.Tensor
            The trunk pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        q : torch.Tensor
            The atom single representation, shape [B, Natom, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, Natom, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, Natom, Natom, c_atompair].
        """
        return self.score_model.get_atom_embeddings(f_input, s_inputs, s_trunk, z)

    def get_pair_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Get the pair bias for the token transformer.
        This is time-independent and can be pre-computed before the diffusion steps.

        Parameters
        ----------
        z : torch.Tensor
            The pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].
        """
        return self.score_model.get_pair_bias(z)

    def get_single_conditioning(
        self, s_inputs: torch.Tensor, s_trunk: torch.Tensor, t_hat: float
    ) -> torch.Tensor:
        """Get the single conditioning for the score model.
        See Section 3.7: Algorithm 21 of AlphaFold3 paper.

        Parameters
        ----------
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        t_hat : float
            Diffusion noise level (or sigma).

        Returns
        -------
        s : torch.Tensor
            The single conditioning, shape [B, 1, Lt, c_s].
        """
        c_noise = self.c_noise(t_hat)
        c_noise = torch.tensor(c_noise, device=s_inputs.device).view(1, 1)
        return self.score_model.get_single_conditioning(s_inputs, s_trunk, c_noise)

    # ============================================================
    # Sampling schedule
    # ============================================================
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
        sampling = self.sampling
        time_max = sampling.time_max
        time_min = sampling.time_min
        ode_time = sampling.ode_start_time
        total_scale = time_max - time_min

        schedule = self.sampling.schedule
        global_u_power = schedule.global_u_power

        ode_power = schedule.ode_power

        ode_unit = (ode_time - time_min) / total_scale
        ode_u = ode_unit**global_u_power

        s = np.linspace(0, 1, num_steps)
        sde_fraction = 1.0 - schedule.ode_fraction
        ode_fraction = schedule.ode_fraction
        sde_mask = s <= sde_fraction
        ode_mask = s > sde_fraction

        u = np.zeros_like(s)
        if sde_mask.any():
            progress = s[sde_mask] / _clip(sde_fraction)
            _u = 1.0 - (1.0 - ode_u) * progress**schedule.sde_power
            u[sde_mask] = _u
        if ode_mask.any():
            progress = (1.0 - s[ode_mask]) / _clip(ode_fraction)
            _u = ode_u * progress**ode_power
            u[ode_mask] = _u

        t_unit = u.clip(0.0, 1.0) ** (1.0 / global_u_power)
        times = time_min + total_scale * t_unit
        times = times.tolist()
        # Append time=0.0 at the end to ensure the final step reaches t=0.
        times.append(0.0)
        return times

    # ============================================================
    # Sampling utilities
    # ============================================================
    def _apply_endpoint_perturbation(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sampling.endpoint_perturb_scale
        noise.masked_fill_(~mask[..., None], 0.0)
        return x + noise

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_churn_target: torch.Tensor,
        decomposer: ChainDecomposition,
        t: float,
        t_next: float,
    ) -> tuple[torch.Tensor, float]:
        if t <= self.sampling.churn_end_time:
            # No churn applied before or at churn_end
            return x_t, t

        dt = abs(t_next - t)
        dt_churn = self.sampling.churn_factor * dt
        if t + dt_churn >= self.sampling.time_max:
            # Do not apply churn
            return x_t, t

        alpha_t: float = self.si_coeffs.alpha(t)
        beta_t: float = self.si_coeffs.beta(t)
        alpha_dot: float = self.si_coeffs.alpha_deriv(t)
        beta_dot: float = self.si_coeffs.beta_deriv(t)
        gamma = self.si_coeffs.gamma(t)
        gamma_dot = self.si_coeffs.gamma_deriv(t)

        f_t = alpha_dot / alpha_t
        s_t = beta_dot - f_t * beta_t
        g = _sqrt(2 * (gamma * gamma_dot - f_t * gamma**2))

        # Decompose coordinates into COM/intra space
        x_t_com, x_t_intra = decomposer.decompose(x_t)
        x_target_com, x_target_intra = decomposer.decompose(x_churn_target)

        # Sample noise for COM and intra space
        noise_com = torch.randn_like(x_t_com)
        _, noise_intra = decomposer.decompose(torch.randn_like(x_t))

        # Euler update with forward-pinned noise
        g_com_t = g * self.gamma_scale_com_t
        g_com_r = g * self.gamma_scale_com_r
        g_intra = g * self.gamma_scale_intra
        mean_com = x_t_com + (f_t * x_t_com + s_t * x_target_com) * dt_churn
        mean_intra = x_t_intra + (f_t * x_t_intra + s_t * x_target_intra) * dt_churn
        x_t_com = add_com_noise(mean_com, noise_com * (dt_churn**0.5), g_com_t, g_com_r)
        x_t_intra = mean_intra + g_intra * noise_intra * (dt_churn**0.5)

        # Recompose to Cartesian coordinates
        x_t = decomposer.recompose(x_t_com, x_t_intra)
        t += dt_churn
        return x_t, t

    def _sde_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        decomposer: ChainDecomposition,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        si_coeffs = self.si_coeffs
        alpha_t: float = si_coeffs.alpha(t)
        beta_t: float = si_coeffs.beta(t)
        alpha_dot: float = si_coeffs.alpha_deriv(t)
        beta_dot: float = si_coeffs.beta_deriv(t)
        gamma: float = si_coeffs.gamma(t)
        gamma_dot: float = si_coeffs.gamma_deriv(t)
        eps: float = si_coeffs.eps(t)

        # Scale eps for COM/intra space separately
        eps_com_r = self.eta_scale_com_r * eps
        eps_com_t = self.eta_scale_com_t * eps
        eps_intra = self.eta_scale_intra * eps

        # Decompose coordinates into COM/intra space
        x_t_com, x_t_intra = decomposer.decompose(x_t)
        x_0_com, x_0_intra = decomposer.decompose(x_0_hat)
        x_T_com, x_T_intra = decomposer.decompose(x_T)

        # Compute \hat{z}_t
        # \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / gamma
        raw_z_com = x_t_com - alpha_t * x_0_com - beta_t * x_T_com
        _com_norm = x_t_com.norm(dim=-1, keepdim=True)
        _r_dir = x_t_com / (_com_norm + 1e-8)
        raw_z_com_r = (raw_z_com * _r_dir).sum(dim=-1, keepdim=True) * _r_dir
        raw_z_com_t = raw_z_com - raw_z_com_r
        del raw_z_com, _com_norm, _r_dir  # Free up memory
        raw_z_intra = x_t_intra - alpha_t * x_0_intra - beta_t * x_T_intra

        base_z_com_r = raw_z_com_r / gamma
        base_z_com_t = raw_z_com_t / gamma
        base_z_intra = raw_z_intra / gamma

        # Compute drift b(t)
        # b(t) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
        #        + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        drift_com = (
            (alpha_dot * x_0_com + beta_dot * x_T_com)
            + ((gamma_dot + eps_com_r / gamma) * base_z_com_r)
            + ((gamma_dot + eps_com_t / gamma) * base_z_com_t)
        )
        drift_intra = (alpha_dot * x_0_intra + beta_dot * x_T_intra) + (
            (gamma_dot + eps_intra / gamma) * base_z_intra
        )

        # === Euler-Maruyama update ===
        # Compute noise for COM and intra space
        noise_com = torch.randn_like(x_t_com)
        _, noise_intra = decomposer.decompose(torch.randn_like(x_t))

        # Update in COM/intra space
        noise_scale_com_t = self.gamma_scale_com_t * abs(2 * eps_com_t * dt) ** 0.5
        noise_scale_com_r = self.gamma_scale_com_r * abs(2 * eps_com_r * dt) ** 0.5
        noise_scale_intra = self.gamma_scale_intra * abs(2 * eps_intra * dt) ** 0.5
        mean_com = x_t_com + drift_com * dt
        mean_intra = x_t_intra + drift_intra * dt
        x_com = add_com_noise(mean_com, noise_com, noise_scale_com_t, noise_scale_com_r)
        x_intra = mean_intra + noise_scale_intra * noise_intra

        # Recompose to Cartesian coordinates
        x_update = decomposer.recompose(x_com, x_intra)

        return x_update

    def _ode_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        coeffs = self.si_coeffs
        alpha, beta = coeffs.alpha(t), coeffs.beta(t)
        alpha_next, beta_next = coeffs.alpha(t + dt), coeffs.beta(t + dt)
        c_skip: float = beta_next / beta
        c_update: float = alpha_next - alpha * c_skip
        x_update = c_skip * x_t + c_update * x_0_hat
        x_update.masked_fill_(~mask[..., None], 0.0)
        return x_update
