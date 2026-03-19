# Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI)
# Based on "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553)
# Adapted from ECSI training code and kfold_ddbm.py

import math
from typing import Protocol

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import CenterRandomAugmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseECSI


class _Route(Protocol):
    route_type: str

    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor: ...

    def beta(self, t: torch.Tensor) -> torch.Tensor: ...

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor: ...

    def gamma(self, t: torch.Tensor) -> torch.Tensor: ...

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor: ...


class _LinearRoute:
    route_type = "linear"

    def __init__(self, gamma_max: float, power: float) -> None:
        self.gamma_max = gamma_max
        self.power = power

    @staticmethod
    def _clamp_t(t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=1e-8)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        t_pow = torch.pow(t_clamped, self.power)
        return 1 - t_pow

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        coeff = self.power * torch.pow(t_clamped, self.power - 1)
        return -coeff

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        return torch.pow(t_clamped, self.power)

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        return self.power * torch.pow(t_clamped, self.power - 1)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        t_pow = torch.pow(t_clamped, self.power)
        return 0.5 * self.gamma_max * torch.sqrt(t_pow * (1 - t_pow) + 1e-8)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        t_pow = torch.pow(t_clamped, self.power)
        denom = torch.sqrt(t_pow * (1 - t_pow) + 1e-8)
        coeff = self.power * torch.pow(t_clamped, self.power - 1)
        return (self.gamma_max / 4) * coeff * (1 - 2 * t_pow) / (denom + 1e-8)


class _DdbmVpRoute:
    route_type = "ddbm_vp"

    def __init__(self, beta_min: float, beta_d: float) -> None:
        self.beta_min = beta_min
        self.beta_d = beta_d

        exponent = 0.5 * beta_d + beta_min
        self.a1 = math.exp(exponent) ** -0.5
        self.sigma1_sq = math.exp(exponent) - 1.0

    def _constants(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = t.new_tensor(self.a1)
        sigma1_sq = t.new_tensor(self.sigma1_sq)
        return a1, sigma1_sq

    def _base(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_d = t.new_tensor(self.beta_d)
        beta_min = t.new_tensor(self.beta_min)
        log_snr = 0.5 * beta_d * t**2 + beta_min * t
        exp_term = torch.exp(log_snr)
        sigma_sq = exp_term - 1.0
        sigma_sq_prime = exp_term * (beta_d * t + beta_min)
        a_t = torch.rsqrt(exp_term)
        a_t_prime = -0.5 * (beta_d * t + beta_min) * a_t
        return a_t, a_t_prime, sigma_sq, sigma_sq_prime

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._base(t)
        a1, sigma1_sq = self._constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        denom = sigma1_sq * a_t_sq + 1e-8
        ratio = sigma_sq * a1_sq / denom
        return a_t * (1 - ratio)

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._base(t)
        a1, sigma1_sq = self._constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        a_t_sq_prime = 2 * a_t * a_t_prime
        inv_a_t_sq = 1 / (a_t_sq + 1e-8)
        k = a1_sq / (sigma1_sq + 1e-8)
        ratio = k * sigma_sq * inv_a_t_sq
        ratio_prime = k * (
            sigma_sq_prime * inv_a_t_sq
            - sigma_sq * a_t_sq_prime * inv_a_t_sq * inv_a_t_sq
        )
        return a_t_prime * (1 - ratio) - a_t * ratio_prime

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._base(t)
        a1, sigma1_sq = self._constants(t)
        denom = sigma1_sq * a_t + 1e-8
        return sigma_sq * a1 / denom

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._base(t)
        a1, sigma1_sq = self._constants(t)
        inv_a_t = 1 / (a_t + 1e-8)
        k = a1 / (sigma1_sq + 1e-8)
        return k * (sigma_sq_prime * inv_a_t - sigma_sq * a_t_prime * inv_a_t * inv_a_t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._base(t)
        a1, sigma1_sq = self._constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        ratio = sigma_sq * a1_sq / (sigma1_sq * a_t_sq + 1e-8)
        gamma_sq = sigma_sq * (1 - ratio)
        return torch.sqrt(torch.clamp(gamma_sq, min=0.0) + 1e-8)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._base(t)
        a1, sigma1_sq = self._constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        a_t_sq_prime = 2 * a_t * a_t_prime
        inv_a_t_sq = 1 / (a_t_sq + 1e-8)
        k = a1_sq / (sigma1_sq + 1e-8)
        ratio = k * sigma_sq * inv_a_t_sq
        ratio_prime = k * (
            sigma_sq_prime * inv_a_t_sq
            - sigma_sq * a_t_sq_prime * inv_a_t_sq * inv_a_t_sq
        )
        gamma_sq = sigma_sq * (1 - ratio)
        gamma_sq_prime = sigma_sq_prime * (1 - ratio) - sigma_sq * ratio_prime
        gamma = torch.sqrt(torch.clamp(gamma_sq, min=0.0) + 1e-8)
        return 0.5 * gamma_sq_prime / (gamma + 1e-8)


@STRUCTURE_MODULE.register()
class KFoldECSI(BaseECSI):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Decoupled kernel parameters (\alpha_t, \beta_t, \gamma_t) for flexible bridge paths
    - Linear route with shared power k:
      \alpha_t=1-t^k, \beta_t=t^k,
      \gamma_t^2=\gamma_{max}^2/4 \cdot t^k(1-t^k)
    - DDBM-VP route (Appendix C.2): configurable via route_type="ddbm_vp"
    - Stochasticity control via \eta parameter during sampling
    - Preconditioning adapted from DDBM

    Reference:
    - ECSI: Zhang et al., "Exploring the Design Space of Diffusion Bridge Models"
    - DDBM: Zhou et al., "Denoising Diffusion Bridge Models"
    """

    class Config(BaseConfig):
        r"""Configuration for the ECSI Structure module.

        Parameters
        ----------
        num_steps : int, optional
            The number of sampling steps, by default 200.
        sigma_min : float, optional
            Minimum time value (near t=0), by default 0.001.
        sigma_max : float, optional
            Maximum time value (near t=T), by default 0.999.
        gamma_max : float, optional
            Scale parameter for \gamma_t, by default 1.0.
            Uses \gamma_t^2 = \gamma_{max}^2/4 * t^k(1-t^k).
        time_power : float, optional
            Shared exponent k for linear route coefficients.
            Uses \alpha_t=1-t^k, \beta_t=t^k, and
            \gamma_t^2=\gamma_{max}^2/4 * t^k(1-t^k), by default 2.0.
        route_type : str, optional
            Route selection for (\alpha_t, \beta_t, \gamma_t). Options: "linear"
            (default) or "ddbm_vp".
        ddbm_vp_beta_min : float, optional
            DDBM-VP beta_min parameter for \sigma_t and a_t schedules.
        ddbm_vp_beta_d : float, optional
            DDBM-VP beta_d parameter for \sigma_t and a_t schedules.
        sigma_data : float, optional
            Standard deviation of target (holo) distribution, by default 16.0.
        sigma_data_end : float, optional
            Standard deviation of source (apo) distribution, by default 16.0.
            Uses physical coordinate scale (not normalized to image-like variance).
        cov_xy : float, optional
            Covariance between source and target distributions, by default 128.0.
            Controls the correlation structure in preconditioning.
        rho : int, optional
            The rho value for Karras schedule, by default 7.
        P_mean : float, optional
            Mean for log-normal noise level sampling, by default -1.2.
        P_std : float, optional
            Standard deviation for log-normal noise level sampling, by default 1.5.
        eta : float, optional
            Stochasticity control parameter, by default 1.0.
            \eta=0 gives deterministic ODE, \eta=1 gives full SDE sampling.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        normalize_data_end : bool, optional
            Whether to normalize the source (apo) input, by default False.
        normalize_coordinate : bool, optional
            Whether to normalize the source and target coordinates, by default False.
        alignment_entity_strategy : str | None, optional
            Strategy for selecting entity to align: None (all entities), "largest",
            or "random_non_ligand", by default "largest".
        """

        num_steps: int = 200
        sigma_min: float = 0.001
        sigma_max: float = 0.999
        gamma_max: float = 0.25
        time_power: float = 2.0
        route_type: str = "linear"
        ddbm_vp_beta_min: float = 0.1
        ddbm_vp_beta_d: float = 16.0
        sigma_data: float = 16.0
        sigma_data_end: float = 16.0
        cov_xy: float = 128.0
        rho: float = 0.7
        sampling_schedule_type: str = "piecewise_power"
        sampling_schedule_piecewise_power: float = 5.0
        sampling_schedule_start_power: float | None = None
        sampling_schedule_end_power: float | None = None
        sampling_schedule_midpoint: float = 0.5
        sampling_schedule_endpoint_trim: float = 0.0
        sampling_schedule_global_u_power: float = 1.0
        sampling_schedule_churn_fraction: float = 0.3
        sampling_schedule_ode_fraction: float = 0.45
        sampling_schedule_middle_power: float = 1.0
        sampling_schedule_churn_power: float = 1.75
        sampling_schedule_ode_power: float = 2.6
        P_mean: float = -1.2
        P_std: float = 1.5
        eta: float = 1.0
        coordinate_augmentation: bool = True
        normalize_data_end: bool = False
        normalize_coordinate: bool = False
        logit_normal_sampling: bool = False
        sampling_alpha: float = 1.0
        sampling_beta: float = 1.0
        use_prior_coords: bool = True
        alignment_entity_strategy: str | None = None
        alignment_level: str = "chain"
        s_trans: float = 1.0
        inference_align_x0_hat_to_x_t: bool = True
        perturb_xt: bool = True
        endpoint_perturb_scale: float | None = 0.1
        ode_time_duration: float = 0.6
        use_forward_pinned_churn: bool = True
        churn_factor: float = 3.0
        churn_until_time: float | None = 0.7

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the ECSI module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.gamma_max: float = cfg.gamma_max
        self.time_power: float = cfg.time_power
        self.route_type: str = cfg.route_type
        self.ddbm_vp_beta_min: float = cfg.ddbm_vp_beta_min
        self.ddbm_vp_beta_d: float = cfg.ddbm_vp_beta_d
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy
        self.rho: int = cfg.rho
        self.sampling_schedule_type: str = cfg.sampling_schedule_type
        self.sampling_schedule_piecewise_power: float = (
            cfg.sampling_schedule_piecewise_power
        )
        self.sampling_schedule_start_power: float | None = (
            cfg.sampling_schedule_start_power
        )
        self.sampling_schedule_end_power: float | None = cfg.sampling_schedule_end_power
        self.sampling_schedule_midpoint: float = cfg.sampling_schedule_midpoint
        self.sampling_schedule_endpoint_trim: float = cfg.sampling_schedule_endpoint_trim
        self.sampling_schedule_global_u_power: float = (
            cfg.sampling_schedule_global_u_power
        )
        self.sampling_schedule_churn_fraction: float = (
            cfg.sampling_schedule_churn_fraction
        )
        self.sampling_schedule_ode_fraction: float = cfg.sampling_schedule_ode_fraction
        self.sampling_schedule_middle_power: float = cfg.sampling_schedule_middle_power
        self.sampling_schedule_churn_power: float = cfg.sampling_schedule_churn_power
        self.sampling_schedule_ode_power: float = cfg.sampling_schedule_ode_power
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.eta: float = cfg.eta
        self.num_steps: int = cfg.num_steps
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation
        self.normalize_data_end: bool = cfg.normalize_data_end
        self.normalize_coordinate: bool = cfg.normalize_coordinate
        self.logit_normal_sampling: bool = cfg.logit_normal_sampling
        self.sampling_alpha: float = cfg.sampling_alpha
        self.sampling_beta: float = cfg.sampling_beta
        self.use_prior_coords: bool = cfg.use_prior_coords
        self.s_trans: float = cfg.s_trans
        self.alignment_level: str = cfg.alignment_level
        self.inference_align_x0_hat_to_x_t: bool = cfg.inference_align_x0_hat_to_x_t
        self.perturb_xt: bool = cfg.perturb_xt
        self.endpoint_perturb_scale: float | None = cfg.endpoint_perturb_scale
        self.ode_time_duration: float = cfg.ode_time_duration
        self.use_forward_pinned_churn: bool = cfg.use_forward_pinned_churn
        self.churn_factor: float = cfg.churn_factor
        self.churn_until_time: float | None = cfg.churn_until_time

        self._route: _Route
        self._configure_route_functions(cfg)

        self.random_augmentation = CenterRandomAugmentation(
            centering=True,
            augmentation=self.coordinate_augmentation,
            s_trans=self.s_trans,
        )

    @property
    def _effective_sigma_data(self) -> float:
        return 1.0 if self.normalize_coordinate else self.sigma_data

    @property
    def _effective_sigma_data_end(self) -> float:
        return 1.0 if self.normalize_coordinate else self.sigma_data_end

    @property
    def _effective_cov_xy(self) -> float:
        if self.normalize_coordinate:
            return self.cov_xy / (self.sigma_data * self.sigma_data_end)
        return self.cov_xy

    def apply_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to coordinates."""
        return self.random_augmentation(coords, mask=mask)

    def _configure_route_functions(self, cfg: Config) -> None:
        route = (cfg.route_type or "linear").lower().replace("-", "_")
        self.route_type = route
        if route == "linear":
            self._route = _LinearRoute(
                gamma_max=self.gamma_max,
                power=self.time_power,
            )
            return
        if route == "ddbm_vp":
            self._route = _DdbmVpRoute(
                beta_min=self.ddbm_vp_beta_min,
                beta_d=self.ddbm_vp_beta_d,
            )
            return
        raise ValueError(
            "Unsupported route_type; expected 'linear' or 'ddbm_vp', "
            f"got {cfg.route_type!r}."
        )

    # === Route Functions (Stochastic Interpolants) === #
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for target (x_0/holo); route selected by config."""
        return self._route.alpha(t)

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of alpha; route selected by config."""
        return self._route.alpha_deriv(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for source (x_T/apo); route selected by config."""
        return self._route.beta(t)

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of beta; route selected by config."""
        return self._route.beta_deriv(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Noise scale; route selected by config."""
        return self._route.gamma(t)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of gamma; route selected by config."""
        return self._route.gamma_deriv(t)

    # === Bridge Preconditioning Coefficients === #
    def _get_bridge_scalings(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bridge diffusion scalings for ECSI.

        Adapted from DDBM formulation using ECSI's decoupled parameterization.

        Parameters
        ----------
        t : torch.Tensor
            Time values. Shape (B, N) or scalar, in range [0, 1].

        Returns
        -------
        c_skip : torch.Tensor
            Skip connection coefficient.
        c_out : torch.Tensor
            Output scaling coefficient.
        c_in : torch.Tensor
            Input scaling coefficient.
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        gamma_t = self.gamma(t)

        sigma_data = self._effective_sigma_data
        sigma_data_end = self._effective_sigma_data_end
        cov_xy = self._effective_cov_xy

        # Total variance A (adapted from DDBM Eq. 81)
        # A = \alpha_t^2 \sigma_0^2 + \beta_t^2 \sigma_T^2
        #   + 2 \alpha_t \beta_t \sigma_{0T} + \gamma_t^2
        A = (
            alpha_t**2 * sigma_data**2
            + beta_t**2 * sigma_data_end**2
            + 2 * alpha_t * beta_t * cov_xy
            + gamma_t**2
        )

        # c_in: input normalization
        c_in = 1 / torch.sqrt(A + 1e-8)

        # c_skip: skip connection weight
        numerator_skip = alpha_t * sigma_data**2 + beta_t * cov_xy
        c_skip = numerator_skip / (A + 1e-8)

        # c_out: output scaling
        numerator_out_sq = (
            beta_t**2 * (sigma_data**2 * sigma_data_end**2 - cov_xy**2)
            + gamma_t**2 * sigma_data**2
        )
        c_out = torch.sqrt(torch.clamp(numerator_out_sq, min=1e-8)) * c_in

        return c_skip, c_out, c_in

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        r"""Skip connection coefficient for ECSI preconditioning.

        Note: In ECSI, 'sigma' parameter represents time t \in [0,1].
        """
        t = sigma
        c_skip, _, _ = self._get_bridge_scalings(t)
        return c_skip

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        r"""Output scaling coefficient for ECSI preconditioning.

        Note: In ECSI, 'sigma' parameter represents time t \in [0,1].
        """
        t = sigma
        _, c_out, _ = self._get_bridge_scalings(t)
        return c_out

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        r"""Input scaling coefficient for ECSI preconditioning.

        Note: In ECSI, 'sigma' parameter represents time t \in [0,1].
        """
        t = sigma
        _, _, c_in = self._get_bridge_scalings(t)
        return c_in

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        r"""Noise level conditioning coefficient.

        Maps t to a conditioning value for the network.
        Uses log-scaling similar to EDM.

        Note: In ECSI, 'sigma' parameter represents time t \in [0,1].
        """
        t = sigma
        return 0.25 * torch.log(t + 1e-8)

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        r"""Compute loss weights based on time t_hat.

        Uses Karras-style weighting: w(t) = 1 / c_{out}(t)^2

        Parameters
        ----------
        t_hat : torch.Tensor
            Time values. Shape (B, N).

        Returns
        -------
        weights : torch.Tensor
            Loss weights. Shape (B, N).
        """
        c_out = self.c_out(t_hat)
        weights = 1 / (c_out**2 + 1e-8)
        return weights

    def forward_model(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor | float,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache=None,
        prior_coords: torch.Tensor | None = None,
        use_prior_coords: bool | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model with ECSI preconditioning.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t_hat : torch.Tensor | float
            Time values. Shape (B, N) or scalar.
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, L, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, L, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, L, L, c_z).
        model_cache : optional
            Model cache for efficiency.
        prior_coords : torch.Tensor | None
            Source (apo) coordinates x_T. Shape (B, N, L, 3).

        Returns
        -------
        denoised_coords : torch.Tensor
            Denoised (target) atom coordinates \\hat{x}_0. Shape (B, N, L, 3).
        """
        if not isinstance(t_hat, torch.Tensor):
            t_hat = torch.full(
                x_noisy.shape[:2], t_hat, device=x_noisy.device, dtype=x_noisy.dtype
            )  # [B, N]
        t_hat_reshaped = t_hat[..., None, None]  # [B, N, 1, 1]

        # Input preconditioning
        r_noisy = self.c_in(t_hat_reshaped) * x_noisy

        # Noise level conditioning
        c_noise = self.c_noise(t_hat)  # [B, N]

        effective_use_prior_coords = (
            self.use_prior_coords if use_prior_coords is None else use_prior_coords
        )
        if effective_use_prior_coords:
            # Concatenate with prior (apo) coordinates
            assert prior_coords is not None and torch.is_tensor(prior_coords), (
                "In ECSI, prior_coords should be Tensor when use_prior_coords=True"
            )
            assert prior_coords.shape == r_noisy.shape, (
                "In ECSI, the shapes of prior_coords and r_noisy should be the same"
            )
            if self.normalize_data_end and not self.normalize_coordinate:
                prior_coords = prior_coords / self.sigma_data_end
            r_noisy = torch.cat([r_noisy, prior_coords], dim=-1)
            assert r_noisy.shape[-1] == 6, "In ECSI, r_noisy last dim should be 6"
        else:
            # Do not condition score_model on prior_coords; keep r_noisy as (.., 3).
            assert r_noisy.shape[-1] == 3, "In ECSI, r_noisy last dim should be 3"

        # Call score model
        r_update = self.score_model(
            r_noisy=r_noisy,  # [B, N, La, 3] or [B, N, La, 6]
            c_noise=c_noise,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            model_cache=model_cache,
        )

        # Output preconditioning: \hat{x}_0 = c_{skip} * x_t + c_{out} * F_\theta
        x_out = (
            self.c_skip(t_hat_reshaped) * x_noisy + self.c_out(t_hat_reshaped) * r_update
        )
        return x_out

    def sample_noise_level(
        self,
        batch_size: int,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        r"""Sample time values for training.

        Returns samples in [sigma_min, sigma_max] which represents
        the time interval [t_{min}, t_{max}] \subset [0, 1].

        If logit_normal_sampling is True, samples from LogitNormal(0, 1).
        Else, samples from Beta(alpha, beta).
        If alpha=1, beta=1, this is equivalent to Uniform(0, 1).
        Finally scales to [sigma_min, sigma_max].

        Returns
        -------
        t : torch.Tensor
            Time values. Shape (B, N).
        """
        shape = (batch_size, num_diffusion_samples)

        if self.logit_normal_sampling:
            # LogitNormal(0, 1) sampling
            y = torch.randn(shape, device=device)
            t = torch.sigmoid(y)
        else:
            # Beta sampling (default to Uniform if alpha=1, beta=1)
            if self.sampling_alpha == 1.0 and self.sampling_beta == 1.0:
                t = torch.rand(shape, device=device)
            else:
                m = torch.distributions.Beta(
                    torch.tensor(self.sampling_alpha, device=device),
                    torch.tensor(self.sampling_beta, device=device),
                )
                t = m.sample(shape)

        # Scale to [sigma_min, sigma_max]
        t = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        return t

    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        r"""Get the time schedule for diffusion sampling.

        Uses Karras schedule adapted for t \in [0, 1].

        Parameters
        ----------
        num_steps : int, optional
            Number of sampling steps. If None, uses self.num_steps.
        device : torch.device, optional
            Device for tensor allocation.

        Returns
        -------
        times : torch.Tensor
            Time schedule. Shape (num_steps + 1,), from t_{max} to 0.
        """
        if num_steps is None:
            num_steps = self.num_steps

        schedule_type = self.sampling_schedule_type.lower()
        if schedule_type == "karras":
            times = self._get_karras_schedule(num_steps=num_steps, device=device)
        elif schedule_type == "piecewise_power":
            times = self._get_piecewise_power_schedule(num_steps=num_steps, device=device)
        elif schedule_type == "phase_power":
            times = self._get_phase_power_schedule(num_steps=num_steps, device=device)
        else:
            raise ValueError(
                "Unsupported sampling_schedule_type; expected 'karras', "
                "'piecewise_power', or 'phase_power', got "
                f"{self.sampling_schedule_type!r}"
            )

        # Last step is t=0 (exactly at target)
        times = F.pad(times, (0, 1), value=0.0)
        return times

    def _get_karras_schedule(
        self, num_steps: int, device: torch.device | None = None
    ) -> torch.Tensor:
        inv_rho = 1 / self.rho
        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        return (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

    def _get_piecewise_power_schedule(
        self, num_steps: int, device: torch.device | None = None
    ) -> torch.Tensor:
        power = float(self.sampling_schedule_piecewise_power)
        if power <= 0.0:
            raise ValueError("sampling_schedule_piecewise_power must be > 0")

        start_power = (
            power
            if self.sampling_schedule_start_power is None
            else float(self.sampling_schedule_start_power)
        )
        end_power = (
            power
            if self.sampling_schedule_end_power is None
            else float(self.sampling_schedule_end_power)
        )
        if start_power <= 0.0 or end_power <= 0.0:
            raise ValueError("piecewise start/end powers must be > 0")

        midpoint = float(self.sampling_schedule_midpoint)
        if not 0.0 < midpoint < 1.0:
            raise ValueError("sampling_schedule_midpoint must lie in (0, 1)")

        endpoint_trim = float(self.sampling_schedule_endpoint_trim)
        if not 0.0 <= endpoint_trim < 0.5:
            raise ValueError("sampling_schedule_endpoint_trim must lie in [0, 0.5)")

        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        u = steps / (num_steps - 1)

        if endpoint_trim > 0.0:
            u = endpoint_trim + (1.0 - 2.0 * endpoint_trim) * u

        t_unit = self._piecewise_power_curve(
            u=u,
            start_power=start_power,
            end_power=end_power,
            midpoint=midpoint,
        )

        if endpoint_trim > 0.0:
            start_u = torch.tensor([endpoint_trim], dtype=u.dtype, device=device)
            end_u = torch.tensor([1.0 - endpoint_trim], dtype=u.dtype, device=device)
            start_value = self._piecewise_power_curve(
                u=start_u,
                start_power=start_power,
                end_power=end_power,
                midpoint=midpoint,
            )
            end_value = self._piecewise_power_curve(
                u=end_u,
                start_power=start_power,
                end_power=end_power,
                midpoint=midpoint,
            )
            denom = start_value - end_value
            if torch.any(torch.abs(denom) < 1e-8):
                raise ValueError(
                    "sampling_schedule_endpoint_trim produced a degenerate schedule"
                )
            t_unit = (t_unit - end_value) / denom

        return self.sigma_min + (self.sigma_max - self.sigma_min) * t_unit

    @staticmethod
    def _piecewise_power_curve(
        u: torch.Tensor,
        start_power: float,
        end_power: float,
        midpoint: float,
    ) -> torch.Tensor:
        t_unit = torch.empty_like(u)
        left = u <= midpoint
        t_unit[left] = 1.0 - 0.5 * torch.pow(u[left] / midpoint, start_power)
        t_unit[~left] = 0.5 * torch.pow(
            (1.0 - u[~left]) / (1.0 - midpoint),
            end_power,
        )
        return t_unit.clamp(min=0.0, max=1.0)

    def _get_phase_power_schedule(
        self, num_steps: int, device: torch.device | None = None
    ) -> torch.Tensor:
        global_u_power = float(self.sampling_schedule_global_u_power)
        if global_u_power <= 0.0:
            raise ValueError("sampling_schedule_global_u_power must be > 0")

        churn_fraction = float(self.sampling_schedule_churn_fraction)
        ode_fraction = float(self.sampling_schedule_ode_fraction)
        if churn_fraction <= 0.0 or ode_fraction <= 0.0:
            raise ValueError(
                "sampling_schedule_churn_fraction and sampling_schedule_ode_fraction "
                "must both be > 0"
            )
        if churn_fraction + ode_fraction >= 1.0:
            raise ValueError(
                "sampling_schedule_churn_fraction + "
                "sampling_schedule_ode_fraction must be < 1"
            )

        churn_power = float(self.sampling_schedule_churn_power)
        middle_power = float(self.sampling_schedule_middle_power)
        ode_power = float(self.sampling_schedule_ode_power)
        if churn_power <= 1.0:
            raise ValueError("sampling_schedule_churn_power must be > 1")
        if middle_power <= 0.0:
            raise ValueError("sampling_schedule_middle_power must be > 0")
        if ode_power <= global_u_power:
            raise ValueError(
                "sampling_schedule_ode_power must be > "
                "sampling_schedule_global_u_power so the late ODE tail closes with "
                "zero slope in t-space"
            )

        total_scale = self.sigma_max - self.sigma_min
        if total_scale <= 0.0:
            raise ValueError("sigma_max must be > sigma_min")

        churn_time = self.churn_until_time
        if churn_time is None:
            churn_time = 0.7
        churn_time = float(churn_time)
        ode_time = float(self.ode_time_duration)
        if not self.sigma_min < ode_time < churn_time < self.sigma_max:
            raise ValueError(
                "phase_power schedule requires sigma_min < ode_time_duration < "
                "churn_until_time < sigma_max"
            )

        churn_unit = (churn_time - self.sigma_min) / total_scale
        ode_unit = (ode_time - self.sigma_min) / total_scale

        churn_u = churn_unit**global_u_power
        ode_u = ode_unit**global_u_power
        if not 0.0 < ode_u < churn_u < 1.0:
            raise ValueError("phase_power schedule produced invalid u-space bounds")

        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        s = steps / (num_steps - 1)
        churn_boundary = churn_fraction
        ode_boundary = 1.0 - ode_fraction

        u_value = torch.empty_like(s)

        head_mask = s <= churn_boundary
        mid_mask = (s > churn_boundary) & (s <= ode_boundary)
        tail_mask = s > ode_boundary

        head_progress = s[head_mask] / churn_boundary
        mid_progress = (s[mid_mask] - churn_boundary) / (ode_boundary - churn_boundary)
        tail_progress = (s[tail_mask] - ode_boundary) / (1.0 - ode_boundary)

        u_value[head_mask] = 1.0 - (1.0 - churn_u) * torch.pow(
            head_progress,
            churn_power,
        )
        u_value[mid_mask] = churn_u - (churn_u - ode_u) * torch.pow(
            mid_progress,
            middle_power,
        )
        u_value[tail_mask] = ode_u * torch.pow(1.0 - tail_progress, ode_power)

        t_unit = torch.pow(u_value.clamp(min=0.0, max=1.0), 1.0 / global_u_power)
        return self.sigma_min + total_scale * t_unit

    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution (apo structures).

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples, by default 1.
        label_coords : torch.Tensor | None, optional
            Label coordinates for alignment.

        Returns
        -------
        prior_coords : torch.Tensor
            prior coordinates. Shape (B, N, La, 3).
        """
        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        all_prior_coords = f_input.atom.prior_coords  # [B, Latom, Nprior, 3]
        num_prior = all_prior_coords.shape[-2]
        prior_index = [i % num_prior for i in range(num_diffusion_samples)]
        prior_coords = all_prior_coords[:, :, prior_index, :]  # [B, Latom, N, 3]
        prior_coords = prior_coords.permute(0, 2, 1, 3)  # [B, N, Latom, 3]

        if label_coords is None:
            # No label provided; apply random augmentation
            prior_mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Latom]
            prior_coords = self.apply_random_augmentation(prior_coords, mask=prior_mask)
        else:
            # Skip random augmentation since we will align to label
            prior_coords = self.align_apo_to_label(prior_coords, label_coords, f_input)

        return prior_coords

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
        f_input: FoldingInput | None = None,
    ) -> torch.Tensor:
        r"""Interpolate between apo and holo using ECSI bridge.

        Samples from the bridge distribution:
        x_t = \alpha_t x_0 + \beta_t x_T + \gamma_t z, where z ~ N(0, I)

        Parameters
        ----------
        noise_coords : torch.Tensor
            The apo (source) coordinates x_T. Shape (B, N, La, 3).
        label_coords : torch.Tensor
            The holo (target) coordinates x_0. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            Time values. Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).
        f_input : FoldingInput | None
            Unused. Kept for backward compatibility.

        Returns
        -------
        noised_coords : torch.Tensor
            Bridge-sampled coordinates x_t. Shape (B, N, La, 3).
        """
        del f_input

        x_apo = noise_coords  # x_T (source)
        x_holo = label_coords  # x_0 (target)

        # Expand t_hat to match coordinate dimensions
        t_expanded = t_hat[:, :, None, None]  # (B, N, 1, 1)

        # Compute interpolation coefficients
        alpha_t = self.alpha(t_expanded)  # weight for x_0 (holo)
        beta_t = self.beta(t_expanded)  # weight for x_T (apo)
        gamma_t = self.gamma(t_expanded)  # noise scale

        # Mean of bridge distribution: \mu_t = \alpha_t x_0 + \beta_t x_T
        mu_t = alpha_t * x_holo + beta_t * x_apo

        # Sample from bridge distribution
        noise = torch.randn_like(x_apo)
        noised_coords = mu_t + gamma_t * noise

        # Mask out padding atoms
        noised_coords = noised_coords * mask[:, None, :, None]

        return noised_coords

    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        diffusion_batch_size: int = 1,
        model_cache: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        batch_size = f_input.batch_size  # =B
        num_diffusion_samples = diffusion_batch_size  # =N
        mask = f_input.atom.pad_mask  # [B, La]

        with torch.no_grad():
            t_hat = self.sample_noise_level(
                batch_size, num_diffusion_samples, device=f_input.device
            )  # [B, N]

            # sample xT from label
            label_coords = self.sample_label(f_input, num_diffusion_samples)
            label_coords = label_coords * mask[..., None, :, None]

            # sample x0 from prior
            prior_coords = self.sample_prior(f_input, num_diffusion_samples, label_coords)
            prior_coords = prior_coords * mask[..., None, :, None]

            if self.normalize_coordinate:
                label_coords_norm = label_coords / self.sigma_data
                prior_coords_norm = prior_coords / self.sigma_data_end
            else:
                label_coords_norm = label_coords
                prior_coords_norm = prior_coords

            # sample xt via interpolation
            noised_atom_coords = self.interpolate(
                prior_coords_norm, label_coords_norm, t_hat, mask, f_input=f_input
            )
            noised_atom_coords = noised_atom_coords * mask[..., None, :, None]

        denoised_atom_coords = self.forward_model(
            x_noisy=noised_atom_coords,  # [B, N, La, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            model_cache=model_cache,
            prior_coords=prior_coords_norm,  # [B, N, La, 3]
        )  # [B, N, La, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "loss_weights": loss_weights,
            "prior_atom_coords": prior_coords_norm,
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "true_atom_coords": label_coords_norm,
        }

    @staticmethod
    def _apply_endpoint_perturbation(
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        perturb_scale: float,
    ) -> torch.Tensor:
        if perturb_scale <= 0.0:
            return coords
        noise = torch.randn_like(coords) * perturb_scale
        return (coords + noise) * atom_mask[..., None]

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_churn_target: torch.Tensor,
        atom_mask: torch.Tensor,
        t_curr: float,
        t_next: float,
        t_exp: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
        dt = t_next - t_curr
        delta_churn = float(self.churn_factor) * abs(dt)
        apply_churn = t_curr + delta_churn <= self.sigma_max
        if self.churn_until_time is not None:
            apply_churn = apply_churn and (t_curr > self.churn_until_time)

        if not apply_churn:
            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]),
                t_curr,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            return x_t, t_curr, t_curr_tensor, t_exp, dt

        alpha_t = self.alpha(t_exp)
        beta_t = self.beta(t_exp)
        gamma_t = self.gamma(t_exp)
        alpha_dot = self.alpha_deriv(t_exp)
        beta_dot = self.beta_deriv(t_exp)
        gamma_dot = self.gamma_deriv(t_exp)

        f_t = alpha_dot / (alpha_t + 1e-8)
        s_t = beta_dot - f_t * beta_t
        base_eps = gamma_t * gamma_dot - f_t * gamma_t**2
        g_t = torch.sqrt(torch.clamp(2.0 * base_eps, min=0.0) + 1e-8)

        churn_noise = torch.randn_like(x_t)
        x_t = (
            x_t
            + (f_t * x_t + s_t * x_churn_target) * delta_churn
            + g_t * (delta_churn**0.5) * churn_noise
        )
        x_t = x_t * atom_mask[..., None]

        t_curr = t_curr + delta_churn
        t_curr_tensor = torch.full(
            (x_t.shape[0], x_t.shape[1]),
            t_curr,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        t_exp = t_curr_tensor[:, :, None, None]
        dt = t_next - t_curr
        return x_t, t_curr, t_curr_tensor, t_exp, dt

    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_steps: int | None = None,
        num_diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        r"""Sample structures via ECSI sampling.

        Implements Algorithm 1 from the paper with Euler discretization.
        Uses stochasticity control via \eta parameter.

        The sampling SDE is:
        dX_t = b(t, X_t, x_T) dt + \sqrt{2\epsilon_t} dW_t

        where:
        b(t, x_t, x_T) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
                       + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
        \epsilon_t = \eta (\gamma_t \dot{\gamma}_t - \dot{\alpha}_t/\alpha_t \gamma_t^2)
        """
        sample_out: dict[str, torch.Tensor] = {}
        traj: list[torch.Tensor] = []

        if num_steps is None:
            num_steps = self.num_steps

        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples

        if self.perturb_xt and self.endpoint_perturb_scale is None:
            raise ValueError(
                "endpoint_perturb_scale must be provided when perturb_xt is enabled."
            )

        model_cache = {}

        # Get time schedule (from t_max toward 0)
        times = self.get_sampling_schedule(
            num_steps=num_steps, device=s_inputs.device
        ).tolist()

        atom_mask = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_diffusion_samples)  # (B, N, Latom, 3)
        x_T = x_T * atom_mask[..., None]

        sample_out["init_coordinates"] = x_T
        if self.normalize_coordinate:
            x_T = x_T / self.sigma_data_end

        x_t = x_T.clone()

        if self.perturb_xt:
            perturb_scale = float(self.endpoint_perturb_scale or 0.0)
            perturb_scale_xt = (
                perturb_scale / self.sigma_data_end
                if self.normalize_coordinate
                else perturb_scale
            )
            x_t = self._apply_endpoint_perturbation(x_t, atom_mask, perturb_scale_xt)
            x_churn_target = x_t.clone()
        else:
            x_churn_target = x_T

        if return_traj:
            traj.append(x_t.cpu())

        # Reverse time sampling from t=T toward t=0
        for step_idx in range(num_steps):
            # Apply random augmentation
            x_t, x_T, x_churn_target = self.random_augmentation(
                x_t, x_T, x_churn_target, mask=atom_mask
            )

            t_curr = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t_curr  # negative since t decreases

            # Convert to tensor
            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]), t_curr, device=x_t.device, dtype=x_t.dtype
            )
            t_exp = t_curr_tensor[:, :, None, None]  # (B, N, 1, 1)

            if self.use_forward_pinned_churn and self.churn_factor > 0.0:
                x_t, t_curr, t_curr_tensor, t_exp, dt = self._apply_forward_pinned_churn(
                    x_t=x_t,
                    x_churn_target=x_churn_target,
                    atom_mask=atom_mask,
                    t_curr=t_curr,
                    t_next=t_next,
                    t_exp=t_exp,
                )

            # Get denoised prediction \hat{x}_0
            x0_hat = torch.zeros_like(x_t)
            for st in range(0, num_diffusion_samples, max_parallel_samples):
                end = min(st + max_parallel_samples, num_diffusion_samples)
                x0_hat[:, st:end] = self.forward_model(
                    x_noisy=x_t[:, st:end],
                    t_hat=t_curr,
                    f_input=f_input,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    model_cache=model_cache,
                    prior_coords=x_T[:, st:end],
                )

                # Align x0_hat to the current state after churn.
                if self.inference_align_x0_hat_to_x_t:
                    x0_hat[:, st:end] = self.align_apo_to_label(
                        apo_coords=x0_hat[:, st:end],
                        label_coords=x_t[:, st:end],
                        f_input=f_input,
                    )

            # Compute route coefficients
            alpha_t = self.alpha(t_exp)
            beta_t = self.beta(t_exp)
            gamma_t = self.gamma(t_exp)
            alpha_dot = self.alpha_deriv(t_exp)
            beta_dot = self.beta_deriv(t_exp)
            gamma_dot = self.gamma_deriv(t_exp)

            # Compute \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
            z_hat = (x_t - alpha_t * x0_hat - beta_t * x_T) / (gamma_t + 1e-8)

            ode_time_duration = float(self.ode_time_duration)
            if ode_time_duration > 0.0 and t_curr <= ode_time_duration:
                t_next_exp = torch.full_like(t_exp, t_next)
                alpha_next = self.alpha(t_next_exp)
                beta_next = self.beta(t_next_exp)
                # Late-stage deterministic SI ODE update.
                x_t = (
                    beta_next / (beta_t + 1e-8) * x_t
                    + (alpha_next - alpha_t * beta_next / (beta_t + 1e-8)) * x0_hat
                )
            else:
                # Compute \epsilon_t = \eta (\gamma_t \dot{\gamma}_t
                #                    - \dot{\alpha}_t/\alpha_t \gamma_t^2)
                eps_t = self.eta * (
                    gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2
                )

                # Compute drift: b(t) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
                #                     + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
                drift = (
                    alpha_dot * x0_hat
                    + beta_dot * x_T
                    + (gamma_dot + eps_t / (gamma_t + 1e-8)) * z_hat
                )

                # Euler step: x_{t+dt} = x_t + b_t * dt + \sqrt{2\epsilon_t |dt|} * noise
                noise = torch.randn_like(x_t)
                diffusion_scale = torch.sqrt(2 * torch.abs(eps_t) * abs(dt) + 1e-8)
                x_t = x_t + drift * dt + diffusion_scale * noise

            # Apply mask
            x_t = x_t * atom_mask[..., None]

            if return_traj:
                traj.append(x_t.cpu())

        if self.normalize_coordinate:
            x_t = x_t * self.sigma_data

        sample_out["sample_coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj)

        return sample_out
