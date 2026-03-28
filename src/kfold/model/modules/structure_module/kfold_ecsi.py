# Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI)
# Based on "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553)
# Adapted from ECSI training code and kfold_ddbm.py

import dataclasses

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import CenterRandomAugmentation, get_center
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseECSI


class SICoeffs:
    """Stochastic interpolant coefficient helper for ECSI."""

    def __init__(
        self,
        *,
        power: float,
        gamma_max: float,
        gamma_scale_com: float,
        gamma_scale_internal: float,
    ) -> None:
        self.power = power
        self.gamma_max = gamma_max
        self.gamma_scale_com = gamma_scale_com
        self.gamma_scale_internal = gamma_scale_internal

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
        return self._gamma_with_max(t, self.gamma_max)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        return self._gamma_deriv_with_max(t, self.gamma_max)

    def gamma_com(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma_scale_com * self.gamma(t)

    def gamma_com_deriv(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma_scale_com * self.gamma_deriv(t)

    def gamma_internal(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma_scale_internal * self.gamma(t)

    def gamma_internal_deriv(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma_scale_internal * self.gamma_deriv(t)

    def _gamma_with_max(self, t: torch.Tensor, gamma_max: float) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        t_pow = torch.pow(t_clamped, self.power)
        return 0.5 * gamma_max * torch.sqrt(t_pow * (1 - t_pow) + 1e-8)

    def _gamma_deriv_with_max(self, t: torch.Tensor, gamma_max: float) -> torch.Tensor:
        t_clamped = self._clamp_t(t)
        t_pow = torch.pow(t_clamped, self.power)
        denom = torch.sqrt(t_pow * (1 - t_pow) + 1e-8)
        coeff = self.power * torch.pow(t_clamped, self.power - 1)
        return (gamma_max / 4) * coeff * (1 - 2 * t_pow) / (denom + 1e-8)


@dataclasses.dataclass(kw_only=True)
class SamplingScheduleConfig:
    """Maps normalized solver progress to reverse-time sampling time.

    Reverse-time sampling itself runs in time-space from `t = time_max` down to `0`.
    Internally, the scheduler first parameterizes solver progress with
    `s in [0, 1]`, then maps that progress to actual sampling time `t`.

    In solver-progress space, phase-power allocates steps across three regions:

      head   : solver progress in [0, churn_fraction]
      middle : solver progress in (churn_fraction, 1 - ode_fraction]
      tail   : solver progress in (1 - ode_fraction, 1]

    Higher `churn_power` concentrates more steps near `time_max`. Higher
    `ode_power` makes the late tail flatter near `t = 0`. These fields decide
    where the solver spends steps, not which dynamics branch is used.
    """

    global_u_power: float = 1.0
    churn_fraction: float = 0.3
    ode_fraction: float = 0.45
    middle_power: float = 1.0
    churn_power: float = 1.75
    ode_power: float = 2.6


@dataclasses.dataclass(kw_only=True)
class SamplingConfig:
    """Controls how reverse-time ECSI sampling proceeds.

    Timeline in time-space (`t: time_max -> 0`):

      1. Prior initialization
         - start from sampled prior `x_T`
         - if `perturb_xt`, add endpoint noise with `endpoint_perturb_scale`

      2. Early high-time region (`t > churn_end_time`)
         - if `use_pinned_churn`, apply the forward-pinned churn substep

      3. Middle stochastic region (`ode_start_time < t <= churn_end_time`)
         - use the expanded ECSI SDE update
         - stochasticity is controlled by `eta`, `eta_com`, `eta_internal`

      4. Late deterministic region (`t <= ode_start_time`)
         - switch to the SI ODE update
         - no diffusion noise is added in this branch

    Parameter groups:
      - horizon: `steps`, `time_min`, `time_max`
      - stochasticity: `eta`, `eta_com`, `eta_internal`
      - endpoint handling: `perturb_xt`, `endpoint_perturb_scale`
      - early churn: `use_pinned_churn`, `churn_factor`, `churn_end_time`
      - late ODE switch: `ode_start_time`
      - time allocation across steps: `schedule`
    """

    steps: int = 200
    time_min: float = 0.001
    time_max: float = 0.999
    eta: float = 1.0
    eta_com: float | None = None
    eta_internal: float | None = None
    align_x0_hat_to_xt: bool = True
    perturb_xt: bool = True
    endpoint_perturb_scale: float | None = 0.1
    ode_start_time: float = 0.6
    use_pinned_churn: bool = True
    churn_factor: float = 3.0
    churn_end_time: float = 0.7
    schedule: SamplingScheduleConfig = dataclasses.field(
        default_factory=SamplingScheduleConfig
    )


@dataclasses.dataclass(kw_only=True)
class TrainTimeSamplingConfig:
    """Configuration for train-time sampling of `t_hat`.

    Parameters
    ----------
    logit_normal : bool, optional
        If `True`, ignore the Beta/Uniform branch and sample from a
        LogitNormal-like `sigmoid(N(0, 1))` distribution.
    alpha : float, optional
        Alpha parameter of the Beta branch.
    beta : float, optional
        Beta parameter of the Beta branch.
    uniform_mix_prob : float, optional
        Per-sample probability of drawing from `Uniform(0, 1)` instead of the
        Beta branch when `logit_normal` is disabled.
    """

    logit_normal: bool = False
    alpha: float = 1.0
    beta: float = 1.0
    uniform_mix_prob: float = 0.0


@STRUCTURE_MODULE.register()
class KFoldECSI(BaseECSI):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Expanded bridge dynamics with separate COM and internal-coordinate noise paths
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
            Shared base bridge maximum used by `gamma(t)`. Expanded COM/internal
            branches are defined by scaling this base gamma with
            `gamma_scale_com` and `gamma_scale_internal`.
        gamma_scale_com : float, optional
            Multiplicative scale applied to the shared base gamma for the COM
            branch of the expanded bridge dynamics.
        gamma_scale_internal : float, optional
            Multiplicative scale applied to the shared base gamma for the
            internal-coordinate branch of the expanded bridge dynamics.
        time_power : float, optional
            Shared exponent `k` for the route coefficients
            `alpha_t = 1 - t^k`, `beta_t = t^k`, and the base gamma schedule.
        sigma_data : float, optional
            Effective target-coordinate scale used in ECSI preconditioning.
        sigma_data_end : float, optional
            Effective source-coordinate scale used in ECSI preconditioning.
        cov_xy : float, optional
            Cross-covariance term between source and target coordinates used by
            the bridge preconditioning formulas.
        sampling : SamplingConfig, optional
            Reverse-time rollout configuration including stochasticity, endpoint
            perturbation, late ODE switching, and the nested step-allocation
            schedule.
        train_time_sampling : TrainTimeSamplingConfig, optional
            Training-time sampling policy for `t_hat`, including the optional
            Uniform mixture applied to the Beta branch.
        coordinate_augmentation : bool, optional
            Whether to apply random rigid-body augmentation when centering
            coordinates.
        normalize_data_end : bool, optional
            Whether prior coordinates are normalized before being concatenated
            into the score-model input.
        normalize_coordinate : bool, optional
            Whether the module internally works on normalized coordinates for
            both source and target structures.
        use_prior_coords : bool, optional
            Whether the score model receives prior/source coordinates as an
            additional conditioning input.
        train_align_prior_to_label : bool, optional
            Whether sampled prior coordinates are rigidly aligned to labels
            during training before interpolation.
        s_trans : float, optional
            Translation scale used by `CenterRandomAugmentation`.
        """

        gamma_max: float = 12.0
        gamma_scale_com: float = 1.0
        gamma_scale_internal: float = 1.0
        time_power: float = 1.0
        sigma_data: float = 16.0
        sigma_data_end: float = 16.0
        cov_xy: float = 128.0
        sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
        train_time_sampling: TrainTimeSamplingConfig = dataclasses.field(
            default_factory=TrainTimeSamplingConfig
        )
        coordinate_augmentation: bool = True
        normalize_data_end: bool = False
        normalize_coordinate: bool = False
        use_prior_coords: bool = True
        train_align_prior_to_label: bool = True
        s_trans: float = 1.0

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the ECSI module.

        The constructor copies the high-level config fields onto runtime
        attributes, then immediately validates and normalizes the nested
        sampling config so downstream code can assume a runtime-ready ECSI
        configuration.
        """
        super().__init__(cfg, score_model)
        self.sampling = cfg.sampling
        self.train_time_sampling = cfg.train_time_sampling

        self.gamma_max: float = float(cfg.gamma_max)
        self.gamma_scale_com: float = float(cfg.gamma_scale_com)
        self.gamma_scale_internal: float = float(cfg.gamma_scale_internal)
        self.time_power: float = cfg.time_power
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation
        self.normalize_data_end: bool = cfg.normalize_data_end
        self.normalize_coordinate: bool = cfg.normalize_coordinate
        self.use_prior_coords: bool = cfg.use_prior_coords
        self.s_trans: float = cfg.s_trans
        self.train_align_prior_to_label: bool = cfg.train_align_prior_to_label

        self.__validate_config()

        self.si_coeffs = SICoeffs(
            power=self.time_power,
            gamma_max=self.gamma_max,
            gamma_scale_com=self.gamma_scale_com,
            gamma_scale_internal=self.gamma_scale_internal,
        )

        self.random_augmentation = CenterRandomAugmentation(
            centering=True,
            augmentation=self.coordinate_augmentation,
            s_trans=self.s_trans,
        )

    def __validate_config(self) -> None:
        """Validate and normalize runtime config used by ECSI.

        This method is the single runtime gate for ECSI-specific config
        correctness. It enforces schedule invariants, validates time/stochastic
        parameter ranges, and normalizes optional sampling fields such as
        `eta_com`, `eta_internal`, and `churn_end_time`.
        """
        schedule = self.sampling.schedule

        if schedule.global_u_power <= 0.0:
            raise ValueError("sampling.schedule.global_u_power must be > 0")
        if schedule.churn_fraction <= 0.0 or schedule.ode_fraction <= 0.0:
            raise ValueError(
                "sampling.schedule.churn_fraction and "
                "sampling.schedule.ode_fraction must both be > 0"
            )
        if schedule.churn_fraction + schedule.ode_fraction >= 1.0:
            raise ValueError(
                "sampling.schedule.churn_fraction + "
                "sampling.schedule.ode_fraction must be < 1"
            )
        if schedule.churn_power <= 1.0:
            raise ValueError("sampling.schedule.churn_power must be > 1")
        if schedule.middle_power <= 0.0:
            raise ValueError("sampling.schedule.middle_power must be > 0")
        if schedule.ode_power <= schedule.global_u_power:
            raise ValueError(
                "sampling.schedule.ode_power must be > "
                "sampling.schedule.global_u_power"
            )

        if self.sampling.steps <= 0:
            raise ValueError("sampling.steps must be > 0")
        if self.sampling.time_max <= self.sampling.time_min:
            raise ValueError("sampling.time_max must be > sampling.time_min")
        if self.sampling.eta < 0.0:
            raise ValueError("sampling.eta must be >= 0")
        if self.sampling.eta_com is None:
            self.sampling.eta_com = self.sampling.eta
        elif self.sampling.eta_com < 0.0:
            raise ValueError("sampling.eta_com must be >= 0")
        if self.sampling.eta_internal is None:
            self.sampling.eta_internal = self.sampling.eta
        elif self.sampling.eta_internal < 0.0:
            raise ValueError("sampling.eta_internal must be >= 0")
        if self.sampling.churn_end_time is None:
            self.sampling.churn_end_time = 0.7
        if self.sampling.perturb_xt and self.sampling.endpoint_perturb_scale is None:
            raise ValueError(
                "sampling.endpoint_perturb_scale must be provided when "
                "sampling.perturb_xt is enabled"
            )
        if not (
            self.sampling.time_min
            < self.sampling.ode_start_time
            < self.sampling.churn_end_time
            < self.sampling.time_max
        ):
            raise ValueError(
                "sampling requires time_min < ode_time < churn_until_time < time_max"
            )

        if self.train_time_sampling.alpha <= 0.0:
            raise ValueError("train_time_sampling.alpha must be > 0")
        if self.train_time_sampling.beta <= 0.0:
            raise ValueError("train_time_sampling.beta must be > 0")
        if not 0.0 <= self.train_time_sampling.uniform_mix_prob <= 1.0:
            raise ValueError(
                "train_time_sampling.uniform_mix_prob must lie in [0, 1]"
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
        alpha_t = self.si_coeffs.alpha(t)
        beta_t = self.si_coeffs.beta(t)
        gamma_t = self.si_coeffs.gamma(t)

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

        Returns samples in [sampling_time_min, sampling_time_max] which represents
        the time interval [t_{min}, t_{max}] \subset [0, 1].

        If `train_time_sampling.logit_normal` is True, sample from
        `sigmoid(N(0, 1))`.
        Else, sample from a Bernoulli mixture between:
        - `Uniform(0, 1)` with probability `uniform_mix_prob`
        - `Beta(alpha, beta)` with probability `1 - uniform_mix_prob`
        If `alpha=1` and `beta=1`, the Beta branch is itself Uniform.
        Finally scales to [sampling_time_min, sampling_time_max].

        Returns
        -------
        t : torch.Tensor
            Time values. Shape (B, N).
        """
        shape = (batch_size, num_diffusion_samples)

        if self.train_time_sampling.logit_normal:
            # LogitNormal(0, 1) sampling
            y = torch.randn(shape, device=device)
            t = torch.sigmoid(y)
        else:
            # Beta sampling branch (defaulting to Uniform if alpha=1, beta=1).
            if (
                self.train_time_sampling.alpha == 1.0
                and self.train_time_sampling.beta == 1.0
            ):
                t = torch.rand(shape, device=device)
            else:
                m = torch.distributions.Beta(
                    torch.tensor(self.train_time_sampling.alpha, device=device),
                    torch.tensor(self.train_time_sampling.beta, device=device),
                )
                t = m.sample(shape)

            mix_prob = float(self.train_time_sampling.uniform_mix_prob)
            if mix_prob > 0.0:
                uniform_sample = torch.rand(shape, device=device)
                uniform_mask = torch.rand(shape, device=device) < mix_prob
                t = torch.where(uniform_mask, uniform_sample, t)

        # Scale to [sampling_time_min, sampling_time_max]
        t = self.sampling.time_min + (self.sampling.time_max - self.sampling.time_min) * t
        return t

    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        r"""Get the time schedule for diffusion sampling.

        Uses the configured phase-power schedule adapted for t \in [0, 1].

        Parameters
        ----------
        num_steps : int, optional
            Number of sampling steps. If None, uses `self.sampling.steps`.
        device : torch.device, optional
            Device for tensor allocation.

        Returns
        -------
        times : torch.Tensor
            Time schedule. Shape (num_steps + 1,), from t_max to 0.
        """
        if num_steps is None:
            num_steps = self.sampling.steps

        times = self._get_phase_power_schedule(num_steps=num_steps, device=device)

        # Last step is t=0 (exactly at target)
        times = F.pad(times, (0, 1), value=0.0)
        return times

    def _get_phase_power_schedule(
        self, num_steps: int, device: torch.device | None = None
    ) -> torch.Tensor:
        sampling = self.sampling
        schedule = sampling.schedule
        global_u_power = float(schedule.global_u_power)
        churn_fraction = float(schedule.churn_fraction)
        ode_fraction = float(schedule.ode_fraction)
        churn_power = float(schedule.churn_power)
        middle_power = float(schedule.middle_power)
        ode_power = float(schedule.ode_power)
        total_scale = sampling.time_max - sampling.time_min
        churn_time = float(sampling.churn_end_time)
        ode_time = float(sampling.ode_start_time)

        churn_unit = (churn_time - sampling.time_min) / total_scale
        ode_unit = (ode_time - sampling.time_min) / total_scale

        churn_u = churn_unit**global_u_power
        ode_u = ode_unit**global_u_power

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
        return sampling.time_min + total_scale * t_unit

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
            if self.train_align_prior_to_label:
                prior_coords = self.align_apo_to_label(prior_coords, label_coords)

        return prior_coords

    @staticmethod
    def _broadcast_mask(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_out = mask.bool()
        while mask_out.ndim < coords.ndim - 1:
            mask_out = mask_out.unsqueeze(-2)
        return mask_out.expand(coords.shape[:-1])

    def compute_com(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_broadcast = self._broadcast_mask(coords, mask)
        return get_center(coords, mask_broadcast)

    def decompose_coords(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_broadcast = self._broadcast_mask(coords, mask)
        com = self.compute_com(coords, mask_broadcast)
        internal = (coords - com) * mask_broadcast.unsqueeze(-1)
        return com, internal

    def recompose_coords(
        self, com: torch.Tensor, internal: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_broadcast = self._broadcast_mask(internal, mask)
        return (com + internal) * mask_broadcast.unsqueeze(-1)

    def _component_scales(
        self, t_exp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma_com = self.si_coeffs.gamma_com(t_exp)
        gamma_internal = self.si_coeffs.gamma_internal(t_exp)
        gamma_dot_com = self.si_coeffs.gamma_com_deriv(t_exp)
        gamma_dot_internal = self.si_coeffs.gamma_internal_deriv(t_exp)
        return gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal

    @staticmethod
    def _compute_eps(
        gamma_t: torch.Tensor,
        gamma_dot: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_dot: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        return eta * (gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2)

    def _compute_drift_components(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        x_T: torch.Tensor,
        t_exp: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t = self.si_coeffs.alpha(t_exp)
        beta_t = self.si_coeffs.beta(t_exp)
        alpha_dot = self.si_coeffs.alpha_deriv(t_exp)
        beta_dot = self.si_coeffs.beta_deriv(t_exp)
        gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal = (
            self._component_scales(t_exp)
        )

        x_t_com, x_t_internal = self.decompose_coords(x_t, mask)
        x0_com, x0_internal = self.decompose_coords(x0_hat, mask)
        xT_com, xT_internal = self.decompose_coords(x_T, mask)

        z_hat_com = (x_t_com - alpha_t * x0_com - beta_t * xT_com) / (gamma_com + 1e-8)
        z_hat_internal = (x_t_internal - alpha_t * x0_internal - beta_t * xT_internal) / (
            gamma_internal + 1e-8
        )

        eps_com = self._compute_eps(
            gamma_t=gamma_com,
            gamma_dot=gamma_dot_com,
            alpha_t=alpha_t,
            alpha_dot=alpha_dot,
            eta=(
                self.sampling.eta
                if self.sampling.eta_com is None
                else self.sampling.eta_com
            ),
        )
        eps_internal = self._compute_eps(
            gamma_t=gamma_internal,
            gamma_dot=gamma_dot_internal,
            alpha_t=alpha_t,
            alpha_dot=alpha_dot,
            eta=(
                self.sampling.eta
                if self.sampling.eta_internal is None
                else self.sampling.eta_internal
            ),
        )

        drift_com = (
            alpha_dot * x0_com
            + beta_dot * xT_com
            + (gamma_dot_com + eps_com / (gamma_com + 1e-8)) * z_hat_com
        )
        drift_internal = (
            alpha_dot * x0_internal
            + beta_dot * xT_internal
            + (gamma_dot_internal + eps_internal / (gamma_internal + 1e-8))
            * z_hat_internal
        )
        return drift_com, drift_internal, eps_com, eps_internal

    def _compute_sampling_drift_and_diffusion(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        x_T: torch.Tensor,
        t_exp: torch.Tensor,
        atom_mask: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drift_com, drift_internal, eps_com, eps_internal = self._compute_drift_components(
            x_t=x_t,
            x0_hat=x0_hat,
            x_T=x_T,
            t_exp=t_exp,
            mask=atom_mask,
        )
        drift = drift_com + drift_internal

        diffusion_noise = torch.randn_like(x_t)
        noise_com, noise_internal = self.decompose_coords(diffusion_noise, atom_mask)
        diffusion = (
            torch.sqrt(2 * torch.abs(eps_com) * abs(dt) + 1e-8) * noise_com
            + torch.sqrt(2 * torch.abs(eps_internal) * abs(dt) + 1e-8) * noise_internal
        )
        return drift, diffusion

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
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

        Returns
        -------
        noised_coords : torch.Tensor
            Bridge-sampled coordinates x_t. Shape (B, N, La, 3).
        """
        t_expanded = t_hat[:, :, None, None]
        alpha_t = self.si_coeffs.alpha(t_expanded)
        beta_t = self.si_coeffs.beta(t_expanded)
        gamma_com, gamma_internal, _, _ = self._component_scales(t_expanded)

        apo_com, apo_internal = self.decompose_coords(noise_coords, mask)
        holo_com, holo_internal = self.decompose_coords(label_coords, mask)

        mean_com = alpha_t * holo_com + beta_t * apo_com
        mean_internal = alpha_t * holo_internal + beta_t * apo_internal

        full_noise = torch.randn_like(noise_coords)
        noise_com, noise_internal = self.decompose_coords(full_noise, mask)

        x_t_com = mean_com + gamma_com * noise_com
        x_t_internal = mean_internal + gamma_internal * noise_internal
        return self.recompose_coords(x_t_com, x_t_internal, mask)

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
                prior_coords_norm, label_coords_norm, t_hat, mask
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
    ) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        dt = t_next - t_curr
        delta_churn = float(self.sampling.churn_factor) * abs(dt)
        apply_churn = t_curr + delta_churn <= self.sampling.time_max
        if self.sampling.churn_end_time is not None:
            apply_churn = apply_churn and (t_curr > self.sampling.churn_end_time)

        if not apply_churn:
            return x_t, t_curr, t_exp, dt

        alpha_t = self.si_coeffs.alpha(t_exp)
        beta_t = self.si_coeffs.beta(t_exp)
        alpha_dot = self.si_coeffs.alpha_deriv(t_exp)
        beta_dot = self.si_coeffs.beta_deriv(t_exp)
        gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal = (
            self._component_scales(t_exp)
        )

        f_t = alpha_dot / (alpha_t + 1e-8)
        s_t = beta_dot - f_t * beta_t
        base_eps_com = gamma_com * gamma_dot_com - f_t * gamma_com**2
        base_eps_internal = gamma_internal * gamma_dot_internal - f_t * gamma_internal**2
        g_com = torch.sqrt(torch.clamp(2.0 * base_eps_com, min=0.0) + 1e-8)
        g_internal = torch.sqrt(torch.clamp(2.0 * base_eps_internal, min=0.0) + 1e-8)

        x_t_com, x_t_internal = self.decompose_coords(x_t, atom_mask)
        x_target_com, x_target_internal = self.decompose_coords(x_churn_target, atom_mask)
        churn_noise = torch.randn_like(x_t)
        noise_com, noise_internal = self.decompose_coords(churn_noise, atom_mask)

        x_t_com = (
            x_t_com
            + (f_t * x_t_com + s_t * x_target_com) * delta_churn
            + g_com * (delta_churn**0.5) * noise_com
        )
        x_t_internal = (
            x_t_internal
            + (f_t * x_t_internal + s_t * x_target_internal) * delta_churn
            + g_internal * (delta_churn**0.5) * noise_internal
        )
        x_t = self.recompose_coords(x_t_com, x_t_internal, atom_mask)

        t_curr = t_curr + delta_churn
        t_curr_tensor = torch.full(
            (x_t.shape[0], x_t.shape[1]),
            t_curr,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        t_exp = t_curr_tensor[:, :, None, None]
        dt = t_next - t_curr
        return x_t, t_curr, t_exp, dt

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
        Uses stochasticity control via `sampling_eta`-style parameters.

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
            num_steps = self.sampling.steps

        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples

        if self.sampling.perturb_xt and self.sampling.endpoint_perturb_scale is None:
            raise ValueError(
                "sampling_endpoint_perturb_scale must be provided when "
                "sampling_perturb_xt is enabled."
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

        if self.sampling.perturb_xt:
            perturb_scale = float(self.sampling.endpoint_perturb_scale or 0.0)
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

            if self.sampling.use_pinned_churn and self.sampling.churn_factor > 0.0:
                x_t, t_curr, t_exp, dt = self._apply_forward_pinned_churn(
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
                if self.sampling.align_x0_hat_to_xt:
                    x0_hat[:, st:end] = self.align_apo_to_label(
                        apo_coords=x0_hat[:, st:end],
                        label_coords=x_t[:, st:end],
                    )

            alpha_t = self.si_coeffs.alpha(t_exp)
            beta_t = self.si_coeffs.beta(t_exp)

            sampling_ode_time = float(self.sampling.ode_start_time)
            if sampling_ode_time > 0.0 and t_curr <= sampling_ode_time:
                t_next_exp = torch.full_like(t_exp, t_next)
                alpha_next = self.si_coeffs.alpha(t_next_exp)
                beta_next = self.si_coeffs.beta(t_next_exp)
                # Late-stage deterministic SI ODE update.
                x_t = (
                    beta_next / (beta_t + 1e-8) * x_t
                    + (alpha_next - alpha_t * beta_next / (beta_t + 1e-8)) * x0_hat
                )
            else:
                drift, diffusion = self._compute_sampling_drift_and_diffusion(
                    x_t=x_t,
                    x0_hat=x0_hat,
                    x_T=x_T,
                    t_exp=t_exp,
                    atom_mask=atom_mask,
                    dt=dt,
                )
                x_t = x_t + drift * dt + diffusion

            # Apply mask
            x_t = x_t * atom_mask[..., None]

            if return_traj:
                traj.append(x_t.cpu())

        if self.normalize_coordinate:
            x_t = x_t * self.sigma_data

        sample_out["sample_coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Latom, 3)

        return sample_out
