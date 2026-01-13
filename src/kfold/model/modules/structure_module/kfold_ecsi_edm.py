# Implementation of EDM formulation using ECSI framework
# EDM: alpha_t = 1, beta_t = 0, gamma_t = t
# This reduces ECSI to standard EDM (Elucidating the Design Space of Diffusion Models)

import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .kfold_ecsi import KFoldECSI


@STRUCTURE_MODULE.register()
class KFoldECSI_EDM(KFoldECSI):
    r"""EDM formulation using ECSI framework.

    Implements EDM (Elucidating the Design Space of Diffusion Models) using
    the ECSI interpolant framework. This is achieved by setting:
    - \alpha_t = 1 (constant weight for target x_0)
    - \beta_t = 0 (no contribution from source x_T)
    - \gamma_t = t (linear noise scale)

    This effectively reduces the bridge interpolant to:
    x_t = x_0 + t * z, where z ~ N(0, I)

    which is the standard EDM/VP-SDE formulation.

    Key differences from ECSI:
    - Prior distribution is Gaussian noise (not apo structure)
    - No conditioning on prior coordinates (use_prior_coords=False)
    - beta_t = 0 means x_T has no contribution to the interpolation

    Reference:
    - EDM: Karras et al., "Elucidating the Design Space of Diffusion-Based
           Generative Models"
    """

    class Config(BaseConfig):
        r"""Configuration for EDM Structure module.

        Inherits from ECSI config but sets EDM-appropriate defaults.
        """

        num_steps: int = 200
        sigma_min: float = 0.004
        sigma_max: float = 160.0
        gamma_max: float = 0.25  # Not used in EDM (gamma_t = t)
        sigma_data: float = 16.0
        sigma_data_end: float = 16.0  # Used for noise scaling
        cov_xy: float = 0.0  # No covariance in EDM (independent prior)
        rho: int = 7
        P_mean: float = -1.2
        P_std: float = 1.5
        eta: float = 1.0
        coordinate_augmentation: bool = True
        synchronize_sigmas: bool = False
        normalize_data_end: bool = False
        normalize_coordinate: bool = False
        logit_normal_sampling: bool = False
        sampling_alpha: float = 1.0
        sampling_beta: float = 1.0
        use_prior_coords: bool = False  # EDM does not condition on prior
        alignment_entity_strategy: str = "largest"
        prior_spread_radius: float = 0.0
        s_trans: float = 0.0

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the EDM module."""
        super().__init__(cfg, score_model)
        # Override cov_xy to 0 for EDM (independent Gaussian prior)
        self.cov_xy = 0.0

    # === EDM Route Functions (Overrides ECSI) === #
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for target (x_0/holo): \alpha_t = 1"""
        return torch.ones_like(t)

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of alpha: \dot{\alpha}_t = 0"""
        return torch.zeros_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for source (x_T/apo): \beta_t = 0"""
        return torch.zeros_like(t)

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of beta: \dot{\beta}_t = 0"""
        return torch.zeros_like(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Noise scale: \gamma_t = t"""
        return t

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative: \dot{\gamma}_t = 1"""
        return torch.ones_like(t)

    def sample_noise_level(
        self,
        batch_size: int,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.
        """

        # See Section 3.7 of AlphaFold3 paper.
        # t_hat = sigma_data * exp(-1.2 + 1.5 * N(0, 1)),
        # where -1.2 is P_mean and 1.5 is P_std.
        def _sample(*shape: int) -> torch.Tensor:
            return self.sigma_data * torch.exp(
                self.P_mean + self.P_std * torch.randn(shape, device=device)
            )

        if self.synchronize_sigmas:
            # synchronize sigmas across diffusion samples
            return _sample(batch_size, 1).repeat(1, num_diffusion_samples)
        else:
            # use different sigmas for each diffusion sample
            return _sample(batch_size, num_diffusion_samples)

    # === Override Prior Sampling for Pure Gaussian === #
    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution (Gaussian noise for EDM).

        Unlike ECSI which uses apo structures as prior, EDM uses
        Gaussian noise scaled by sigma_max (representing the state at t=T).

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples, by default 1.
        label_coords : torch.Tensor | None, optional
            Label coordinates (unused in EDM, kept for API compatibility).

        Returns
        -------
        noise : torch.Tensor
            Gaussian noise samples scaled by sigma_max. Shape (B, N, La, 3).
        """
        B = f_input.batch_size
        N = num_diffusion_samples
        La = f_input.num_atoms
        # Initial sample must be scaled by sigma_max to match the noise level at t=T
        return torch.randn((B, N, La, 3), device=f_input.device)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_data).clamp(1e-20).log() * 0.25

    # === For sampling === #
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
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.
        """

        sample_out: dict[str, torch.Tensor] = {}
        traj: list[torch.Tensor] = []

        if num_steps is None:
            num_steps = self.num_steps

        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples

        model_cache = {}

        # Get noise schedule
        sigmas = self.get_sampling_schedule(num_steps=num_steps, device=s_inputs.device)
        gammas = torch.where(sigmas > 1.0, 0.8, 0.0)
        sigmas, gammas = sigmas.tolist(), gammas.tolist()

        # NOTE: for sampling, there is no unresolved atoms.
        # Therefore, we can use pad_mask here.
        atom_mask = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)

        # Line 1
        init_sigma = sigmas[0]
        prior_coords = self.sample_prior(
            f_input, num_diffusion_samples
        )  # (B, N, Latom, 3)
        atom_coords: torch.Tensor = init_sigma * prior_coords  # (B, N, Latom, 3)
        start_coords = atom_coords

        if return_traj:
            traj.append(atom_coords.cpu())  # Move to cpu to save memory

        # Line 2: gradually denoise
        for step_idx in range(1, num_steps):
            # Line 3
            atom_coords = self.random_augmentation(atom_coords, mask=atom_mask)

            # Line 4
            sigma_tm, sigma_t, gamma = (
                sigmas[step_idx - 1],
                sigmas[step_idx],
                gammas[step_idx],
            )

            # Line 5
            t_hat: float = sigma_tm * (1 + gamma)

            # Line 6
            noise_var: float = 1.003**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * torch.randn_like(atom_coords)

            # Line 7
            atom_coords_noisy = atom_coords + eps

            # Line 8
            # Process in chunks for memory efficiency
            atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
            for st in range(0, num_diffusion_samples, max_parallel_samples):
                end = min(st + max_parallel_samples, num_diffusion_samples)
                atom_coords_denoised[:, st:end] = self.forward_model(
                    x_noisy=atom_coords_noisy[:, st:end],
                    t_hat=t_hat,
                    f_input=f_input,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    model_cache=model_cache,
                )

            # Line 9
            delta_coords = (atom_coords_noisy - atom_coords_denoised) / t_hat

            # line 10
            dt = sigma_t - t_hat

            # Line 11
            atom_coords = atom_coords_noisy + 1.5 * dt * delta_coords

            if return_traj:
                traj.append(atom_coords.cpu())  # Move to cpu to save memory

        sample_out["init_coordinates"] = start_coords
        sample_out["sample_coordinates"] = atom_coords
        if return_traj:
            sample_out["traj"] = torch.stack(traj)

        return sample_out

    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the noise schedule for diffusion sampling."""

        if num_steps is None:
            num_steps = self.num_steps

        inv_rho = 1 / self.rho

        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas
