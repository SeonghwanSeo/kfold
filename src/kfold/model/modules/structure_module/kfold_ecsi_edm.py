# Implementation of EDM formulation using ECSI framework
# EDM: alpha_t = 1, beta_t = 0, gamma_t = t
# This reduces ECSI to standard EDM (Elucidating the Design Space of Diffusion Models)

import torch

from kfold.data.model_input import FoldingInput
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
        sigma_min: float = 0.001
        sigma_max: float = 0.999
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

    # === Override Prior Sampling for Pure Gaussian === #
    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution (unit Gaussian noise for EDM).

        Unlike ECSI which uses apo structures as prior, EDM uses
        isotropic unit Gaussian noise. No alignment needed since
        Gaussian is rotation-invariant.

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
            Unit Gaussian noise samples. Shape (B, N, La, 3).
        """
        B = f_input.batch_size
        N = num_diffusion_samples
        La = f_input.num_atoms
        return torch.randn((B, N, La, 3), device=f_input.device)
