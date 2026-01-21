# Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI)
# Based on "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553)
# Adapted from ECSI training code and kfold_ddbm.py

import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import CenterRandomAugmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseECSI


@STRUCTURE_MODULE.register()
class KFoldECSI(BaseECSI):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Decoupled kernel parameters (\alpha_t, \beta_t, \gamma_t) for flexible bridge paths
    - Linear route: \alpha_t=1-t, \beta_t=t,
      \gamma_t^2=\gamma_{max}^2/4 \cdot t(1-t)
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
            Uses \gamma_t^2 = \gamma_{max}^2/4 * t(1-t).
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
        synchronize_sigmas : bool, optional
            Whether to synchronize sigmas across diffusion samples, by default False.
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
        route_type: str = "linear"
        ddbm_vp_beta_min: float = 0.1
        ddbm_vp_beta_d: float = 16.0
        sigma_data: float = 16.0
        sigma_data_end: float = 16.0
        cov_xy: float = 128.0
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
        use_prior_coords: bool = True
        alignment_entity_strategy: str | None = None
        alignment_level: str = "chain"
        s_trans: float = 1.0
        inference_align_x0_hat_to_x_apo: bool = True
        chain_wise_perturbation: bool = True
        inference_independent_diffusion_apo_sampling: bool = False
        inference_apo_translation_scale: float = 0.0
        inference_apo_chain_com_sampling_radius: float | None = None
        ode_time_duration: float = 0.5

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the ECSI module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.gamma_max: float = cfg.gamma_max
        self.route_type: str = cfg.route_type
        self.ddbm_vp_beta_min: float = cfg.ddbm_vp_beta_min
        self.ddbm_vp_beta_d: float = cfg.ddbm_vp_beta_d
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy
        self.rho: int = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.eta: float = cfg.eta
        self.num_steps: int = cfg.num_steps
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation
        self.synchronize_sigmas: bool = cfg.synchronize_sigmas
        self.normalize_data_end: bool = cfg.normalize_data_end
        self.normalize_coordinate: bool = cfg.normalize_coordinate
        self.logit_normal_sampling: bool = cfg.logit_normal_sampling
        self.sampling_alpha: float = cfg.sampling_alpha
        self.sampling_beta: float = cfg.sampling_beta
        self.use_prior_coords: bool = cfg.use_prior_coords
        self.s_trans: float = cfg.s_trans
        self.alignment_level: str = cfg.alignment_level
        self.inference_align_x0_hat_to_x_apo: bool = cfg.inference_align_x0_hat_to_x_apo
        self.chain_wise_perturbation: bool = cfg.chain_wise_perturbation
        self.inference_independent_diffusion_apo_sampling: bool = (
            cfg.inference_independent_diffusion_apo_sampling
        )
        self.inference_apo_translation_scale: float = cfg.inference_apo_translation_scale
        self.inference_apo_chain_com_sampling_radius: float | None = (
            cfg.inference_apo_chain_com_sampling_radius
        )
        self.ode_time_duration: float = cfg.ode_time_duration

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

    def apply_chain_random_augmentation(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Apply chain-wise random augmentation to coordinates."""
        if not self.coordinate_augmentation:
            return coords

        added_sample_dim = False
        if coords.dim() == 3:
            coords = coords.unsqueeze(1)
            added_sample_dim = True

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1 and coords.shape[1] > 1:
            mask = mask.expand(-1, coords.shape[1], -1)

        if coords.dim() != 4 or mask.dim() != 3:
            raise ValueError(
                "Expected coords shape (B, N, L, 3) and mask shape (B, N, L), "
                f"got coords {coords.shape} and mask {mask.shape}."
            )

        token_asym_id = f_input.token.asym_id
        atom_token_index = f_input.atom.token_index

        if token_asym_id.dim() == 1:
            token_asym_id = token_asym_id.unsqueeze(0)
        if atom_token_index.dim() == 1:
            atom_token_index = atom_token_index.unsqueeze(0)

        if token_asym_id.shape[0] == 1 and coords.shape[0] > 1:
            token_asym_id = token_asym_id.expand(coords.shape[0], -1)
        if atom_token_index.shape[0] == 1 and coords.shape[0] > 1:
            atom_token_index = atom_token_index.expand(coords.shape[0], -1)

        atom_chain_id = token_asym_id.gather(-1, atom_token_index.clamp(min=0))

        mask_bool = mask.bool()
        valid_mask = mask_bool.any(dim=1) & (atom_chain_id >= 0)
        if not valid_mask.any():
            return coords.squeeze(1) if added_sample_dim else coords

        max_chain_id = atom_chain_id.masked_select(valid_mask).max()
        chain_id_stride = max_chain_id + 1
        batch_idx = torch.arange(coords.shape[0], device=coords.device).unsqueeze(-1)
        global_chain_id = atom_chain_id + batch_idx * chain_id_stride

        _, chain_index = torch.unique(global_chain_id[valid_mask], return_inverse=True)
        num_chains = int(chain_index.max().item() + 1)
        chain_index_full = torch.full_like(atom_chain_id, -1)
        chain_index_full[valid_mask] = chain_index

        chain_mask = F.one_hot(
            chain_index_full.clamp(min=0), num_classes=num_chains
        ).bool()
        chain_mask = chain_mask & valid_mask[..., None]
        chain_mask = chain_mask.permute(0, 2, 1)
        chain_mask = chain_mask.unsqueeze(1) & mask_bool.unsqueeze(2)

        chain_coords = coords.unsqueeze(2).masked_fill(~chain_mask[..., None], 0.0)
        chain_counts = chain_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        chain_centers = chain_coords.sum(dim=-2) / chain_counts.to(coords.dtype)
        chain_centers = chain_centers.unsqueeze(-2)
        batch_size, num_samples, num_chains, num_atoms = chain_mask.shape
        flat_coords = chain_coords.reshape(
            batch_size * num_samples * num_chains, num_atoms, 3
        )
        flat_mask = chain_mask.reshape(batch_size * num_samples * num_chains, num_atoms)
        flat_coords = self.random_augmentation(flat_coords, mask=flat_mask)
        chain_coords = flat_coords.reshape(
            batch_size, num_samples, num_chains, num_atoms, 3
        )
        chain_coords = chain_coords + chain_centers * chain_mask[..., None].to(
            chain_coords.dtype
        )
        coords = chain_coords.sum(dim=2)

        if added_sample_dim:
            coords = coords.squeeze(1)

        return coords

    def _sample_uniform_sphere_surface_torch(
        self,
        radius: float,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample points uniformly from a sphere surface (torch).

        Returns tensor of shape (*shape, 3).
        """
        # Sample direction ~ N(0, I), then normalize -> uniform on sphere.
        vec = torch.randn((*shape, 3), dtype=dtype, device=device)
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        return vec * float(radius)

    def _independent_apo_chain_sampling(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        token_asym_id: torch.Tensor,
        atom_token_index: torch.Tensor,
        translation_scale: float,
        chain_com_sampling_radius: float | None,
    ) -> torch.Tensor:
        """Independently sample apo coordinates per (sample, chain).

        This reproduces dataset-side apo initialization behavior for validation/inference:
        - If chain_com_sampling_radius is set: apply chain-wise random rotation (no random
          translation) and then translate each chain so its COM lies on a sphere surface.
        - Else: apply chain-wise random rotation + random translation
          (scale=translation_scale).

        Parameters
        ----------
        coords : torch.Tensor
            Apo coordinates. Shape (B, N, L, 3).
        mask : torch.Tensor
            Apo mask. Shape (B, N, L) or (B, L).
        token_asym_id : torch.Tensor
            Token chain IDs. Shape (B, Lt) or (Lt,).
        atom_token_index : torch.Tensor
            Atom-to-token mapping indices. Shape (B, L) or (L,).
        translation_scale : float
            Random translation scale (Angstrom).
        chain_com_sampling_radius : float | None
            If set, place chain COM on sphere surface of this radius.

        Returns
        -------
        torch.Tensor
            Independently sampled apo coordinates. Shape (B, N, L, 3).
        """
        if not self.coordinate_augmentation:
            return coords

        if coords.dim() != 4:
            raise ValueError(f"Expected coords shape (B, N, L, 3), got {coords.shape}.")

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        if mask.dim() != 3:
            raise ValueError(
                f"Expected mask shape (B, N, L) or (B, L), got {mask.shape}."
            )
        if mask.shape[1] == 1 and coords.shape[1] > 1:
            mask = mask.expand(-1, coords.shape[1], -1)

        # Normalize token/atom mapping shapes
        if token_asym_id.dim() == 1:
            token_asym_id = token_asym_id.unsqueeze(0)
        if atom_token_index.dim() == 1:
            atom_token_index = atom_token_index.unsqueeze(0)
        if token_asym_id.shape[0] == 1 and coords.shape[0] > 1:
            token_asym_id = token_asym_id.expand(coords.shape[0], -1)
        if atom_token_index.shape[0] == 1 and coords.shape[0] > 1:
            atom_token_index = atom_token_index.expand(coords.shape[0], -1)

        atom_chain_id = token_asym_id.gather(-1, atom_token_index.clamp(min=0))

        mask_bool = mask.bool()
        valid_mask = mask_bool.any(dim=1) & (atom_chain_id >= 0)
        if not valid_mask.any():
            return coords

        max_chain_id = atom_chain_id.masked_select(valid_mask).max()
        chain_id_stride = max_chain_id + 1
        batch_idx = torch.arange(coords.shape[0], device=coords.device).unsqueeze(-1)
        global_chain_id = atom_chain_id + batch_idx * chain_id_stride

        _, chain_index = torch.unique(global_chain_id[valid_mask], return_inverse=True)
        num_chains = int(chain_index.max().item() + 1)
        chain_index_full = torch.full_like(atom_chain_id, -1)
        chain_index_full[valid_mask] = chain_index

        # (B, L, C) -> (B, C, L) -> (B, N, C, L)
        chain_mask = F.one_hot(
            chain_index_full.clamp(min=0), num_classes=num_chains
        ).bool()
        chain_mask = chain_mask & valid_mask[..., None]
        chain_mask = chain_mask.permute(0, 2, 1)
        chain_mask = chain_mask.unsqueeze(1) & mask_bool.unsqueeze(2)

        chain_coords = coords.unsqueeze(2).masked_fill(~chain_mask[..., None], 0.0)
        batch_size, num_samples, num_chains, num_atoms = chain_mask.shape

        flat_coords = chain_coords.reshape(
            batch_size * num_samples * num_chains, num_atoms, 3
        )
        flat_mask = chain_mask.reshape(batch_size * num_samples * num_chains, num_atoms)

        # Dataset behavior:
        # - If chain_com_sampling_radius is set: force random translation scale to 0.0
        # - Else: use translation_scale
        if chain_com_sampling_radius is not None:
            translation_scale = 0.0

        augment = CenterRandomAugmentation(
            centering=True,
            augmentation=True,
            s_trans=float(translation_scale),
        )
        flat_coords = augment(flat_coords, mask=flat_mask)
        chain_coords = flat_coords.reshape(
            batch_size, num_samples, num_chains, num_atoms, 3
        )

        if chain_com_sampling_radius is not None:
            # Translate each chain so its COM lies on the sphere surface.
            chain_counts = chain_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            chain_centers = chain_coords.sum(dim=-2) / chain_counts.to(chain_coords.dtype)
            target_centers = self._sample_uniform_sphere_surface_torch(
                radius=float(chain_com_sampling_radius),
                shape=(batch_size, num_samples, num_chains),
                dtype=chain_coords.dtype,
                device=chain_coords.device,
            )
            shift = target_centers - chain_centers  # (B, N, C, 3)
            chain_coords = chain_coords + shift.unsqueeze(-2) * chain_mask[..., None].to(
                chain_coords.dtype
            )

        # Combine chains back: (B, N, C, L, 3) -> (B, N, L, 3)
        coords = chain_coords.sum(dim=2)
        return coords

    def _configure_route_functions(self, cfg: Config) -> None:
        route = (cfg.route_type or "linear").lower().replace("-", "_")
        self.route_type = route
        if route == "linear":
            self._alpha_fn = self._alpha_linear
            self._alpha_deriv_fn = self._alpha_deriv_linear
            self._beta_fn = self._beta_linear
            self._beta_deriv_fn = self._beta_deriv_linear
            self._gamma_fn = self._gamma_linear
            self._gamma_deriv_fn = self._gamma_deriv_linear
            return
        if route == "ddbm_vp":
            self._alpha_fn = self._alpha_ddbm_vp
            self._alpha_deriv_fn = self._alpha_deriv_ddbm_vp
            self._beta_fn = self._beta_ddbm_vp
            self._beta_deriv_fn = self._beta_deriv_ddbm_vp
            self._gamma_fn = self._gamma_ddbm_vp
            self._gamma_deriv_fn = self._gamma_deriv_ddbm_vp
            return
        raise ValueError(
            "Unsupported route_type; expected 'linear' or 'ddbm_vp', "
            f"got {cfg.route_type!r}."
        )

    @property
    def ddbm_vp_a1(self) -> float:
        exponent = 0.5 * self.ddbm_vp_beta_d + self.ddbm_vp_beta_min
        return math.exp(exponent) ** -0.5

    @property
    def ddbm_vp_sigma1_sq(self) -> float:
        exponent = 0.5 * self.ddbm_vp_beta_d + self.ddbm_vp_beta_min
        return math.exp(exponent) - 1.0

    def _ddbm_vp_constants(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = t.new_tensor(self.ddbm_vp_a1)
        sigma1_sq = t.new_tensor(self.ddbm_vp_sigma1_sq)
        return a1, sigma1_sq

    def _ddbm_vp_base(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_d = t.new_tensor(self.ddbm_vp_beta_d)
        beta_min = t.new_tensor(self.ddbm_vp_beta_min)
        log_snr = 0.5 * beta_d * t**2 + beta_min * t
        exp_term = torch.exp(log_snr)
        sigma_sq = exp_term - 1.0
        sigma_sq_prime = exp_term * (beta_d * t + beta_min)
        a_t = torch.rsqrt(exp_term)
        a_t_prime = -0.5 * (beta_d * t + beta_min) * a_t
        return a_t, a_t_prime, sigma_sq, sigma_sq_prime

    # === Route Functions (Stochastic Interpolants) === #
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for target (x_0/holo); route selected by config."""
        return self._alpha_fn(t)

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of alpha; route selected by config."""
        return self._alpha_deriv_fn(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for source (x_T/apo); route selected by config."""
        return self._beta_fn(t)

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of beta; route selected by config."""
        return self._beta_deriv_fn(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Noise scale; route selected by config."""
        return self._gamma_fn(t)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of gamma; route selected by config."""
        return self._gamma_deriv_fn(t)

    # === Linear Route Functions === #
    def _alpha_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for target (x_0/holo): \alpha_t = 1 - t"""
        return 1 - t

    def _alpha_deriv_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of alpha: \dot{\alpha}_t = -1"""
        return -torch.ones_like(t)

    def _beta_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for source (x_T/apo): \beta_t = t"""
        return t

    def _beta_deriv_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of beta: \dot{\beta}_t = 1"""
        return torch.ones_like(t)

    def _gamma_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Noise scale: \gamma_t^2 = \gamma_{max}^2/4 * t(1-t)"""
        return 0.5 * self.gamma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def _gamma_deriv_linear(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative: \dot{\gamma}_t = \gamma_{max} * (1-2t) / (4\sqrt{t(1-t)})"""
        denom = torch.sqrt(t * (1 - t) + 1e-8)
        return self.gamma_max * (1 - 2 * t) / (4 * denom)

    # === DDBM-VP Route Functions (Appendix C.2) === #
    def _alpha_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        denom = sigma1_sq * a_t_sq + 1e-8
        ratio = sigma_sq * a1_sq / denom
        return a_t * (1 - ratio)

    def _alpha_deriv_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
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

    def _beta_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
        denom = sigma1_sq * a_t + 1e-8
        return sigma_sq * a1 / denom

    def _beta_deriv_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
        inv_a_t = 1 / (a_t + 1e-8)
        k = a1 / (sigma1_sq + 1e-8)
        return k * (sigma_sq_prime * inv_a_t - sigma_sq * a_t_prime * inv_a_t * inv_a_t)

    def _gamma_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, _, sigma_sq, _ = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
        a1_sq = a1 * a1
        a_t_sq = a_t * a_t
        ratio = sigma_sq * a1_sq / (sigma1_sq * a_t_sq + 1e-8)
        gamma_sq = sigma_sq * (1 - ratio)
        return torch.sqrt(torch.clamp(gamma_sq, min=0.0) + 1e-8)

    def _gamma_deriv_ddbm_vp(self, t: torch.Tensor) -> torch.Tensor:
        a_t, a_t_prime, sigma_sq, sigma_sq_prime = self._ddbm_vp_base(t)
        a1, sigma1_sq = self._ddbm_vp_constants(t)
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
        if self.synchronize_sigmas:
            shape = (batch_size, 1)
        else:
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

        if self.synchronize_sigmas:
            t = t.repeat(1, num_diffusion_samples)

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

        inv_rho = 1 / self.rho

        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        # t goes from sigma_max (near 1) to sigma_min (near 0)
        times = (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        # Last step is t=0 (exactly at target)
        times = F.pad(times, (0, 1), value=0.0)
        return times

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
        apo_coords : torch.Tensor
            Apo coordinates. Shape (B, N, La, 3).
        """
        if label_coords is None and self.inference_independent_diffusion_apo_sampling:
            # Validation/Inference: independently sample x_T per diffusion sample
            # using parameters from structure module config.
            apo_coords = self.sample_apo(
                f_input, num_diffusion_samples, random_augment=False
            )
            apo_mask = f_input.atom.apo_mask
            apo_coords = self._independent_apo_chain_sampling(
                coords=apo_coords,
                mask=apo_mask,
                token_asym_id=f_input.token.asym_id,
                atom_token_index=f_input.atom.token_index,
                translation_scale=float(self.inference_apo_translation_scale),
                chain_com_sampling_radius=self.inference_apo_chain_com_sampling_radius,
            )
            return apo_coords

        do_random_augment = label_coords is None
        apo_coords = self.sample_apo(f_input, num_diffusion_samples, do_random_augment)

        if self.chain_wise_perturbation:
            apo_mask = f_input.atom.apo_mask
            apo_coords = self.apply_chain_random_augmentation(
                apo_coords, apo_mask, f_input
            )

        if label_coords is not None:
            # apo_mask = ~(apo_coords == 0.0).all(-1)
            apo_coords = self.align_apo_to_label(
                apo_coords,
                label_coords,
                f_input,
            )
            # apo_coords = do_centering(apo_coords, apo_mask, mask_to_zero=True)

        return apo_coords

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

        model_cache = {}

        # Get time schedule (from t_max toward 0)
        times = self.get_sampling_schedule(
            num_steps=num_steps, device=s_inputs.device
        ).tolist()

        atom_mask = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)

        # Sample x_T from prior (apo structures)
        x_apo = self.sample_prior(f_input, num_diffusion_samples)  # (B, N, Latom, 3)

        sample_out["init_coordinates"] = x_apo
        if self.normalize_coordinate:
            x_apo = x_apo / self.sigma_data_end

        x_t = x_apo.clone()

        if return_traj:
            traj.append(x_t.cpu())

        # Reverse time sampling from t=T toward t=0
        for step_idx in range(num_steps):
            # Apply random augmentation
            x_t, x_apo = self.random_augmentation(x_t, x_apo, mask=atom_mask)

            t_curr = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t_curr  # negative since t decreases

            # Convert to tensor
            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]), t_curr, device=x_t.device, dtype=x_t.dtype
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
                    prior_coords=x_apo[:, st:end],
                )

                # align x0_hat to x_apo
                if self.inference_align_x0_hat_to_x_apo:
                    # Kabsch-align x0_hat into the x_apo frame.
                    x0_hat[:, st:end] = self.align_apo_to_label(
                        apo_coords=x0_hat[:, st:end],
                        label_coords=x_apo[:, st:end],
                        f_input=f_input,
                    )

            # Expand t for coefficient computation
            t_exp = t_curr_tensor[:, :, None, None]  # (B, N, 1, 1)

            # Compute route coefficients
            alpha_t = self.alpha(t_exp)
            beta_t = self.beta(t_exp)
            gamma_t = self.gamma(t_exp)
            alpha_dot = self.alpha_deriv(t_exp)
            beta_dot = self.beta_deriv(t_exp)
            gamma_dot = self.gamma_deriv(t_exp)

            # Compute \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
            z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)

            # Last 2 steps: use deterministic update (\epsilon_t = 0)
            # if step_idx >= num_steps - 2:

            ode_time_duration = float(self.ode_time_duration)
            if ode_time_duration > 0.0 and t_curr <= ode_time_duration:
                # x_{t-\Delta t} = \alpha_{t-\Delta t} \hat{x}_0 + \beta_{t-\Delta t} x_T
                #                + \gamma_{t-\Delta t} \hat{z}_t
                t_next_exp = torch.full_like(t_exp, t_next)
                alpha_next = self.alpha(t_next_exp)
                beta_next = self.beta(t_next_exp)
                gamma_next = self.gamma(t_next_exp)

                # NOTE: weghting factor for z_hat is (cos(2pi(t_next-0.5)) + 1) / 2
                weighting_factor = (
                    math.cos(math.pi * (t_next - ode_time_duration) / ode_time_duration)
                    + 1
                ) / 2

                # x_t = alpha_next * x0_hat + beta_next * x_apo + gamma_next * z_hat
                # x_t = alpha_next * x0_hat + beta_next * x_apo
                x_t = (
                    alpha_next * x0_hat
                    + beta_next * x_apo
                    + gamma_next * z_hat * weighting_factor
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
                    + beta_dot * x_apo
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
