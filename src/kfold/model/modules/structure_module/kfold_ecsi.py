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
    - Linear interpolation: \alpha_t=1-t, \beta_t=t,
      \gamma_t^2=\gamma_{max}^2/4 \cdot t(1-t)
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
        alignment_entity_strategy : str, optional
            Strategy for selecting entity to align: "largest" or "random_non_ligand",
            by default "largest".
        """

        num_steps: int = 200
        sigma_min: float = 0.001
        sigma_max: float = 0.999
        gamma_max: float = 0.25
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
        alignment_entity_strategy: str = "largest"
        prior_spread_radius: float = 10.0
        s_trans: float = 1.0

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the ECSI module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.gamma_max: float = cfg.gamma_max
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
        self.prior_spread_radius: float = cfg.prior_spread_radius
        self.s_trans: float = cfg.s_trans

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

    # === Linear Route Functions (Stochastic Interpolants) === #
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for target (x_0/holo): \alpha_t = 1 - t"""
        return 1 - t

    def alpha_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of alpha: \dot{\alpha}_t = -1"""
        return -torch.ones_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""Weight for source (x_T/apo): \beta_t = t"""
        return t

    def beta_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative of beta: \dot{\beta}_t = 1"""
        return torch.ones_like(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Noise scale: \gamma_t^2 = \gamma_{max}^2/4 * t(1-t)"""
        return 0.5 * self.gamma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def gamma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        r"""Derivative: \dot{\gamma}_t = \gamma_{max} * (1-2t) / (4\sqrt{t(1-t)})"""
        denom = torch.sqrt(t * (1 - t) + 1e-8)
        return self.gamma_max * (1 - 2 * t) / (4 * denom)

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
            Denoised (target) atom coordinates \hat{x}_0. Shape (B, N, L, 3).
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

    def _apply_fibonacci_spread(
        self,
        apo_coords: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Apply Fibonacci sphere spreading to apo coordinates.

        Spreads asymmetric units uniformly on a sphere for inputs with >2 units.
        """
        # apo_coords: [B, N_diff, L_atom, 3]
        B, N_diff, L_atom, _ = apo_coords.shape
        device = apo_coords.device

        token_asym_id = f_input.token.asym_id
        atom_token_index = f_input.atom.token_index

        if token_asym_id.dim() == 1:
            token_asym_id = token_asym_id.unsqueeze(0).expand(B, -1)
        if atom_token_index.dim() == 1:
            atom_token_index = atom_token_index.unsqueeze(0).expand(B, -1)

        # Map asym_id to atoms [B, L_atom]
        atom_asym_id = torch.gather(token_asym_id, 1, atom_token_index.clamp(min=0))
        atom_pad_mask = f_input.atom.pad_mask
        if atom_pad_mask.dim() == 1:
            atom_pad_mask = atom_pad_mask.unsqueeze(0).expand(B, -1)

        new_apo_coords = apo_coords.clone()

        for b in range(B):
            # Get valid unique asym_ids
            mask = atom_pad_mask[b]
            b_asym_ids = atom_asym_id[b][mask]
            # Valid asym_ids > 0
            unique_asym_ids = torch.unique(b_asym_ids)
            unique_asym_ids = unique_asym_ids[unique_asym_ids > 0]

            num_units = len(unique_asym_ids)

            if num_units > 2:
                # Generate Fibonacci points on unit sphere
                indices = torch.arange(num_units, dtype=torch.float32, device=device)
                phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
                y = 1 - (indices / float(num_units - 1)) * 2
                radius = torch.sqrt((1 - y * y).clamp(min=0))
                theta = phi * indices

                x = torch.cos(theta) * radius
                z = torch.sin(theta) * radius

                # [num_units, 3]
                sphere_points = torch.stack([x, y, z], dim=1)

                # Scale by radius
                sphere_points = sphere_points * self.prior_spread_radius

                # Random rotation for the sphere
                rot_mat = torch.linalg.qr(torch.randn(3, 3, device=device))[0]
                sphere_points = sphere_points @ rot_mat.T

                for i, uid in enumerate(unique_asym_ids):
                    # Find atoms for this unit
                    unit_mask = (atom_asym_id[b] == uid) & mask
                    if not unit_mask.any():
                        continue

                    target_center = sphere_points[i]  # [3]

                    # Move unit center to target_center for each diffusion sample
                    # coords: [N_diff, L_atom, 3] view
                    b_coords = new_apo_coords[b]

                    # Get current COM of the unit (per sample)
                    # unit_vals: [N_diff, N_unit_atoms, 3]
                    unit_vals = b_coords[:, unit_mask, :]
                    com = unit_vals.mean(dim=1, keepdim=True)  # [N_diff, 1, 3]

                    # Translate
                    new_apo_coords[b, :, unit_mask, :] = (unit_vals - com) + target_center

        return new_apo_coords

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
        do_random_augment = label_coords is None
        apo_coords = self.sample_apo(f_input, num_diffusion_samples, do_random_augment)

        # Apply Fibonacci sphere spreading for multi-chain complexes
        if self.prior_spread_radius > 0:
            apo_coords = self._apply_fibonacci_spread(apo_coords, f_input)

        if label_coords is not None and self.alignment_entity_strategy:
            apo_coords = self.align_apo_to_label_by_entity_selection(
                apo_coords,
                label_coords,
                f_input,
            )

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
            if step_idx >= num_steps - 2:
                # x_{t-\Delta t} = \alpha_{t-\Delta t} \hat{x}_0 + \beta_{t-\Delta t} x_T
                #                + \gamma_{t-\Delta t} \hat{z}_t
                t_next_exp = torch.full_like(t_exp, t_next)
                alpha_next = self.alpha(t_next_exp)
                beta_next = self.beta(t_next_exp)
                gamma_next = self.gamma(t_next_exp)
                x_t = alpha_next * x0_hat + beta_next * x_apo + gamma_next * z_hat
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
