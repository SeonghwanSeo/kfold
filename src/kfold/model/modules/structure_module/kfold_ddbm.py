# started from code from https://github.com/jwohlwend/boltz, MIT License
# adapted with DDBM bridge diffusion approach

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.utils import CenterRandomAugmentation
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseStructureModule


@STRUCTURE_MODULE.register()
class KFoldBridgeDiffusion(BaseStructureModule):
    """Diffusion Bridge module for biomolecular structure prediction.

    Implements a bridge diffusion process that transitions from apo (unbound)
    to holo (bound) protein structures, adapting the DDBM (Diffusion Bridge)
    approach from the image domain to biomolecular structure prediction.

    Reference:
    - DDBM: Zhou et al., "Denoising Diffusion Bridge Models"
    - EDM: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models"
    """

    @dataclass
    class Config(BaseConfig):
        """Configuration for the Bridge Diffusion Structure module.

        Parameters
        ----------
        num_steps : int, optional
            The number of sampling steps, by default 200.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the source data distribution (apo structures), by default 16.0.
        sigma_data_end : float, optional
            The standard deviation of the target data distribution (holo structures), by default 16.0.
        cov_xy : float, optional
            The covariance between source and target (sigma_0T in paper), by default 128.0 (= sigma_data^2 / 2).
            This controls the c_skip coefficient: 0 means no correlation (pure exploration),
            sigma_data^2/2 means c_skip=0.5 at sigma_max (moderate guidance).

            IMPORTANT: This hyperparameter encodes the expected correlation between apo (source)
            and holo (target) structures in the bridge distribution q(x0, xT). For biomolecular
            structures, apo and holo conformations are typically correlated (shared backbone),
            so cov_xy > 0 is appropriate. The default value sigma_data^2/2 provides moderate
            correlation, resulting in c_skip=0.5 at sigma_max, balancing the skip connection
            between input and network output in the preconditioning (see Eq. 11-12, Appendix A.5).
        c : float, optional
            Noise scaling factor, by default 1.0.
        rho : int, optional
            The rho value for Karras schedule, by default 7.
        P_mean : float, optional
            The mean value of P for noise level sampling, by default -1.2.
        P_std : float, optional
            The standard deviation of P for noise level sampling, by default 1.5.
        noise_scale : float, optional
            The noise scale for stochastic sampling, by default 1.003.
        step_scale : float, optional
            The step scale for ODE integration, by default 1.5.
        churn_step_ratio : float, optional
            Churn step ratio for adding controlled stochasticity, by default 0.33.
            This parameter controls the strength of the stochastic Euler-Maruyama step
            inserted before each deterministic Heun step. Higher values add more noise
            for exploration. Set to 0.0 for pure deterministic ODE sampling.
            Recommended value: 0.33 (from DDBM paper).
        guidance_weight : float, optional
            Guidance weight for fidelity to target structure, by default 1.0.
            Higher values increase fidelity to the conditioning (holo) structure.

            THEORETICAL NOTE: This corresponds to parameter 'w' in DDBM paper Eq. 13:
            dxt = [f(xt,t) - g^2(t)(0.5s(xt,t,y,T) - wh(xt,t,y,T))]dt
            where h is Doob's h-transform. Setting w≠1 modulates the "strength" of drift
            adjustment towards the target endpoint, allowing exploration of a wider class
            of marginal densities. However, w≠1 changes the marginal distribution of the
            bridge process, so use with caution. For faithful bridge sampling, use w=1.0.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
            This is important for SE(3)-equivariant biomolecular modeling.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas across diffusion samples, by default False.
        """

        num_steps: int = 200
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        sigma_data_end: float = 16.0
        cov_xy: float = 128.0  # sigma_data^2 / 2
        c: float = 1.0
        rho: int = 7
        P_mean: float = -1.2
        P_std: float = 1.5
        noise_scale: float = 1.003
        step_scale: float = 1.5
        churn_step_ratio: float = 0.33
        guidance_weight: float = 1.0
        coordinate_augmentation: bool = True
        synchronize_sigmas: bool = False

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the bridge diffusion module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.sigma_data: float = cfg.sigma_data
        self.sigma_data_end: float = cfg.sigma_data_end
        self.cov_xy: float = cfg.cov_xy
        self.c: float = cfg.c
        self.rho: int = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.num_steps: int = cfg.num_steps
        self.noise_scale: float = cfg.noise_scale
        self.step_scale: float = cfg.step_scale
        self.churn_step_ratio: float = cfg.churn_step_ratio
        self.guidance_weight: float = cfg.guidance_weight
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation
        self.synchronize_sigmas: bool = cfg.synchronize_sigmas

        if self.coordinate_augmentation:
            self.random_augmentation = CenterRandomAugmentation(
                centering=True,
                random_rotate=self.coordinate_augmentation,
            )

    # === Bridge EDM diffusion coefficients === #
    def _get_bridge_scalings(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bridge diffusion scalings for variance-exploding (ve) mode.

        Based on DDBM formulation for image-to-image translation,
        adapted for biomolecular structure prediction (apo -> holo).
        Note that alpha_t=1, sigma_t=sigma for bridge VE diffusion.
        Also, self.c is only introduced in DDBM code level, which is 1 by default.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise levels. Shape (B, N) or scalar.

        Returns
        -------
        c_skip : torch.Tensor
            Skip connection coefficient.
        c_out : torch.Tensor
            Output scaling coefficient.
        c_in : torch.Tensor
            Input scaling coefficient.
        """
        # Compute A: total variance at noise level sigma
        # A = (sigma^4/sigma_max^4)sigma_end^2 + (1-sigma^2/sigma_max^2)^2sigma_data^2
        #     + 2(sigma^2/sigma_max^2)(1-sigma^2/sigma_max^2)cov_xy
        #     + c^2sigma^2(1-sigma^2/sigma_max^2)
        sigma_ratio_sq = sigma**2 / self.sigma_max**2  # a_t in DDDBM (p. 19)
        sigma_ratio_4th = sigma_ratio_sq**2  # need this to compute c_t in DDBM (p. 19)
        one_minus_ratio_sq = 1 - sigma_ratio_sq  # b_t in DDDBM (p. 19)
        sigma_sq_one_minus_ratio_sq = sigma**2 * one_minus_ratio_sq  # c_t in DDBM (p. 19)

        # square of denominator of c_in
        # a_t^2 * sigma_end^2 + b_t^2 * sigma_data^2 + c_t
        A = (
            sigma_ratio_4th * self.sigma_data_end**2
            + one_minus_ratio_sq**2 * self.sigma_data**2
            + 2 * sigma_ratio_sq * one_minus_ratio_sq * self.cov_xy
            + self.c**2 * sigma_sq_one_minus_ratio_sq
        )

        # c_in: input normalization (Eq. 81)
        c_in = 1 / torch.sqrt(A)

        # c_skip: skip connection weight (Eq. 82)
        # Controls how much of the input x_t is passed through
        numerator_skip = (
            one_minus_ratio_sq * self.sigma_data**2 + sigma_ratio_sq * self.cov_xy
        )
        c_skip = numerator_skip / A

        # c_out: output scaling (Eq. 83)
        # Controls the magnitude of the network output
        numerator_out_sq = (
            sigma_ratio_4th
            * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2)
            + self.sigma_data**2 * self.c**2 * sigma_sq_one_minus_ratio_sq**2
        )
        c_out = torch.sqrt(numerator_out_sq) * c_in

        return c_skip, c_out, c_in

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection coefficient for bridge diffusion preconditioning."""
        c_skip, _, _ = self._get_bridge_scalings(sigma)
        return c_skip

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output scaling coefficient for bridge diffusion preconditioning."""
        _, c_out, _ = self._get_bridge_scalings(sigma)
        return c_out

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling coefficient for bridge diffusion preconditioning."""
        _, _, c_in = self._get_bridge_scalings(sigma)
        return c_in

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise level conditioning coefficient (same as EDM)."""
        return (sigma / self.sigma_data).clamp(1e-20).log() * 0.25

    def compute_loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat.

        Uses bridge Karras weighting that accounts for the correlation
        structure between source (apo) and target (holo) structures.
        See w(t) in Eq. 12 of DDBM paper.

        Parameters
        ----------
        t_hat : torch.Tensor
            Noise levels. Shape (B, N).

        Returns
        -------
        weights : torch.Tensor
            Loss weights. Shape (B, N).
        """
        sigma = t_hat
        _, c_out, _ = self._get_bridge_scalings(sigma)
        weights = 1 / c_out**2
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
    ) -> torch.Tensor:
        """Forward pass through the score model with bridge preconditioning.

        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper,
        adapted with bridge diffusion scalings.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t_hat : torch.Tensor | float
            Diffusion noise level (or sigmas). Shape (B, N).
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

        Returns
        -------
        denoised_coords : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 3).
        """
        if not isinstance(t_hat, torch.Tensor):
            t_hat = torch.full(
                x_noisy.shape[:2], t_hat, device=x_noisy.device, dtype=x_noisy.dtype
            )  # [B, N]

        x_out = self.score_model(
            x_noisy=x_noisy,  # [B, N, La, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            model_cache=model_cache,
        )
        return x_out

    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_steps: int | None = None,
        num_diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
        use_ground_truth_holo: bool = False,
    ) -> torch.Tensor:
        """Sample structures via bridge diffusion sampling using Heun's method.

        Implements a bridge diffusion process that transitions from apo structures
        to holo structures, using the DDBM ODE formulation.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, Lt, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, Lt, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, Lt, Lt, c_z).
        num_steps : int, optional
            Number of sampling steps. If None, uses self.num_steps.
        num_diffusion_samples : int, optional
            Number of diffusion samples, by default 1.
        max_parallel_samples : int, optional
            Maximum number of parallel samples for memory efficiency.
            If None, processes all samples in parallel.
        use_ground_truth_holo : bool, optional
            Whether to use ground truth holo structures (training mode) or
            progressive refinement (inference mode). By default False.
            - If True (training): Uses ground truth holo from f_input as fixed target.
            - If False (inference): Uses progressive refinement with denoised predictions.

        Returns
        -------
        atom_coords : torch.Tensor
            Sampled atom coordinates. Shape (B, N, Latom, 3).
        """

        if num_steps is None:
            num_steps = self.num_steps

        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples

        # Get noise schedule
        sigmas = self.get_sampling_schedule(num_steps=num_steps, device=s_inputs.device)
        sigmas = sigmas.tolist()

        atom_mask = f_input.atom.pad_mask.float().unsqueeze(1)  # (B, 1, Latom)

        # Model cache for efficiency
        model_cache: dict = {}

        # Line 1: Initialize from apo structure (source)
        x_apo = self.sample_prior(f_input, num_diffusion_samples)  # (B, N, Latom, 3)
        atom_coords = x_apo.clone()

        # Store target (holo) structure for bridge conditioning (training mode only)
        x_holo = None
        if use_ground_truth_holo:
            x_holo = self.sample_holo(f_input, num_diffusion_samples)  # (B, N, Latom, 3)

        # Line 2: Heun sampling loop
        for step_idx in range(len(sigmas) - 1):
            sigma_curr = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # TODO: Check random augmentation in our setting is valid
            if self.coordinate_augmentation and step_idx > 0:
                atom_coords = self.random_augmentation(atom_coords, atom_mask=atom_mask)

            # === Churn step: stochastic Euler-Maruyama step for exploration ===
            if self.churn_step_ratio > 0:
                # Compute intermediate noise level
                sigma_hat = (sigma_next - sigma_curr) * self.churn_step_ratio + sigma_curr

                # Get denoised prediction at current sigma
                atom_coords_denoised_churn = torch.zeros_like(atom_coords)
                for st in range(0, num_diffusion_samples, max_parallel_samples):
                    end = min(st + max_parallel_samples, num_diffusion_samples)
                    atom_coords_denoised_churn[:, st:end] = self.forward_model(
                        x_noisy=atom_coords[:, st:end],
                        t_hat=sigma_curr,
                        f_input=f_input,
                        s_inputs=s_inputs,
                        s_trunk=s_trunk,
                        z_trunk=z_trunk,
                        model_cache=model_cache,
                    )

                # Compute drift and diffusion with stochastic=True
                d_churn, gt2 = self._to_d_bridge(
                    x=atom_coords,
                    sigma=sigma_curr,
                    denoised=atom_coords_denoised_churn,
                    x_apo=x_apo,
                    x_holo=x_holo,
                    guidance_weight=self.guidance_weight,
                    use_progressive_refinement=not use_ground_truth_holo,
                    stochastic=True,
                )

                # Euler-Maruyama step: deterministic drift + stochastic diffusion
                dt_churn = sigma_hat - sigma_curr
                noise = torch.randn_like(atom_coords)
                atom_coords = (
                    atom_coords
                    + d_churn * dt_churn
                    + noise * torch.sqrt(torch.abs(dt_churn)) * torch.sqrt(gt2)
                )

                # Update sigma_curr for Heun step
                sigma_curr = sigma_hat

            # First-order estimate
            # Process in chunks for memory efficiency
            atom_coords_denoised = torch.zeros_like(atom_coords)
            for st in range(0, num_diffusion_samples, max_parallel_samples):
                end = min(st + max_parallel_samples, num_diffusion_samples)
                atom_coords_denoised[:, st:end] = self.forward_model(
                    x_noisy=atom_coords[:, st:end],
                    t_hat=sigma_curr,
                    f_input=f_input,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    model_cache=model_cache,
                )

            # Compute ODE derivative using bridge formulation
            d = self._to_d_bridge(
                x=atom_coords,
                sigma=sigma_curr,
                denoised=atom_coords_denoised,
                x_apo=x_apo,
                x_holo=x_holo,
                guidance_weight=self.guidance_weight,
                use_progressive_refinement=not use_ground_truth_holo,
            )

            dt = sigma_next - sigma_curr

            if sigma_next == 0:
                # Final step: Euler step
                atom_coords = atom_coords + d * dt
            else:
                # Heun's method: second-order correction
                atom_coords_euler = atom_coords + d * dt

                # Second-order estimate
                atom_coords_denoised_2 = torch.zeros_like(atom_coords_euler)
                for st in range(0, num_diffusion_samples, max_parallel_samples):
                    end = min(st + max_parallel_samples, num_diffusion_samples)
                    atom_coords_denoised_2[:, st:end] = self.forward_model(
                        x_noisy=atom_coords_euler[:, st:end],
                        t_hat=sigma_next,
                        f_input=f_input,
                        s_inputs=s_inputs,
                        s_trunk=s_trunk,
                        z_trunk=z_trunk,
                        model_cache=model_cache,
                    )

                d_2 = self._to_d_bridge(
                    x=atom_coords_euler,
                    sigma=sigma_next,
                    denoised=atom_coords_denoised_2,
                    x_apo=x_apo,
                    x_holo=x_holo,
                    guidance_weight=self.guidance_weight,
                    use_progressive_refinement=not use_ground_truth_holo,
                )

                # Average derivatives
                d_avg = (d + d_2) / 2

                # Update with averaged derivative
                # NOTE: DDBM used step_scale=1.0
                atom_coords = atom_coords + self.step_scale * d_avg * dt

        return atom_coords

    def _to_d_bridge(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        denoised: torch.Tensor,
        x_apo: torch.Tensor | None = None,
        x_holo: torch.Tensor | None = None,
        guidance_weight: float = 1.0,
        use_progressive_refinement: bool = False,
        stochastic: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert denoiser output to bridge ODE derivative.

        Implements the bridge diffusion ODE that balances:
        1. Denoising towards predicted structure (grad w.r.t. x_0)
        2. Guidance towards target holo structure (grad w.r.t. x_T)

        This implements Theorem 1 (Eq. 7) from DDBM paper:
        dxt = [f(xt,t) - g^2(t)(0.5s(xt,t,y,T) - h(xt,t,y,T))]dt

        where s(xt,t,y,T) is the bridge score \nabla_xt log q(xt | xT) and
        h(xt,t,y,T) is Doob's h-transform \nabla_xt log p(xT | xt).

        Parameters
        ----------
        x : torch.Tensor
            Current state. Shape (B, N, La, 3).
        sigma : float or torch.Tensor
            Current noise level.
        denoised : torch.Tensor
            Denoised prediction. Shape (B, N, La, 3).
            - If use_progressive_refinement=False: approximates x_0 (apo/source)
            - If use_progressive_refinement=True: approximates x_T (holo/target)
        x_apo : torch.Tensor, optional
            Source apo structure (x_0 endpoint). Required if use_progressive_refinement=True.
            Shape (B, N, La, 3).
        x_holo : torch.Tensor, optional
            Target holo structure (x_T endpoint). Required if use_progressive_refinement=False.
            Shape (B, N, La, 3).
        guidance_weight : float, optional
            Weight for guidance towards target, by default 1.0.
        use_progressive_refinement : bool, optional
            If True, use denoised as progressive x_holo estimate (inference mode).
            If False, use fixed x_holo ground truth (training mode).
            By default False.
        stochastic : bool, optional
            If True, use stochastic mode for Euler-Maruyama churn step.
            In stochastic mode: guidance is disabled (w=0), full coefficient (1.0),
            and returns both drift and diffusion coefficient.
            By default False.

        Returns
        -------
        d : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If stochastic=False: ODE derivative d. Shape (B, N, La, 3).
            If stochastic=True: Tuple of (drift d, diffusion gt2). Both Shape (B, N, La, 3).
        """
        if isinstance(sigma, float):
            sigma_tensor = torch.tensor(sigma, device=x.device, dtype=x.dtype)
        else:
            sigma_tensor = sigma

        # Expand sigma to match x dimensions if needed
        if sigma_tensor.ndim < x.ndim:
            # sigma is (B, N), x is (B, N, La, 3)
            while sigma_tensor.ndim < x.ndim:
                sigma_tensor = sigma_tensor.unsqueeze(-1)

        # Determine endpoints based on mode
        if use_progressive_refinement:
            # Inference mode: model predicts x_T (holo), x_apo is fixed
            if x_apo is None:
                raise ValueError("x_apo is required when use_progressive_refinement=True")
            x_holo_estimate = denoised  # Progressive estimate of target
            x_apo_fixed = x_apo  # Known source
        else:
            # Training mode: model predicts x_0 (apo), x_holo is fixed ground truth
            if x_holo is None:
                raise ValueError("x_holo is required when use_progressive_refinement=False")
            x_holo_estimate = x_holo  # Ground truth target
            x_apo_fixed = denoised  # Model's prediction of source

        # === Bridge score: \nabla_xt log q(xt | x0, xT) ===
        # From Eq. 8 in DDBM paper, the bridge distribution is:
        # q(xt | x0, xT) = N(\hat{mu}t, \hat{sigma}^2t I) where:
        #   \hat{mu}t = (sigma^2/sigma^2_max)*xT + (1 - sigma^2/sigma^2_max)*x0
        #   \hat{sigma}^2t = sigma^2(1 - sigma^2/sigma^2_max)
        # Therefore: \nabla_xt log q(xt | x0, xT) = -(xt - \hat{mu}t) / \hat{sigma}^2t

        # Compute bridge coefficients
        sigma_ratio_sq = sigma_tensor**2 / (self.sigma_max**2)
        at = sigma_ratio_sq  # weight for x_holo (xT)
        bt = 1 - sigma_ratio_sq  # weight for x_apo (x0)
        ct = sigma_tensor**2 * (1 - sigma_ratio_sq)  # bridge variance \hat{sigma}^2t

        # Mean of bridge distribution
        mu_t = at * x_holo_estimate + bt * x_apo_fixed

        # Bridge score (gradient w.r.t. bridge mean)
        grad_pxtlx0 = -(x - mu_t) / ct

        # === Doob's h-transform: \nabla_xt log p(xT | xt) ===
        # From Table 1 (VE bridge): \nabla_xt log p(xT | xt) = (xT - xt)/(sigma^2_T - sigma^2_t)
        grad_pxTlxt = (x_holo_estimate - x) / (self.sigma_max**2 - sigma_tensor**2)

        # === Bridge ODE derivative ===
        # THEORETICAL NOTE: g^2(sigma) parameterization in sigma-space
        #
        # The DDBM paper formulates the ODE in time t (Eq. 7):
        #   dxt = [f(xt,t) - g^2(t)(0.5s - h)]dt
        #
        # For VE bridges: f(xt,t) = 0, g^2(t) = d/dt sigma^2t
        #
        # We use sigma as the integration variable instead of t (Karras et al., EDM).
        # With the change of variables dt -> dsigma:
        #   dxt/dsigma = (dxt/dt) * (dt/dsigma) = [- g^2(t)(0.5s - h)] * (dt/dsigma)
        #
        # For variance-exploding SDE: dsigma^2t = g^2(t)dt
        # Therefore: g^2(t) = dsigma^2t/dt = 2sigmat * dsigmat/dt
        # And: dt/dsigma = 1/(dsigma/dt), so: (dt/dsigma) * g^2(t) = 2sigmat
        #
        # This gives: dxt/dsigma = -2sigmat * (0.5s - w*h) = -sigmat * (s - 2w*h)
        # Rearranging: d = -0.5 * 2sigmat * (s - w*h)
        #
        # In sigma-parameterization with dsigma as integration variable:
        gt2 = 2 * sigma_tensor

        # Stochastic mode for Euler-Maruyama churn step
        if stochastic:
            # Stochastic Euler step: disable guidance, use full coefficient
            # DDBM formulation: d = -gt2 * grad_pxtlx0 (no Doob's h-transform term)
            d = -gt2 * grad_pxtlx0
            return d, gt2
        else:
            # Deterministic ODE step: standard bridge formulation
            # Final ODE derivative: d = -0.5 * g^2(sigma) * (s - w*h)
            d = -0.5 * gt2 * (grad_pxtlx0 - guidance_weight * grad_pxTlxt)
            return d

    def sample_noise_level(
        self,
        batch_size: int,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels for training.

        Uses log-normal distribution as in EDM.

        Parameters
        ----------
        batch_size : int
            Batch size.
        num_diffusion_samples : int
            Number of diffusion samples.
        device : torch.device, optional
            Device for tensor allocation.

        Returns
        -------
        t_hat : torch.Tensor
            Sampled noise levels. Shape (B, N).
        """

        def _sample(*shape: int) -> torch.Tensor:
            return self.sigma_data * torch.exp(
                self.P_mean + self.P_std * torch.randn(shape, device=device)
            )

        if self.synchronize_sigmas:
            # synchronize sigmas across diffusion samples
            return _sample(batch_size, 1).expand(-1, num_diffusion_samples)
        else:
            # use different sigmas for each diffusion sample
            return _sample(batch_size, num_diffusion_samples)

    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the Karras noise schedule for diffusion sampling.

        Parameters
        ----------
        num_steps : int, optional
            Number of sampling steps. If None, uses self.num_steps.
        device : torch.device, optional
            Device for tensor allocation.

        Returns
        -------
        sigmas : torch.Tensor
            Noise schedule. Shape (num_steps + 1,), ending with 0.
        """

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

    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the prior distribution (apo structures).

        In bridge diffusion, the prior is the apo (unbound) structure,
        which serves as the starting point for the bridge process.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples, by default 1.

        Returns
        -------
        apo_coords : torch.Tensor
            Apo coordinates. Shape (B, N, La, 3).
        """
        # Use apo structures as the source/prior
        apo_coords = self.sample_apo(f_input, num_diffusion_samples)  # [B, N, La, 3]

        # Apply coordinate augmentation if enabled
        if self.coordinate_augmentation:
            atom_mask = f_input.atom.pad_mask.float()  # (B, La)
            B, N, L = apo_coords.shape[:3]
            apo_coords = apo_coords.view(B * N, L, 3)  # (B*N, La, 3)
            atom_mask = atom_mask.repeat_interleave(N, dim=0)  # (B * N, La)

            # Apply coordinate augmentation
            apo_coords = self.random_augmentation(apo_coords, atom_mask=atom_mask)

            # Mask out the padding atoms
            apo_coords = apo_coords * atom_mask[:, :, None]  # (B*N, La, 3)

            apo_coords = apo_coords.view(B, N, L, 3)

        return apo_coords

    def sample_holo(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
    ) -> torch.Tensor:
        """Sample holo structures (target for bridge diffusion).

        In bridge diffusion, the holo (bound) structure is the target
        that we condition on during sampling.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples, by default 1.

        Returns
        -------
        holo_coords : torch.Tensor
            Holo coordinates. Shape (B, N, La, 3).
        """
        holo_coords = super().sample_holo(
            f_input, num_diffusion_samples
        )  # [B, N, Latom, 3]

        if self.coordinate_augmentation:
            atom_mask = f_input.atom.pad_mask.float()  # (B, Latom)

            B, N, L = holo_coords.shape[:3]
            holo_coords = holo_coords.view(B * N, L, 3)  # (B*N, Latom, 3)
            atom_mask = atom_mask.repeat_interleave(N, dim=0)  # (B * N, Latom)

            # Apply coordinate augmentation
            holo_coords = self.random_augmentation(holo_coords, atom_mask=atom_mask)

            # Mask out the padding atoms
            holo_coords = holo_coords * atom_mask[:, :, None]  # (B*N, Latom, 3)

            holo_coords = holo_coords.view(B, N, L, 3)

        return holo_coords

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between apo and holo structures using bridge diffusion.
        Note that t is the same as sigma in EDM/bridge diffusion formulation.

        Implements the bridge diffusion interpolation formula:

        mu_t = (t^2/sigma_max^2) * x_holo + (1 - t^2/sigma_max^2) * x_apo
        std_t = t * sqrt(1 - t^2/sigma_max^2)
        x_t = mu_t + std_t * noise

        Parameters
        ----------
        noise_coords : torch.Tensor
            The apo (source) coordinates. Shape (B, N, La, 3).
        label_coords : torch.Tensor
            The holo (target) coordinates. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            The noise levels (sigma values). Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).

        Returns
        -------
        noised_coords : torch.Tensor
            Bridge-sampled coordinates. Shape (B, N, La, 3).
        """
        # noise_coords is x_apo (source), label_coords is x_holo (target)
        x_apo = noise_coords
        x_holo = label_coords

        # Expand t_hat to match coordinate dimensions
        t_expanded = t_hat[:, :, None, None]  # (B, N, 1, 1)

        # Compute interpolation weights
        sigma_ratio_sq = t_expanded**2 / self.sigma_max**2
        weight_holo = sigma_ratio_sq
        weight_apo = 1 - sigma_ratio_sq

        # Mean of bridge distribution
        mu_t = weight_holo * x_holo + weight_apo * x_apo

        # Standard deviation of bridge distribution
        # std_t = t * sqrt(1 - t^2/sigma_max^2)
        std_t = t_expanded * torch.sqrt(1 - sigma_ratio_sq)

        # Sample from bridge distribution
        noise = torch.randn_like(x_apo)
        noised_coords = mu_t + std_t * noise

        # Mask out the padding atoms
        noised_coords = noised_coords * mask[:, None, :, None]

        return noised_coords
