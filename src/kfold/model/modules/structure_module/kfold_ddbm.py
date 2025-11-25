# started from code from https://github.com/jwohlwend/boltz, MIT License
# adapted with DDBM bridge diffusion approach

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.utils import CenterRandomAugmentation
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.geometry.rigid_align import rigid_align
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
    - EDM: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
                           Models"
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
            The standard deviation of the target data distribution (holo structures),
            by default 16.0.
        sigma_data_end : float, optional
            The standard deviation of the source data distribution (apo structures),
            by default 16.0.
        cov_xy : float, optional
            The covariance between source and target (sigma_0T in paper),
            by default 128.0 (= sigma_data^2 / 2).
            This controls the c_skip coefficient:
                0 means no correlation (pure exploration),
                sigma_data^2/2 means c_skip=0.5 at sigma_max (moderate guidance).

            IMPORTANT: This hyperparameter encodes the expected correlation between apo
            (source) and holo (target) structures in the bridge distribution q(x0, xT).
            For biomolecular structures, apo and holo conformations are typically
            correlated (shared backbone), so cov_xy > 0 is appropriate. The default value
            sigma_data^2/2 provides moderate correlation, resulting in c_skip=0.5 at
            sigma_max, balancing the skip connection between input and network output
            in the preconditioning (see Eq. 11-12, Appendix A.5).
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
                augmentation=self.coordinate_augmentation,
                s_trans=1.0,  # not used when augmentation is False
            )

    # === Bridge EDM diffusion coefficients === #
    def _get_bridge_scalings(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute bridge diffusion scalings for variance-exploding (ve) mode.

        Based on DDBM formulation for image-to-image translation,
        adapted for biomolecular structure prediction (apo -> holo).
        Note that alpha_t=1, sigma_t=sigma for bridge VE diffusion.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise levels. Shape (B, N) or scalar. (\sigma_t in DDBM paper)

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
        # A = (sigma^4/sigma_data_end^4)sigma_data_end^2
        #     + (1-sigma^2/sigma_data_end^2)^2sigma_data^2
        #     + 2(sigma^2/sigma_data_end^2)(1-sigma^2/sigma_data_end^2)cov_xy
        #     + sigma^2(1-sigma^2/sigma_data_end^2)
        T = self.sigma_max * self.sigma_data
        a_t = sigma**2 / T**2  # a_t in DDBM (p. 19)
        b_t = 1 - a_t  # b_t in DDBM (p. 19)
        c_t = sigma**2 * b_t  # c_t in DDBM (p. 19)

        # square of denominator of c_in
        # a_t^2 * sigma_T^2 + b_t^2 * sigma_0^2 + 2 * a_t * b_t * sigma_0T + c_t
        A = (
            a_t**2 * self.sigma_data_end**2
            + b_t**2 * self.sigma_data**2
            + 2 * a_t * b_t * self.cov_xy
            + c_t
        )

        # c_in: input normalization (Eq. 81)
        c_in = 1 / torch.sqrt(A)

        # c_skip: skip connection weight (Eq. 82)
        # Controls how much of the input x_t is passed through
        numerator_skip = b_t * self.sigma_data**2 + a_t * self.cov_xy
        c_skip = numerator_skip / A

        # c_out: output scaling (Eq. 83)
        # Controls the magnitude of the network output
        numerator_out_sq = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2)
            + self.sigma_data**2 * c_t
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

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
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
        c_out = self.c_out(t_hat)
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
        t_hat_reshaped = t_hat[..., None, None]  # [B, N, 1, 1]

        # Line 2 of Algorithm 20: Input preconditioning
        r_noisy = self.c_in(t_hat_reshaped) * x_noisy

        # Line 8 of Algorithm 21: Noise level conditioning
        c_noise = self.c_noise(t_hat)  # [B, N]

        # Call the score model with correct interface
        r_update = self.score_model(
            r_noisy=r_noisy,  # [B, N, La, 3]
            c_noise=c_noise,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            model_cache=model_cache,
        )

        # Line 8 of Algorithm 20: Output preconditioning
        x_out = (
            self.c_skip(t_hat_reshaped) * x_noisy
            + self.c_out(t_hat_reshaped) * r_update  # [B, N, La, 3]
        )
        return x_out

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
            return (
                torch.exp(
                    self.P_mean + self.P_std * torch.randn(shape, device=device)
                ).clamp(max=self.sigma_max)
                * self.sigma_data
            )

        if self.synchronize_sigmas:
            # synchronize sigmas across diffusion samples
            return _sample(batch_size, 1).repeat(1, num_diffusion_samples)
        else:
            # use different sigmas for each diffusion sample
            return _sample(batch_size, num_diffusion_samples)

    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the Karras noise schedule for diffusion sampling.
        See Supp. A.6 of DDBM paper for details.

        Parameters
        ----------
        num_steps : int, optional
            Number of sampling steps. If None, uses self.num_steps.
        device : torch.device, optional
            Device for tensor allocation.

        Returns
        -------
        sigmas : torch.Tensor
            Noise schedule. Shape (num_steps + 1,), starting with sigma_data (t_N) and
            ending with 0 (t_0).
        """

        if num_steps is None:
            num_steps = self.num_steps

        inv_rho = 1 / self.rho

        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        # sigma_max = T, sigma_min = t_min
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        # last step is sigma value of 0.
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
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

        # We skip random rotation when label_coords is given since the apo coords
        # would be aligned to holo.
        do_random_augment = label_coords is None
        apo_coords = self.sample_apo(f_input, num_diffusion_samples, do_random_augment)

        if label_coords is not None:
            # Align to label coordinates
            # NOTE (SeonghwanSeo): Actually, this masking is not rigorous since
            # the apo coordinates of single ions are always zeros(0,0,0). However,
            # this does not harm the performance.
            apo_mask = ~(apo_coords == 0.0).all(-1)
            label_mask = ~(label_coords == 0.0).all(-1)

            apo_coords = rigid_align(
                coords=apo_coords,
                target=label_coords,
                mask=apo_mask & label_mask,
            )
            # Centering to zero
            apo_coords = do_centering(apo_coords, apo_mask, mask_to_zero=True)

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
            holo_coords = self.random_augmentation(holo_coords, mask=atom_mask)

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
        T = self.sigma_max * self.sigma_data
        a_t = t_expanded**2 / T**2  # a_t
        b_t = 1 - a_t  # b_t (weight for x_0)

        # Mean of bridge distribution
        mu_t = a_t * x_apo + b_t * x_holo  # a_t x_T + b_t x_0

        std_t = t_expanded * torch.sqrt(b_t)  # sqrt(c_t) = sigma_t * sqrt(b_t)

        # Sample from bridge distribution
        noise = torch.randn_like(x_apo)
        noised_coords = mu_t + std_t * noise

        # Mask out the padding atoms
        noised_coords = noised_coords * mask[:, None, :, None]

        return noised_coords

    def sample_structure(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.
        """
        # TODO: Implement this
        raise NotImplementedError(
            "Structure sampling not yet implemented for KFoldBridgeDiffusion."
        )
