# started from code from https://github.com/jwohlwend/boltz, MIT License

import math

import torch
import torch.nn.functional as F

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.utils import CenterRandomAugmentation
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseStructureModule


@STRUCTURE_MODULE.register()
class AF3AtomDiffusion(BaseStructureModule):
    """Atom diffusion module used in AlphaFold3.
    See Section 3.7 Algorithm 18: SampleDiffusion in the AF3 paper.
    """

    class Config(BaseConfig):
        """Configuration for the Structure module.

        Parameters
        ----------
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
            This may be useful for non-equivariant score models.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        """

        num_sampling_steps: int = 5
        sigma_min: float = 0.0004
        sigma_max: float = 160.0
        sigma_data: float = 16.0
        rho: int = 7
        P_mean: float = -1.2
        P_std: float = 1.5
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_scale: float = 1.003
        step_scale: float = 1.5
        coordinate_augmentation: bool = True
        synchronize_sigmas: bool = False

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the atom diffusion module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.sigma_data: float = cfg.sigma_data
        self.rho: int = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.num_sampling_steps: int = cfg.num_sampling_steps
        self.gamma_0: float = cfg.gamma_0
        self.gamma_min: float = cfg.gamma_min
        self.noise_scale: float = cfg.noise_scale
        self.step_scale: float = cfg.step_scale
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation
        self.synchronize_sigmas: bool = cfg.synchronize_sigmas

        if self.coordinate_augmentation:
            self.random_augmentation = CenterRandomAugmentation(
                centering=True,
                random_rotate=self.coordinate_augmentation,
            )

    # === EDM diffusion coefficients === #
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_data).clamp(1e-20).log() * 0.25

    def forward_model(
        self,
        x_noisy: torch.Tensor,
        times: torch.Tensor | float,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache=None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (N_diffusion_samples, L, 3).
        times : torch.Tensor | float
            Diffusion times. Shape (N_diffusion_samples,).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (L, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (L, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (L, L, c_z).
        model_cache : optional
            Model cache for efficiency.

        Returns
        -------
        denoised_coords : torch.Tensor
            Denoised atom coordinates. Shape (N_diffusion_samples, L, 3).
        """
        if not isinstance(times, torch.Tensor):
            times = torch.full(
                (x_noisy.shape[0],), times, device=x_noisy.device, dtype=x_noisy.dtype
            )

        x_update = self.score_model(
            x_noisy=x_noisy,
            times=times,
            f_input=f_input,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            model_cache=model_cache,
        )
        return x_update

    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_sampling_steps: int | None = None,
        num_diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
    ) -> torch.Tensor:
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.
        """

        if num_sampling_steps is None:
            num_sampling_steps = self.num_sampling_steps

        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples

        # Get noise schedule
        sigmas = self.get_sampling_schedule(
            num_sampling_steps=num_sampling_steps, device=s_inputs.device
        )
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas, gammas = sigmas.tolist(), gammas.tolist()

        atom_mask = f_input.atom.pad_mask.unsqueeze(0)  # (1, Natom)

        # Model cache for efficiency
        model_cache: dict = {}

        # Line 1
        init_sigma = sigmas[0]
        prior_coords = self.sample_prior(f_input, num_diffusion_samples)  # (B, Natom, 3)
        atom_coords = init_sigma * prior_coords  # (B, Natom, 3)

        # Line 2: gradually denoise
        for step_idx in range(1, num_sampling_steps):
            # Line 3
            atom_coords = self.random_augmentation(atom_coords, atom_mask=atom_mask)

            # Line 4
            sigma_tm, sigma_t, gamma = (
                sigmas[step_idx - 1],
                sigmas[step_idx],
                gammas[step_idx],
            )

            # Line 5
            t_hat: float = sigma_tm * (1 + gamma)

            # Line 6
            noise_var: float = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * torch.randn_like(atom_coords)

            # Line 7
            atom_coords_noisy = atom_coords + eps

            # Line 8
            # Process in chunks for memory efficiency
            atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
            for st in range(0, num_diffusion_samples, max_parallel_samples):
                end = min(st + max_parallel_samples, num_diffusion_samples)
                atom_coords_denoised[st:end] = self.forward_model(
                    x_noisy=atom_coords_noisy[st:end],
                    times=t_hat,
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
            atom_coords = atom_coords_noisy + self.step_scale * dt * delta_coords

        return atom_coords

    def sample_sigma(
        self,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels (sigma) during model training."""

        def _sample(n: int):
            return (
                self.sigma_data
                * (self.P_mean + self.P_std * torch.randn((n,), device=device)).exp()
            )

        if self.synchronize_sigmas:
            # synchronize sigmas across diffusion samples
            return _sample(1).expand(num_diffusion_samples)
        else:
            # use different sigmas for each diffusion sample
            return _sample(num_diffusion_samples)

    def get_sampling_schedule(
        self,
        num_sampling_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the noise schedule for diffusion sampling."""

        if num_sampling_steps is None:
            num_sampling_steps = self.num_sampling_steps

        inv_rho = 1 / self.rho

        steps = torch.arange(num_sampling_steps, dtype=torch.float32, device=device)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
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
        """Sample from the prior distribution."""
        Nsample = num_diffusion_samples
        Natom = len(f_input.atom)
        return torch.randn((Nsample, Natom, 3), device=f_input.device)

    def sample_holo(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the prior distribution."""
        holo_coords = super().sample_holo(f_input, num_diffusion_samples)
        atom_mask = f_input.atom.pad_mask.unsqueeze(0)  # (1, Natom)

        # Apply coordinate augmentation
        holo_coords = self.random_augmentation(holo_coords, atom_mask)

        # Mask out the padding atoms
        holo_coords = torch.masked_fill(holo_coords, ~atom_mask, 0.0)
        return holo_coords

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        sigma: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        noise_coords : torch.Tensor
            The noisy coordinates. Shape (B, N, 3).
        label_coords : torch.Tensor
            The label coordinates. Shape (B, N, 3).
        sigma : torch.Tensor
            The sigma values. Shape (B,).
        f_input : FoldingInput
            The FoldingInput object.
        """
        atom_mask = f_input.atom.pad_mask.unsqueeze(0)  # (1, Natom)

        noised_atom_coords = (
            label_coords + sigma[:, None, None] * noise_coords
        )  # (B, Natom, 3)

        # Mask out the padding atoms
        noised_atom_coords = torch.masked_fill(noised_atom_coords, ~atom_mask, 0.0)
        return noised_atom_coords
