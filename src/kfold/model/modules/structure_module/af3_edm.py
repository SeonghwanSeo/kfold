import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import CenterRandomAugmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseEDM


@STRUCTURE_MODULE.register()
class AF3SampleDiffusion(BaseEDM):
    """Atom diffusion module used in AlphaFold3.
    See Section 3.7 Algorithm 18: SampleDiffusion in the AF3 paper.
    """

    class Config(BaseConfig):
        """Configuration for the Structure module.

        Parameters
        ----------
        num_steps : int, optional
            The number of sampling steps, by default 200.
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
        """

        num_steps: int = 200
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

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        """Initialize the atom diffusion module."""
        super().__init__(cfg, score_model)
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.sigma_data: float = cfg.sigma_data
        self.rho: int = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.num_steps: int = cfg.num_steps
        self.gamma_0: float = cfg.gamma_0
        self.gamma_min: float = cfg.gamma_min
        self.noise_scale: float = cfg.noise_scale
        self.step_scale: float = cfg.step_scale
        self.coordinate_augmentation: bool = cfg.coordinate_augmentation

        self.random_augmentation = CenterRandomAugmentation(
            centering=True,
            augmentation=self.coordinate_augmentation,
            s_trans=1.0,  # not used when augmentation is False
        )

    def apply_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to coordinates."""
        return self.random_augmentation(coords, mask=mask)

    # === EDM diffusion coefficients === #
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma / self.sigma_data).clamp(1e-20).log() * 0.25

    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution."""
        B = f_input.batch_size
        N = num_diffusion_samples
        La = f_input.num_atoms
        return torch.randn((B, N, La, 3), device=f_input.device)

    # === For model training === #
    def forward_model(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor | float,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        prior_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t_hat : torch.Tensor | float
            Diffusion noise level (or sigmas of EDM). Shape (B, N).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, L, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, L, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, L, L, c_z).

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 3).
        """
        if not isinstance(t_hat, torch.Tensor):
            t_hat = torch.full(
                x_noisy.shape[:2], t_hat, device=x_noisy.device, dtype=x_noisy.dtype
            )  # [B, N]
        t_hat_reshaped = t_hat[..., None, None]  # [B, N, 1, 1]

        # Line 2 of Algorithm 20
        r_noisy = self.c_in(t_hat_reshaped) * x_noisy

        # Line 8 of Algorithm 21
        c_noise = self.c_noise(t_hat)  # [B, N]

        r_update = self.score_model(
            r_noisy=r_noisy,  # [B, N, La, 3]
            c_noise=c_noise,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
        )

        # Line 8 of Algorithm 20
        x_out = (
            self.c_skip(t_hat_reshaped) * x_noisy
            + self.c_out(t_hat_reshaped) * r_update  # [B, N, La, 3]
        )
        return x_out

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat.
        See Section 3.7.1 Equation 6 of AlphaFold3 paper.

        NOTE: We replace `+` with `*` in the denominator compared to the AlphaFold3 paper.
        This matches the implementation in Boltz1, Protenix, and Openfold-3, and provides
        better training stability.
        """
        return (t_hat**2 + self.sigma_data**2) / ((t_hat * self.sigma_data) ** 2)

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
        shape = (batch_size, num_diffusion_samples)
        return self.sigma_data * torch.exp(
            self.P_mean + self.P_std * torch.randn(shape, device=device)
        )

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        EDM equation:
        sigma = t_hat
        x_noised = x_label + sigma * noise
        where noise ~ N(0, I)

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        noise_coords : torch.Tensor
            The noisy coordinates. Shape (B, N, La, 3).
        label_coords : torch.Tensor
            The label coordinates. Shape (B, N, La, 3).
        sigma : torch.Tensor
            The sigma values. Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).
        """
        noised_atom_coords = (
            label_coords + t_hat[:, :, None, None] * noise_coords
        )  # (B, N, Latom, 3)
        return noised_atom_coords

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

        # Get noise schedule
        sigmas = self.get_sampling_schedule(num_steps=num_steps, device=s_inputs.device)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
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
            noise_var: float = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
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
                )

            # Line 9
            delta_coords = (atom_coords_noisy - atom_coords_denoised) / t_hat

            # line 10
            dt = sigma_t - t_hat

            # Line 11
            atom_coords = atom_coords_noisy + self.step_scale * dt * delta_coords

            if return_traj:
                traj.append(atom_coords.cpu())  # Move to cpu to save memory

        sample_out["init_coordinates"] = start_coords
        sample_out["sample_coordinates"] = atom_coords
        if return_traj:
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Latom, 3)

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
