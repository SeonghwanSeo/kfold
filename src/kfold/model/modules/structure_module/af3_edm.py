import math
from typing import TypeVar

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.af3_diffusion import AF3DiffusionModule
from kfold.utils.geometry.random_augment import CenterRandomAugmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseEDM

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


@STRUCTURE_MODULE.register()
class AF3SampleDiffusion(BaseEDM):
    """Atom diffusion module used in AlphaFold3.
    See Section 3.7 Algorithm 18: SampleDiffusion in the AF3 paper.
    """

    class Config(BaseConfig):
        """Configuration for the Structure module.

        Parameters
        ----------
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
        """

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

    def __init__(self, cfg: Config, score_model: AF3DiffusionModule):
        """Initialize the atom diffusion module."""
        super().__init__(cfg, score_model)
        self.score_model: AF3DiffusionModule = score_model
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.sigma_data: float = cfg.sigma_data
        self.rho: int = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.gamma_0: float = cfg.gamma_0
        self.gamma_min: float = cfg.gamma_min
        self.noise_scale: float = cfg.noise_scale
        self.step_scale: float = cfg.step_scale
        self.random_augmentation = CenterRandomAugmentation()

    # === EDM diffusion coefficients === #
    def c_skip(self, sigma: _ScalarOrTensor) -> _ScalarOrTensor:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return sigma * self.sigma_data / _sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return 1 / _sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: _ScalarOrTensor) -> _ScalarOrTensor:
        _log = lambda x: math.log(x) if isinstance(x, float) else torch.log(x)  # noqa
        _clip = lambda x, v: max(x, v) if isinstance(x, float) else x.clamp(v)  # noqa
        return _log(_clip(sigma / self.sigma_data, 1e-20)) * 0.25

    def apply_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to coordinates."""
        return self.random_augmentation(coords, mask=mask)

    # ============================================================
    # For model training
    # ============================================================
    def forward_train(
        self,
        x_t: torch.Tensor,
        t_hat: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            Diffusion noise level (or sigmas of EDM). Shape (B, N).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, Lt, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, Lt, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, Lt, Lt, c_z).

        Returns
        -------
        x_0_hat : torch.Tensor
            Denoised atom coordinates. Shape (B, N, La, 3).
        """
        c_in = self.c_in(t_hat)  # [B, N]
        c_noise = self.c_noise(t_hat)  # [B, N]
        c_skip = self.c_skip(t_hat)  # [B, N]
        c_out = self.c_out(t_hat)  # [B, N]

        # Line 2 of Algorithm 20
        r_noisy = c_in[..., None, None] * x_t  # [B, N, La, 3]

        r_update = self.score_model.train_step(
            f_input=f_input,
            r_noisy=r_noisy,  # [B, N, La, 3]
            c_noise=c_noise,  # [B, N]
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
        )

        # Line 8 of Algorithm 20
        x_0_hat = c_skip[..., None, None] * x_t + c_out[..., None, None] * r_update
        return x_0_hat

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat.
        See Section 3.7.1 Equation 6 of AlphaFold3 paper.

        NOTE: We replace `+` with `*` in the denominator compared to the AlphaFold3 paper.
        This matches the implementation in Boltz1, Protenix, and Openfold-3, and provides
        better training stability.
        """
        return (t_hat**2 + self.sigma_data**2) / ((t_hat * self.sigma_data) ** 2)

    def sample_noise_level(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.
        """
        # See Section 3.7 of AlphaFold3 paper.
        # t_hat = sigma_data * exp(-1.2 + 1.5 * N(0, 1)),
        # where -1.2 is P_mean and 1.5 is P_std.
        normal = torch.randn(shape, dtype=torch.float32, device=device)
        return self.sigma_data * torch.exp(self.P_mean + self.P_std * normal)

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        EDM equation:
        sigma = t_hat
        x_t = x_0 + sigma * noise
        where noise ~ N(0, I)

        Parameters
        ----------
        x_0 : torch.Tensor
            The label coordinates. Shape (B, N, La, 3).
        x_T : torch.Tensor
            The noise. Shape (B, N, La, 3).
        sigma : torch.Tensor
            The sigma values. Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).
        """
        noise = x_T
        sigma = t_hat
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x_t = x_0 + sigma[:, :, None, None] * noise
        x_t.masked_fill_(~mask[:, None, :, None], 0.0)  # apply atom mask
        return x_t

    # ============================================================
    # For inference
    # ============================================================
    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_steps: int = 200,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.
        """
        # Get noise schedule
        sigmas: list[float] = self.get_sampling_schedule(num_steps)
        gammas: list[float] = [
            self.gamma_0 if s > self.gamma_min else 0.0 for s in sigmas
        ]

        # Line 1
        sigma_0 = sigmas[0]
        x: torch.Tensor = sigma_0 * self.sample_prior(f_input, num_samples)
        mask: torch.Tensor = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)
        x.masked_fill_(~mask[:, :, :, None], 0.0)  # apply atom mask

        # Compute time-independent variables
        z = self.get_pair_conditioning(f_input, z_trunk)
        q, c, p = self.get_atom_embeddings(f_input, s_inputs, s_trunk, z)
        pair_bias = self.get_pair_bias(z)
        del z_trunk, z  # Free up memory for large LxL tensors

        def run_step(x_t: torch.Tensor, t_hat: float) -> torch.Tensor:
            s = self.get_single_conditioning(s_inputs, s_trunk, t_hat)
            return self.inference_step(
                f_input=f_input,
                x_t=x_t,
                t_hat=t_hat,
                q=q,
                c=c,
                p=p,
                s=s,
                pair_bias=pair_bias,
                chunk_size=chunk_size,
            )

        # Line 2: gradually denoise
        traj: list[torch.Tensor] = []
        for step_idx in range(1, num_steps + 1):
            if return_traj:
                traj.append(x.cpu())  # Move to cpu to save memory

            # Line 3
            x = self.random_augmentation(x, mask=mask)

            # Line 4
            sigma_tm = sigmas[step_idx - 1]
            sigma_t = sigmas[step_idx]
            gamma = gammas[step_idx]

            # Line 5
            t_hat: float = sigma_tm * (1 + gamma)

            # Line 6
            noise_var: float = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * torch.randn_like(x)
            eps.masked_fill_(~mask[:, :, :, None], 0.0)  # apply atom mask

            # Line 7
            x_noisy = x + eps

            # Line 8
            x_denoised = run_step(x_noisy, t_hat)

            # Line 9
            delta = (x_noisy - x_denoised) / t_hat

            # line 10
            dt = sigma_t - t_hat

            # Line 11
            x = x_noisy + self.step_scale * dt * delta

        sample_out: dict[str, torch.Tensor] = {
            "sample_coordinates": x,
        }
        if return_traj:
            traj.append(x.cpu())  # Move to cpu to save memory
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Latom, 3)

        return sample_out

    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        """Get the noise schedule for diffusion sampling as a Python list."""
        inv_rho = 1 / self.rho
        sigma_max_pow = self.sigma_max**inv_rho
        sigma_min_pow = self.sigma_min**inv_rho

        sigmas: list[float] = []
        for i in range(num_steps):
            # Linearly interpolate in the noise space (rho-domain)
            step_ratio = i / max((num_steps - 1), 1)
            interpolated = sigma_max_pow + step_ratio * (sigma_min_pow - sigma_max_pow)
            # Scale and transform back
            sigma = (interpolated**self.rho) * self.sigma_data
            sigmas.append(float(sigma))

        # Last step is sigma value of 0.
        sigmas.append(0.0)
        return sigmas

    def inference_step(
        self,
        f_input: FoldingInput,
        x_t: torch.Tensor,
        t_hat: float,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        s: torch.Tensor,
        pair_bias: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        x_t : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 3).
        t_hat : float
            Diffusion noise level (or sigmas of EDM).
        q : torch.Tensor
            The atom single representation, shape [B, La, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, La, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, La, La, c_atompair].
        s : torch.Tensor
            Single conditioning. Shape (B, 1, L, c_s), broadcast to (B, N, L, c_s).
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 3).
        """
        # Line 2 of Algorithm 20
        r_noisy = self.c_in(t_hat) * x_t

        def _step(_r: torch.Tensor) -> torch.Tensor:
            return self.score_model.step(
                _r,  # [B, N, L, 3]
                q,  # [B, La, c_atom]
                c,  # [B, La, c_atom]
                p,  # [B, La, La, c_atompair]
                f_input.atom.token_index,  # [B, La]
                f_input.atom.pad_mask,  # [B, La]
                s,  # [B, 1, L, c_s], broadcast to [B, N, L, c_s]
                pair_bias,  # [B, Nblock, H, Lt, Lt]
                f_input.token.pad_mask,  # [B, L]
            )

        if chunk_size is None:
            r_update = _step(r_noisy)
        else:
            r_update = torch.zeros_like(r_noisy)
            for st in range(0, r_noisy.shape[1], chunk_size):
                end = min(st + chunk_size, r_noisy.shape[1])
                r_update[:, st:end] = _step(r_noisy[:, st:end])

        # Line 8 of Algorithm 20
        x_out = self.c_skip(t_hat) * x_t + self.c_out(t_hat) * r_update  # [B, N, La, 3]
        return x_out

    def get_pair_conditioning(
        self, f_input: FoldingInput, z_trunk: torch.Tensor
    ) -> torch.Tensor:
        """Get the pair conditioning for the score model.
        See Section 3.7: Algorithm 21 of AlphaFold3 paper.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, Lt, c_z].

        Returns
        -------
        z : torch.Tensor
            The pair conditioning, shape [B, Lt, Lt, c_z].
        """
        return self.score_model.get_pair_conditioning(f_input, z_trunk)

    def get_atom_embeddings(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the inputs which are static across diffusion steps.
        # Algorithm 5 Line 1-10, 13-14.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        z : torch.Tensor
            The trunk pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        q : torch.Tensor
            The atom single representation, shape [B, La, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, La, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, La, La, c_atompair].
        """
        return self.score_model.get_atom_embeddings(f_input, s_inputs, s_trunk, z)

    def get_pair_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Get the pair bias for the token transformer.
        This is time-independent and can be pre-computed before the diffusion steps.

        Parameters
        ----------
        z : torch.Tensor
            The pair conditioning, shape [B, Lt, Lt, c_z].

        Returns
        -------
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, Lt, Lt].
        """
        return self.score_model.get_pair_bias(z)

    def get_single_conditioning(
        self, s_inputs: torch.Tensor, s_trunk: torch.Tensor, t_hat: float
    ) -> torch.Tensor:
        """Get the single conditioning for the score model.
        See Section 3.7: Algorithm 21 of AlphaFold3 paper.

        Parameters
        ----------
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        t_hat : float
            Diffusion noise level (or sigma).

        Returns
        -------
        s : torch.Tensor
            The single conditioning, shape [B, 1, Lt, c_s].
        """
        c_noise = self.c_noise(t_hat)
        c_noise = torch.tensor(c_noise, device=s_inputs.device)
        c_noise = c_noise.view(1, 1)
        return self.score_model.get_single_conditioning(s_inputs, s_trunk, c_noise)
