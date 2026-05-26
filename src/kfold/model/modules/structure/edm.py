import math
from typing import TypeVar

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives.utils import expand_dim
from kfold.utils.geometry.random_augment import CenterRandomAugmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .base import BaseStructureModule
from .score_model import DiffusionModule

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


@STRUCTURE_MODULE.register()
class AF3SampleDiffusion(BaseStructureModule):
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
        rho : float, optional
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
        rho: float = 7
        P_mean: float = -1.2
        P_std: float = 1.5
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_scale: float = 1.003
        step_scale: float = 1.5

    def __init__(self, cfg: Config, score_model: DiffusionModule):
        """Initialize the atom diffusion module."""
        super().__init__(cfg, score_model)
        self.score_model: DiffusionModule = score_model
        self.sigma_min: float = cfg.sigma_min
        self.sigma_max: float = cfg.sigma_max
        self.sigma_data: float = cfg.sigma_data
        self.rho: float = cfg.rho
        self.P_mean: float = cfg.P_mean
        self.P_std: float = cfg.P_std
        self.gamma_0: float = cfg.gamma_0
        self.gamma_min: float = cfg.gamma_min
        self.noise_scale: float = cfg.noise_scale
        self.step_scale: float = cfg.step_scale
        self.random_augmentation = CenterRandomAugmentation()

    # === EDM diffusion coefficients === #
    def c_skip(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        return (self.sigma_data**2) / (t_hat**2 + self.sigma_data**2)  # type: ignore

    def c_out(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return t_hat * self.sigma_data / _sqrt(self.sigma_data**2 + t_hat**2)

    def c_in(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return 1 / _sqrt(t_hat**2 + self.sigma_data**2)

    def c_noise(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _log = lambda x: math.log(x) if isinstance(x, float) else torch.log(x)  # noqa
        _clip = lambda x, v: max(x, v) if isinstance(x, float) else x.clamp(v)  # noqa
        return _log(_clip(t_hat / self.sigma_data, 1e-20)) * 0.25

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        return 1 / self.c_out(t_hat) ** 2

    # ============================================================
    # For model training
    # ============================================================
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        with torch.autocast(f_input.device.type, enabled=False):
            train_input = self.sample_train_input(f_input, diffusion_batch_size)

        t_hat = train_input["t_hat"]  # [B, N]
        x_0 = train_input["x_0"]  # [B, N, La, 3]
        x_t = train_input["x_t"]  # [B, N, La, 3]

        x_0_hat = self._forward_train(
            x_t=x_t,  # [B, N, La, 3]
            t=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
        )  # [B, N, La, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "x_t": x_t,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
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
        t : torch.Tensor
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
        c_in, c_noise, c_skip, c_out = (
            self.c_in(t),
            self.c_noise(t),
            self.c_skip(t),
            self.c_out(t),
        )

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

    def sample_noise_level(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.
        """
        # See Section 3.7 of AlphaFold3 paper.
        normal = torch.randn(shape, dtype=torch.float32, device=device)
        return self.sigma_data * torch.exp(self.P_mean + self.P_std * normal)

    def sample_train_input(
        self,
        f_input: FoldingInput,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample training inputs for the structure module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        diffusion_batch_size : int
            The number of samples to generate for training.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the x_0, x_t, and related representations.
        """
        num_samples = diffusion_batch_size
        t_hat = self.sample_noise_level((f_input.batch_size, num_samples), f_input.device)

        # === Sample x_0 from label === #
        x_label = f_input.atom.label_coords  # [B, Natom, 3]
        x_label_mask = f_input.atom.resolved_mask  # [B, Natom]
        # repeat holo coords
        x_0 = expand_dim(x_label, num_samples, dim=-3)  # [B, N, Natom, 3]
        x_0_mask = x_label_mask.unsqueeze(-2)  # [B, 1, Natom]
        # Apply centering/coordinate augmentation
        x_0 = self.random_augmentation(x_0, mask=x_0_mask)

        # === Sample x_t by adding noise to x_0 === #
        noise = torch.randn_like(x_0)
        x_t = x_0 + t_hat[:, :, None, None] * noise
        x_t_mask = f_input.atom.pad_mask.unsqueeze(1)  # (B, 1, Latom)
        x_t.masked_fill_(~x_t_mask[..., None], 0.0)  # apply atom mask
        return {
            "t_hat": t_hat,
            "x_0": x_0,
            "x_t": x_t,
        }

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
        model = self.score_model

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
        z = model.get_pair_conditioning(f_input, z_trunk)
        q, c, p = model.get_atom_embeddings(f_input, s_trunk, z)
        pair_bias = model.get_pair_bias(z)
        del z_trunk, z  # Free up memory for large LxL tensors

        def run_step(x_t: torch.Tensor, t_hat: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t_hat), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, s_trunk, c_noise.view(1, 1))
            return self.inference_step(
                f_input, x_t, t_hat, q, c, p, s, pair_bias, chunk_size
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
            "coordinates": x,
        }
        if return_traj:
            traj.append(x.cpu())  # Move to cpu to save memory
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Latom, 3)

        return sample_out

    def sample_prior(self, f_input: FoldingInput, num_samples: int) -> torch.Tensor:
        """Sample from the prior distribution."""
        B = f_input.batch_size
        N = num_samples
        La = f_input.num_atoms
        mask = f_input.atom.pad_mask
        x = torch.randn((B, N, La, 3), device=f_input.device, dtype=torch.float32)
        x.masked_fill_(~mask[:, None, :, None], 0.0)
        return x

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
        x_out = self.c_skip(t_hat) * x_t + self.c_out(t_hat) * r_update
        return x_out
