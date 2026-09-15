"""Elucidated diffusion model (EDM) structure generation."""

import dataclasses
import math
from typing import TypeVar

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives.utils import expand_dim
from kfold.utils.config import configurable
from kfold.utils.geometry.random_augment import CenterRandomAugmentation

from .ecsi import BaseStructureModule
from .score_model import DiffusionModule

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


@configurable
class AF3SampleDiffusion(BaseStructureModule):
    """AlphaFold 3 atom diffusion using EDM preconditioning and sampling."""

    @dataclasses.dataclass(kw_only=True)
    class Config(BaseStructureModule.Config):
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
        z: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Sample EDM training inputs and predict the clean coordinates."""
        with torch.autocast(f_input.device.type, enabled=False):
            train_input = self.sample_train_input(f_input, diffusion_batch_size)

        t_hat = train_input["t_hat"]
        x_0 = train_input["x_0"]
        x_t = train_input["x_t"]
        x_0_hat = self._forward_train(
            x_t=x_t,
            t=t_hat,
            f_input=f_input,
            s_inputs=s_inputs,
            z=z,
        )

        return {
            "t_hat": t_hat,
            "x_t": x_t,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": self.loss_weights(t_hat),
        }

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        c_in, c_noise, c_skip, c_out = (
            self.c_in(t),
            self.c_noise(t),
            self.c_skip(t),
            self.c_out(t),
        )
        r_noisy = c_in[..., None, None] * x_t
        r_update = self.score_model.train_step(
            f_input=f_input,
            r_noisy=r_noisy,
            c_noise=c_noise,
            s_inputs=s_inputs,
            z=z,
        )
        return c_skip[..., None, None] * x_t + c_out[..., None, None] * r_update

    def sample_noise_level(self, shape: tuple, device: torch.device) -> torch.Tensor:
        normal = torch.randn(shape, dtype=torch.float32, device=device)
        return self.sigma_data * torch.exp(self.P_mean + self.P_std * normal)

    def sample_train_input(
        self,
        f_input: FoldingInput,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        num_samples = diffusion_batch_size
        t_hat = self.sample_noise_level((f_input.batch_size, num_samples), f_input.device)

        x_label = f_input.atom.label_coords
        x_label_mask = f_input.atom.resolved_mask
        x_0 = expand_dim(x_label, num_samples, dim=-3)
        x_0 = self.random_augmentation(x_0, mask=x_label_mask.unsqueeze(-2))

        noise = torch.randn_like(x_0)
        x_t = x_0 + t_hat[:, :, None, None] * noise
        x_t_mask = f_input.atom.pad_mask.unsqueeze(1)
        x_t.masked_fill_(~x_t_mask[..., None], 0.0)
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
        z: torch.Tensor,
        num_steps: int = 200,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        model = self.score_model
        sigmas = self.get_sampling_schedule(num_steps)
        gammas = [self.gamma_0 if sigma > self.gamma_min else 0.0 for sigma in sigmas]

        sigma_0 = sigmas[0]
        x = sigma_0 * self.sample_prior(f_input, num_samples)
        mask = f_input.atom.pad_mask.unsqueeze(1)
        x.masked_fill_(~mask[..., None], 0.0)

        z = model.get_pair_conditioning(f_input, z)
        q, c, p = model.get_atom_embeddings(f_input, z)
        pair_bias = model.get_pair_bias(z)

        def run_step(x_t: torch.Tensor, t_hat: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t_hat), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, c_noise.view(1, 1))
            return self.inference_step(
                f_input, x_t, t_hat, q, c, p, s, pair_bias, chunk_size
            )

        traj: list[torch.Tensor] = []
        for step_idx in range(1, num_steps + 1):
            if return_traj:
                traj.append(x.cpu())

            x = self.random_augmentation(x, mask=mask)
            sigma_tm = sigmas[step_idx - 1]
            sigma_t = sigmas[step_idx]
            gamma = gammas[step_idx]
            t_hat = sigma_tm * (1 + gamma)

            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * torch.randn_like(x)
            eps.masked_fill_(~mask[..., None], 0.0)
            x_noisy = x + eps
            x_denoised = run_step(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat
            x = x_noisy + self.step_scale * (sigma_t - t_hat) * delta

        sample_out = {"coordinates": x}
        if return_traj:
            traj.append(x.cpu())
            sample_out["traj"] = torch.stack(traj, dim=-3)
        return sample_out

    def sample_prior(self, f_input: FoldingInput, num_samples: int) -> torch.Tensor:
        x = torch.randn(
            (f_input.batch_size, num_samples, f_input.num_atoms, 3),
            device=f_input.device,
            dtype=torch.float32,
        )
        x.masked_fill_(~f_input.atom.pad_mask[:, None, :, None], 0.0)
        return x

    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        inv_rho = 1 / self.rho
        sigma_max_pow = self.sigma_max**inv_rho
        sigma_min_pow = self.sigma_min**inv_rho

        sigmas = []
        for i in range(num_steps):
            step_ratio = i / max((num_steps - 1), 1)
            interpolated = sigma_max_pow + step_ratio * (sigma_min_pow - sigma_max_pow)
            sigmas.append(float((interpolated**self.rho) * self.sigma_data))
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
        r_noisy = self.c_in(t_hat) * x_t

        def _step(_r: torch.Tensor) -> torch.Tensor:
            return self.score_model.step(
                _r,
                q,
                c,
                p,
                f_input.atom.token_index,
                f_input.atom.pad_mask,
                s,
                pair_bias,
                f_input.token.pad_mask,
            )

        if chunk_size is None:
            r_update = _step(r_noisy)
        else:
            r_update = torch.zeros_like(r_noisy)
            for start in range(0, r_noisy.shape[1], chunk_size):
                end = min(start + chunk_size, r_noisy.shape[1])
                r_update[:, start:end] = _step(r_noisy[:, start:end])

        return self.c_skip(t_hat) * x_t + self.c_out(t_hat) * r_update
