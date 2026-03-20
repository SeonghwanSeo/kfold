import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.base import BaseScoreModel
from kfold.utils.geometry.random_augment import get_center
from kfold.utils.registry import STRUCTURE_MODULE

from .kfold_ecsi import KFoldECSI


@STRUCTURE_MODULE.register()
class KFoldExpandedECSI(KFoldECSI):
    """ECSI with separate COM and internal-coordinate bridge dynamics."""

    class Config(KFoldECSI.Config):
        gamma_scale_com: float = 1.0
        gamma_scale_internal: float = 1.0
        eta_com: float | None = None
        eta_internal: float | None = None

    def __init__(self, cfg: Config, score_model: BaseScoreModel):
        super().__init__(cfg, score_model)
        self.gamma_scale_com = float(cfg.gamma_scale_com)
        self.gamma_scale_internal = float(cfg.gamma_scale_internal)
        self.eta_com = self.eta if cfg.eta_com is None else float(cfg.eta_com)
        self.eta_internal = (
            self.eta if cfg.eta_internal is None else float(cfg.eta_internal)
        )

    @staticmethod
    def _broadcast_mask(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_out = mask.bool()
        while mask_out.ndim < coords.ndim - 1:
            mask_out = mask_out.unsqueeze(-2)
        return mask_out.expand(coords.shape[:-1])

    def compute_com(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_broadcast = self._broadcast_mask(coords, mask)
        return get_center(coords, mask_broadcast)

    def decompose_coords(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_broadcast = self._broadcast_mask(coords, mask)
        com = self.compute_com(coords, mask_broadcast)
        internal = (coords - com) * mask_broadcast.unsqueeze(-1)
        return com, internal

    def recompose_coords(
        self, com: torch.Tensor, internal: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_broadcast = self._broadcast_mask(internal, mask)
        return (com + internal) * mask_broadcast.unsqueeze(-1)

    def _component_scales(
        self, t_exp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma_base = self.gamma(t_exp)
        gamma_dot_base = self.gamma_deriv(t_exp)
        gamma_com = self.gamma_scale_com * gamma_base
        gamma_internal = self.gamma_scale_internal * gamma_base
        gamma_dot_com = self.gamma_scale_com * gamma_dot_base
        gamma_dot_internal = self.gamma_scale_internal * gamma_dot_base
        return gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal

    @staticmethod
    def _compute_eps(
        gamma_t: torch.Tensor,
        gamma_dot: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_dot: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        return eta * (
            gamma_t * gamma_dot - (alpha_dot / (alpha_t + 1e-8)) * gamma_t**2
        )

    def _compute_drift_components(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        x_T: torch.Tensor,
        t_exp: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha(t_exp)
        beta_t = self.beta(t_exp)
        alpha_dot = self.alpha_deriv(t_exp)
        beta_dot = self.beta_deriv(t_exp)
        gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal = (
            self._component_scales(t_exp)
        )

        x_t_com, x_t_internal = self.decompose_coords(x_t, mask)
        x0_com, x0_internal = self.decompose_coords(x0_hat, mask)
        xT_com, xT_internal = self.decompose_coords(x_T, mask)

        z_hat_com = (x_t_com - alpha_t * x0_com - beta_t * xT_com) / (gamma_com + 1e-8)
        z_hat_internal = (
            x_t_internal - alpha_t * x0_internal - beta_t * xT_internal
        ) / (gamma_internal + 1e-8)

        eps_com = self._compute_eps(
            gamma_t=gamma_com,
            gamma_dot=gamma_dot_com,
            alpha_t=alpha_t,
            alpha_dot=alpha_dot,
            eta=self.eta_com,
        )
        eps_internal = self._compute_eps(
            gamma_t=gamma_internal,
            gamma_dot=gamma_dot_internal,
            alpha_t=alpha_t,
            alpha_dot=alpha_dot,
            eta=self.eta_internal,
        )

        drift_com = (
            alpha_dot * x0_com
            + beta_dot * xT_com
            + (gamma_dot_com + eps_com / (gamma_com + 1e-8)) * z_hat_com
        )
        drift_internal = (
            alpha_dot * x0_internal
            + beta_dot * xT_internal
            + (gamma_dot_internal + eps_internal / (gamma_internal + 1e-8))
            * z_hat_internal
        )
        return drift_com, drift_internal, eps_com, eps_internal

    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
        f_input: FoldingInput | None = None,
    ) -> torch.Tensor:
        del f_input

        t_exp = t_hat[:, :, None, None]
        alpha_t = self.alpha(t_exp)
        beta_t = self.beta(t_exp)
        gamma_com, gamma_internal, _, _ = self._component_scales(t_exp)

        apo_com, apo_internal = self.decompose_coords(noise_coords, mask)
        holo_com, holo_internal = self.decompose_coords(label_coords, mask)

        mean_com = alpha_t * holo_com + beta_t * apo_com
        mean_internal = alpha_t * holo_internal + beta_t * apo_internal

        full_noise = torch.randn_like(noise_coords)
        noise_com, noise_internal = self.decompose_coords(full_noise, mask)

        x_t_com = mean_com + gamma_com * noise_com
        x_t_internal = mean_internal + gamma_internal * noise_internal
        return self.recompose_coords(x_t_com, x_t_internal, mask)

    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_churn_target: torch.Tensor,
        atom_mask: torch.Tensor,
        t_curr: float,
        t_next: float,
        t_exp: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
        dt = t_next - t_curr
        delta_churn = float(self.churn_factor) * abs(dt)
        apply_churn = t_curr + delta_churn <= self.sigma_max
        if self.churn_until_time is not None:
            apply_churn = apply_churn and (t_curr > self.churn_until_time)

        if not apply_churn:
            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]),
                t_curr,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            return x_t, t_curr, t_curr_tensor, t_exp, dt

        alpha_t = self.alpha(t_exp)
        beta_t = self.beta(t_exp)
        alpha_dot = self.alpha_deriv(t_exp)
        beta_dot = self.beta_deriv(t_exp)
        gamma_com, gamma_internal, gamma_dot_com, gamma_dot_internal = (
            self._component_scales(t_exp)
        )

        f_t = alpha_dot / (alpha_t + 1e-8)
        s_t = beta_dot - f_t * beta_t
        base_eps_com = gamma_com * gamma_dot_com - f_t * gamma_com**2
        base_eps_internal = (
            gamma_internal * gamma_dot_internal - f_t * gamma_internal**2
        )
        g_com = torch.sqrt(torch.clamp(2.0 * base_eps_com, min=0.0) + 1e-8)
        g_internal = torch.sqrt(
            torch.clamp(2.0 * base_eps_internal, min=0.0) + 1e-8
        )

        x_t_com, x_t_internal = self.decompose_coords(x_t, atom_mask)
        x_target_com, x_target_internal = self.decompose_coords(x_churn_target, atom_mask)
        churn_noise = torch.randn_like(x_t)
        noise_com, noise_internal = self.decompose_coords(churn_noise, atom_mask)

        x_t_com = (
            x_t_com
            + (f_t * x_t_com + s_t * x_target_com) * delta_churn
            + g_com * (delta_churn**0.5) * noise_com
        )
        x_t_internal = (
            x_t_internal
            + (f_t * x_t_internal + s_t * x_target_internal) * delta_churn
            + g_internal * (delta_churn**0.5) * noise_internal
        )
        x_t = self.recompose_coords(x_t_com, x_t_internal, atom_mask)

        t_curr = t_curr + delta_churn
        t_curr_tensor = torch.full(
            (x_t.shape[0], x_t.shape[1]),
            t_curr,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        t_exp = t_curr_tensor[:, :, None, None]
        dt = t_next - t_curr
        return x_t, t_curr, t_curr_tensor, t_exp, dt

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
        sample_out: dict[str, torch.Tensor] = {}
        traj: list[torch.Tensor] = []

        if num_steps is None:
            num_steps = self.num_steps
        if max_parallel_samples is None:
            max_parallel_samples = num_diffusion_samples
        if self.perturb_xt and self.endpoint_perturb_scale is None:
            raise ValueError(
                "endpoint_perturb_scale must be provided when perturb_xt is enabled."
            )

        model_cache = {}
        times = self.get_sampling_schedule(
            num_steps=num_steps, device=s_inputs.device
        ).tolist()
        atom_mask = f_input.atom.pad_mask.unsqueeze(1)

        x_T = self.sample_prior(f_input, num_diffusion_samples)
        x_T = x_T * atom_mask[..., None]
        sample_out["init_coordinates"] = x_T

        if self.normalize_coordinate:
            x_T = x_T / self.sigma_data_end

        x_t = x_T.clone()
        if self.perturb_xt:
            perturb_scale = float(self.endpoint_perturb_scale or 0.0)
            perturb_scale_xt = (
                perturb_scale / self.sigma_data_end
                if self.normalize_coordinate
                else perturb_scale
            )
            x_t = self._apply_endpoint_perturbation(x_t, atom_mask, perturb_scale_xt)
            x_churn_target = x_t.clone()
        else:
            x_churn_target = x_T

        if return_traj:
            traj.append(x_t.cpu())

        for step_idx in range(num_steps):
            x_t, x_T, x_churn_target = self.random_augmentation(
                x_t, x_T, x_churn_target, mask=atom_mask
            )

            t_curr = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t_curr

            t_curr_tensor = torch.full(
                (x_t.shape[0], x_t.shape[1]),
                t_curr,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            t_exp = t_curr_tensor[:, :, None, None]

            if self.use_forward_pinned_churn and self.churn_factor > 0.0:
                x_t, t_curr, t_curr_tensor, t_exp, dt = self._apply_forward_pinned_churn(
                    x_t=x_t,
                    x_churn_target=x_churn_target,
                    atom_mask=atom_mask,
                    t_curr=t_curr,
                    t_next=t_next,
                    t_exp=t_exp,
                )

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
                    prior_coords=x_T[:, st:end],
                )
                if self.inference_align_x0_hat_to_x_t:
                    x0_hat[:, st:end] = self.align_apo_to_label(
                        apo_coords=x0_hat[:, st:end],
                        label_coords=x_t[:, st:end],
                        f_input=f_input,
                    )

            alpha_t = self.alpha(t_exp)
            beta_t = self.beta(t_exp)

            if float(self.ode_time_duration) > 0.0 and t_curr <= float(
                self.ode_time_duration
            ):
                t_next_exp = torch.full_like(t_exp, t_next)
                alpha_next = self.alpha(t_next_exp)
                beta_next = self.beta(t_next_exp)
                x_t = (
                    beta_next / (beta_t + 1e-8) * x_t
                    + (alpha_next - alpha_t * beta_next / (beta_t + 1e-8)) * x0_hat
                )
            else:
                drift_com, drift_internal, eps_com, eps_internal = (
                    self._compute_drift_components(
                        x_t=x_t,
                        x0_hat=x0_hat,
                        x_T=x_T,
                        t_exp=t_exp,
                        mask=atom_mask,
                    )
                )
                drift = drift_com + drift_internal

                diffusion_noise = torch.randn_like(x_t)
                noise_com, noise_internal = self.decompose_coords(
                    diffusion_noise, atom_mask
                )
                diffusion = (
                    torch.sqrt(2 * torch.abs(eps_com) * abs(dt) + 1e-8) * noise_com
                    + torch.sqrt(2 * torch.abs(eps_internal) * abs(dt) + 1e-8)
                    * noise_internal
                )
                x_t = x_t + drift * dt + diffusion

            x_t = x_t * atom_mask[..., None]
            if return_traj:
                traj.append(x_t.cpu())

        if self.normalize_coordinate:
            x_t = x_t * self.sigma_data

        sample_out["sample_coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj)
        return sample_out
