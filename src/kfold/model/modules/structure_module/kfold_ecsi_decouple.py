"""Implementation of Endpoint-Conditioned Stochastic Interpolant (ECSI) for biomolecular
structure prediction.

Reference: "Exploring the Design Space of Diffusion Bridge Models" (arXiv:2410.21553).

# Diffusion Path Design

Models the transition from source (apo) to target (holo) conformations.
- t=1 (Prior): Apo chain structures perturbed with 50 Å translational noise.
- t=0 (Data): Ground-truth assembled holo complexes.

# Decoupling COM and Intra-chain Dynamics

In standard diffusion formulations, the variance of the Center of Mass (COM) scales
proportionally to 1/sqrt(N_atom). This naturally suppresses global motion, leading to
overly deterministic COM trajectories and suboptimal sampling. To resolve this, our
ECSI bridge formulation decouples COM and intra-chain dynamics, enabling independent
noise scheduling for rigid-body translations versus local conformational changes.

# Decoupling Radial and Tangential COM Noise
To further enhance sampling diversity, we decompose the COM noise into radial and
tangential components. This allows separate control over the magnitude of noise that
pushes the structure directly towards/away from the target (radial) versus noise that
induces lateral "tentacle-like" exploration around the direct path (tangential).
"""

import math
from typing import TypeVar

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model.ecsi_diffusion import ECSIDiffusionModule
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.geometry.rigid_align import rigid_align
from kfold.utils.registry import STRUCTURE_MODULE

from . import kfold_ecsi as base_ecsi

RIGID_ALIGN = base_ecsi.RIGID_ALIGN  # conduct centering ; kabsch align
NO_ALIGN = base_ecsi.NO_ALIGN  # no centering; no kabsch align

_T = TypeVar("_T", float, torch.Tensor)


# === Utility functions with type flexibility and numerical stability handling === #
def _clip(t: _T, eps: float = 1e-10) -> _T:
    return t.clip(min=eps) if isinstance(t, torch.Tensor) else max(t, eps)  # type: ignore


def _sqrt(t: _T) -> _T:
    return _clip(t, eps=0) ** 0.5  # type: ignore


def _log(t: _T) -> _T:
    log = torch.log if isinstance(t, torch.Tensor) else math.log
    return log(_clip(t, eps=1e-10))  # type: ignore


# === Helper class/functions for ECSI bridge decomposition and noise handling === #
class ChainDecomposition:
    def __init__(self, f_input: FoldingInput):
        """Helper class for chain-aware decomposition and composition of coordinates.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs, used to extract chain_id.
        """
        num_chains = f_input.num_chains
        atom_mask = f_input.atom.pad_mask  # [B, Natom]
        token_mask = f_input.token.pad_mask  # [B, Ntoken]
        token_index = f_input.atom.token_index  # [B, Natom]

        # Renumber asym id to chain id: 1, 2, 5, 6, ... -> 1, 2, 3, ..., 0 0 0(pad)
        # NOTE: This is necessary for training because original asym id can be very large
        asym_id = f_input.token.asym_id
        assert (asym_id[~f_input.token.pad_mask] == -1).all(), (
            "Pad tokens must have asym_id = -1"
        )
        assert (asym_id[f_input.token.pad_mask] >= 1).all(), (
            "Non-pad tokens must have asym_id >= 1"
        )
        # Token to atom mapping
        chain_id = torch.zeros_like(asym_id)
        for b_i in range(f_input.batch_size):
            asym_id_i = asym_id[b_i]
            mask_i = token_mask[b_i]
            uniq_id = torch.sort(torch.unique(asym_id_i[mask_i]))[0]
            for c_i, a_i in enumerate(uniq_id, start=1):
                chain_id[b_i, asym_id_i == a_i] = c_i  # [B, Ntoken]

        b_i = torch.arange(f_input.batch_size, device=asym_id.device)[:, None]
        chain_id = chain_id[b_i, token_index]  # [B, Natom]
        chain_id[~atom_mask] = 0  # Set pad atoms to chain_id 0

        self.num_chains = num_chains + 1  # 0 is reserved for padding
        self.chain_id = chain_id  # [B, Natom]
        self.pad_mask = atom_mask  # [B, Natom]

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose coordinates into each chain components
        (B, *, L, 3) -> (B, *, Nchain, 3), (B, *, L, 3)
        """
        assert x.dtype == torch.float32
        B, *mid, L, _ = x.shape

        # Compute chain-wise COM
        view_shape = [B] + [1] * len(mid) + [L, 1]
        chain_id = self.chain_id.view(*view_shape).expand(x.shape)
        chain_com = torch.zeros(
            (B, *mid, self.num_chains, 3), device=x.device, dtype=x.dtype
        ).scatter_reduce_(-2, chain_id, x, reduce="mean", include_self=False)
        # Zero out COM for padding chain (chain_id=0)
        chain_com[..., 0, :] = 0.0

        x_com = chain_com.gather(-2, chain_id)  # [B, *, Natom, 3]
        x_intra = x - x_com

        # Zero out padding
        pad_mask = self.pad_mask.view(*view_shape).expand(x.shape)
        x_intra.masked_fill_(~pad_mask, 0.0)
        return chain_com, x_intra

    def recompose(self, x_com: torch.Tensor, x_intra: torch.Tensor) -> torch.Tensor:
        """Recompose coordinates from chain COM and intra components
        (B, *, Nchain, 3), (B, *, L, 3) -> (B, *, L, 3)
        """
        assert x_com.dtype == torch.float32 and x_intra.dtype == torch.float32
        B, *mid, Natom, _ = x_intra.shape
        view_shape = [B] + [1] * len(mid) + [Natom, 1]
        chain_id = self.chain_id.view(*view_shape).expand(x_intra.shape)
        pad_mask = self.pad_mask.view(*view_shape).expand(x_intra.shape)
        x_com = x_com.gather(-2, chain_id)  # [B, *, Natom, 3]
        x = x_com + x_intra
        # Zero out padding
        x.masked_fill_(~pad_mask, 0.0)
        return x


def add_com_noise(
    com: torch.Tensor,
    noise: torch.Tensor,
    tentacle_scale: float | torch.Tensor,
    radial_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Add noise to coordinates with separate tentacle and radial components for COM.

    Parameters
    ----------
    com : torch.Tensor
        Input COM coordinates. Shape (B, N, Nchain, 3).
    noise : torch.Tensor
        Noise tensor. Shape (B, N, Nchain, 3).
    tentacle_scale : float | torch.Tensor
        Scale for the tangential (tentacle) component of the COM noise.
    radial_scale : float | torch.Tensor
        Scale for the radial component of the COM noise.
    Returns
    -------
    com_noisy : torch.Tensor
        Noisy COM coordinates. Shape (B, N, Nchain, 3).
    """
    # Decompose noise into radial and tangential components
    com_norm = torch.norm(com, dim=-1, keepdim=True)

    radial_dir = com / com_norm.clamp(min=1e-8)
    radial_noise = (noise * radial_dir).sum(dim=-1, keepdim=True) * radial_dir
    tangential_noise = noise - radial_noise
    com_noise = radial_noise * radial_scale + tangential_noise * tentacle_scale

    # Edge case handling: if COM is near zero, apply isotropic noise instead
    near_zero_mask = com_norm < 1e-3
    isotropic_noise = noise * radial_scale  # Use radial_scale for isotropic noise
    com_noise = torch.where(near_zero_mask, isotropic_noise, com_noise)

    # Scale and combine noise components
    noisy_com = com + com_noise

    return noisy_com


# === Main ECSI module implementation === #
@STRUCTURE_MODULE.register()
class KFoldECSI_Decoupling(base_ecsi.KFoldECSI):
    r"""Endpoint-Conditioned Stochastic Interpolant module for structure prediction.

    Implements the ECSI framework from "Exploring the Design Space of Diffusion Bridge
    Models" for biomolecular structure prediction (apo -> holo translation).

    Key features:
    - Expanded bridge dynamics with separate COM and intra-coordinate noise paths
    - Linear route with shared power k:
      \alpha_t=1-t^k and \beta_t=t^k
    - Shared base gamma with component scales:
      \gamma_t^2=\gamma_{\max}^2/4 \cdot t^k(1-t^k)
      \gamma_{\mathrm{com}, t}=s_{\mathrm{com}}\gamma_t
      \gamma_{\mathrm{int}, t}=s_{\mathrm{int}}\gamma_t
    - Stochasticity control via \eta_{com}, \eta_{int} during sampling
    - Preconditioning adapted from DDBM using the shared base gamma

    Reference:
    - ECSI: Zhang et al., "Exploring the Design Space of Diffusion Bridge Models"
    """

    class Config(base_ecsi.KFoldECSI.Config):
        """Configuration for the ECSI structure module.

        Parameters
        ----------
        gamma_max : float, optional
            Shared base bridge maximum used by `gamma(t)`.
        time_power : float, optional
            Shared exponent `k` for the route coefficients
            `alpha_t = 1 - t^k`, `beta_t = t^k`, and the base gamma schedule.
        gamma_scale_intra : float, optional
            Multiplicative scale applied to the shared base gamma for the
            intra-coordinate branch of the expanded bridge dynamics.
        gamma_scale_com_tentacle : float, optional
            Multiplicative scale applied to the shared base gamma for the
            tangential component of the COM branch of the expanded bridge dynamics.
        gamma_scale_com_radial : float, optional
            Multiplicative scale applied to the shared base gamma for the
            radial component of the COM branch of the expanded bridge dynamics.
        sigma_data : float, optional
            Effective target-coordinate scale used in ECSI preconditioning.
        sigma_data_end : float, optional
            Effective source-coordinate scale used in ECSI preconditioning.
        cov_xy : float, optional
            Cross-covariance term between source and target coordinates used by
            the bridge preconditioning formulas.
        sampling : SamplingConfig, optional
            Reverse-time rollout configuration including stochasticity, endpoint
            perturbation, late ODE switching, and the nested step-allocation
            schedule.
        train_time_sampling : TrainTimeSamplingConfig, optional
            Training-time sampling policy for `t_hat`, including the optional
            Uniform mixture applied to the Beta branch.
        """

        gamma_scale_intra: float = 1.0
        gamma_scale_com_tentacle: float = 0.5  # Tangential noise scale for COM
        gamma_scale_com_radial: float = 0.5  # Radial noise scale for COM

    def __init__(self, cfg: Config, score_model: ECSIDiffusionModule):
        """Initialize the ECSI module.

        The constructor copies the high-level config fields onto runtime
        attributes, then immediately validates and normalizes the nested
        sampling config so downstream code can assume a runtime-ready ECSI
        configuration.
        """
        super().__init__(cfg, score_model)
        self.gamma_scale_intra = cfg.gamma_scale_intra
        self.gamma_scale_com_t = cfg.gamma_scale_com_tentacle
        self.gamma_scale_com_r = cfg.gamma_scale_com_radial

    # ============================================================
    # For training
    # ============================================================
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        diffusion_batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        batch_size = f_input.batch_size  # =B
        num_samples = diffusion_batch_size  # =N
        mask = f_input.atom.pad_mask  # [B, Natom]
        device = f_input.device
        decomposer = ChainDecomposition(f_input)

        t_hat = self.sample_noise_level((batch_size, num_samples), device)  # [B, N]

        # sample xT from label
        x_0, x_T = self.sample_x_0_and_x_T(f_input, num_samples)

        # sample xt via interpolation
        x_t = self.interpolate(x_0, x_T, t_hat, mask, decomposer)  # [B, N, Natom, 3]

        x_0_hat = self.forward_train(
            x_t=x_t,  # [B, N, Natom, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            x_T=x_T,  # [B, N, Natom, 3]
        )  # [B, N, Natom, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "x_t": x_t,
            "x_T": x_T,
            "x_0_hat": x_0_hat,
            "x_gt": x_0,
            "loss_weights": loss_weights,
        }

    def interpolate(  # type: ignore
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
        decomposer: ChainDecomposition,
    ) -> torch.Tensor:
        r"""Interpolate between x_0 and x_T using ECSI bridge.

        Samples from the bridge distribution:
        x_t = \alpha_t x_0 + \beta_t x_T + \gamma_t z, where z ~ N(0, I)

        Parameters
        ----------
        x_0 : torch.Tensor
            The target (holo) coordinates x_0. Shape (B, N, Natom, 3).
        x_T : torch.Tensor
            The source (apo) coordinates x_T. Shape (B, N, Natom, 3).
        t_hat : torch.Tensor
            Time values. Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, Natom).
        decomposer : ChainDecomposition
            ChainDecomposition object for decomposing/recomposing coordinates
            into COM/intra space for ECSI interpolation.

        Returns
        -------
        x_t : torch.Tensor
            Bridge-sampled coordinates x_t. Shape (B, N, Natom, 3).
        """
        B, N, Natom, _ = x_0.shape
        t_expanded = t_hat[:, :, None, None]
        alpha_t = self.si_coeffs.alpha(t_expanded)
        beta_t = self.si_coeffs.beta(t_expanded)
        gamma = self.si_coeffs.gamma(t_expanded)

        gamma_intra = gamma * self.gamma_scale_intra
        gamma_com_t = gamma * self.gamma_scale_com_t
        gamma_com_r = gamma * self.gamma_scale_com_r

        # Interpolate
        x_t_mean = alpha_t * x_0 + beta_t * x_T
        mean_com, mean_intra = decomposer.decompose(x_t_mean)

        # Add noise in COM/intra space
        noise_com = torch.randn_like(mean_com)
        _, noise_intra = decomposer.decompose(torch.randn_like(x_T))

        x_t_com = add_com_noise(mean_com, noise_com, gamma_com_t, gamma_com_r)
        x_t_intra = mean_intra + gamma_intra * noise_intra

        # Recompose to Cartesian coordinates
        x_t = decomposer.recompose(x_t_com, x_t_intra)

        if self.align_mode == RIGID_ALIGN:
            x_t = rigid_align(x_t, x_t_mean, mask=mask)
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
        r"""Sample structures via ECSI sampling.

        Implements Algorithm 1 from the paper with Euler discretization.
        Uses stochasticity control via `sampling_eta`-style parameters.

        The sampling SDE is:
        dX_t = b(t, X_t, x_T) dt + \sqrt{2\epsilon_t} dW_t

        where:
        b(t, x_t, x_T) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
                       + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / \gamma_t
        \epsilon_t = \eta (\gamma_t \dot{\gamma}_t - \dot{\alpha}_t/\alpha_t \gamma_t^2)
        """
        model = self.score_model

        # Construct chain decomposer
        decomposer = ChainDecomposition(f_input)

        # Get time schedule (from t_max toward t_min)
        times = self.get_sampling_schedule(num_steps)

        # Sample x_T from prior (apo structures)
        x_T = self.sample_prior(f_input, num_samples)  # (B, N, Natom, 3)
        x_t = x_T.clone()
        mask = f_input.atom.pad_mask[..., None, :]  # (B, 1, Natom)

        if self.sampling.perturb_x_t:
            x_t = self._apply_endpoint_perturbation(x_t, mask)
        x_churn_target = x_t.clone()

        # Compute time-independent variables
        z = model.get_pair_conditioning(f_input, z_trunk)
        q, c, p = model.get_atom_embeddings(f_input, s_inputs, s_trunk, z)
        pair_bias = model.get_pair_bias(z)
        del z_trunk, z  # Free up memory for large LxL tensors

        def run_step(x_t: torch.Tensor, t_hat: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t_hat), device=s_inputs.device)
            s = model.get_single_conditioning(s_inputs, s_trunk, c_noise.view(1, 1))
            return self.inference_step(
                f_input=f_input,
                x_t=x_t,
                x_T=x_T,
                t_hat=t_hat,
                q=q,
                c=c,
                p=p,
                s=s,
                pair_bias=pair_bias,
                chunk_size=chunk_size,
            )

        traj: list[torch.Tensor] = []

        def append_traj(x_t: torch.Tensor):
            if return_traj:
                traj.append(x_t.cpu())

        # Time schedule and branch control parameters
        churn_end_time: float = self.sampling.churn_end_time
        ode_start_time: float = self.sampling.ode_start_time

        # Sampling loop
        for step_idx in range(num_steps):
            append_traj(x_t)

            # Apply random augmentation
            x_t, x_T, x_churn_target = self.random_augmentation(
                x_t, x_T, x_churn_target, mask=mask
            )

            t = times[step_idx]
            t_next = times[step_idx + 1]
            dt = t_next - t  # negative since t decreases

            if churn_end_time < t:
                # Early-stage forward-pinned churn.
                x_t, t = self._apply_forward_pinned_churn(
                    x_t, x_churn_target, decomposer, t, t_next
                )
                dt = t_next - t

            # Get denoised prediction \hat{x}_0
            x_0_hat = run_step(x_t, t)

            if self.sampling.align_x_0_hat_to_x_t:
                # Rigidly align x_0_hat to x_t before centering.
                x_0_hat = rigid_align(x_0_hat, x_t, mask)

            if self.align_mode == NO_ALIGN:
                # Centering c_0_hat
                x_0_hat = do_centering(x_0_hat, mask)

            if t > ode_start_time:
                # Early/Mid-stage stochastic SI SDE update.
                x_t = self._sde_step(x_t, x_0_hat, x_T, decomposer, t, dt)
            else:
                # Late-stage deterministic ODE update.
                x_t = self._ode_step(x_t, x_0_hat, mask, t, dt)

        append_traj(x_t)

        sample_out: dict[str, torch.Tensor] = {}
        sample_out["init_coordinates"] = x_T
        sample_out["sample_coordinates"] = x_t
        if return_traj:
            sample_out["traj"] = torch.stack(traj, dim=-3)  # (B, N, num_steps, Natom, 3)

        return sample_out

    def sample_prior(self, f_input: FoldingInput, num_samples: int) -> torch.Tensor:
        """Sample xT (prior) coordinates for ECSI sampling.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_samples : int
            Number of diffusion samples

        Returns
        -------
        xT : torch.Tensor
            prior coordinates. Shape (B, N, Natom, 3).

        """
        # Sample from prior coordinates
        # If num_diffusion_samples > num_prior, cycle through prior coords
        all_prior_coords = f_input.atom.prior_coords  # [B, Natom, Nprior, 3]
        num_prior = all_prior_coords.shape[-2]
        idx = [i % num_prior for i in range(num_samples)]
        xT = all_prior_coords[:, :, idx, :]  # [B, Natom, N, 3]
        xT = xT.permute(0, 2, 1, 3)  # [B, N, Natom, 3]

        # Apply random augmentation to prior coords without centering.
        mask = f_input.atom.pad_mask[..., None, :]  # [B, 1, Natom]
        xT = self.random_augmentation(xT, mask=mask)
        return xT

    # ============================================================
    # Sampling utilities
    # ============================================================
    def _apply_forward_pinned_churn(
        self,
        x_t: torch.Tensor,
        x_churn_target: torch.Tensor,
        decomposer: ChainDecomposition,
        t: float,
        t_next: float,
    ) -> tuple[torch.Tensor, float]:
        if t <= self.sampling.churn_end_time:
            # No churn applied before or at churn_end
            return x_t, t

        dt = abs(t_next - t)
        dt_churn = self.sampling.churn_factor * dt
        if t + dt_churn >= self.sampling.time_max:
            # Do not apply churn
            return x_t, t

        alpha_t: float = self.si_coeffs.alpha(t)
        beta_t: float = self.si_coeffs.beta(t)
        alpha_dot: float = self.si_coeffs.alpha_deriv(t)
        beta_dot: float = self.si_coeffs.beta_deriv(t)
        gamma = self.si_coeffs.gamma(t)
        gamma_dot = self.si_coeffs.gamma_deriv(t)

        f_t = alpha_dot / alpha_t
        s_t = beta_dot - f_t * beta_t
        g = _sqrt(2 * (gamma * gamma_dot - f_t * gamma**2))

        # Scale COM/intra noise separately
        g_com_t = g * self.gamma_scale_com_t
        g_com_r = g * self.gamma_scale_com_r
        g_intra = g * self.gamma_scale_intra

        # Decompose coordinates into COM/intra space
        x_t_com, x_t_intra = decomposer.decompose(x_t)
        x_target_com, x_target_intra = decomposer.decompose(x_churn_target)

        # Euler update with forward-pinned noise
        mean_com = x_t_com + (f_t * x_t_com + s_t * x_target_com) * dt_churn
        mean_intra = x_t_intra + (f_t * x_t_intra + s_t * x_target_intra) * dt_churn

        noise_com = torch.randn_like(x_t_com)
        x_t_com = add_com_noise(mean_com, noise_com * (dt_churn**0.5), g_com_t, g_com_r)

        _, noise_intra = decomposer.decompose(torch.randn_like(x_t))
        x_t_intra = mean_intra + g_intra * noise_intra * (dt_churn**0.5)

        # Recompose to Cartesian coordinates
        x_t = decomposer.recompose(x_t_com, x_t_intra)
        t += dt_churn
        return x_t, t

    def _sde_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        x_T: torch.Tensor,
        decomposer: ChainDecomposition,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        si_coeffs = self.si_coeffs
        alpha_t: float = si_coeffs.alpha(t)
        beta_t: float = si_coeffs.beta(t)
        alpha_dot: float = si_coeffs.alpha_deriv(t)
        beta_dot: float = si_coeffs.beta_deriv(t)
        gamma: float = si_coeffs.gamma(t)
        gamma_dot: float = si_coeffs.gamma_deriv(t)
        eps: float = si_coeffs.eps(t)

        # Compute \hat{z}_t
        # \hat{z}_t = (x_t - \alpha_t \hat{x}_0 - \beta_t x_T) / gamma
        raw_z = x_t - alpha_t * x_0_hat - beta_t * x_T
        base_z = raw_z / gamma

        # Compute drift b(t)
        # b(t) = \dot{\alpha}_t \hat{x}_0 + \dot{\beta}_t x_T
        #        + (\dot{\gamma}_t + \epsilon_t/\gamma_t) \hat{z}_t
        drift = (alpha_dot * x_0_hat + beta_dot * x_T) + (
            (gamma_dot + eps / gamma) * base_z
        )

        # === Euler-Maruyama update ===
        # Update drift
        x_upd_mean = x_t + drift * dt

        # Decompose updated mean into COM/intra space for noise scheduling
        mean_com, mean_intra = decomposer.decompose(x_upd_mean)

        # Compute noise for COM and intra space
        noise_com = torch.randn_like(mean_com)
        _, noise_intra = decomposer.decompose(torch.randn_like(x_t))

        # Update in COM/intra space
        noise_scale_com_t = self.gamma_scale_com_t * abs(2 * eps * dt) ** 0.5
        noise_scale_com_r = self.gamma_scale_com_r * abs(2 * eps * dt) ** 0.5
        noise_scale_intra = self.gamma_scale_intra * abs(2 * eps * dt) ** 0.5
        x_com = add_com_noise(mean_com, noise_com, noise_scale_com_t, noise_scale_com_r)
        x_intra = mean_intra + noise_scale_intra * noise_intra

        # Recompose to Cartesian coordinates
        x_update = decomposer.recompose(x_com, x_intra)

        return x_update

    def _ode_step(
        self,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        mask: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        coeffs = self.si_coeffs
        alpha, beta = coeffs.alpha(t), coeffs.beta(t)
        alpha_next, beta_next = coeffs.alpha(t + dt), coeffs.beta(t + dt)
        c_skip: float = beta_next / beta
        c_update: float = alpha_next - alpha * c_skip
        x_update = c_skip * x_t + c_update * x_0_hat
        return x_update
