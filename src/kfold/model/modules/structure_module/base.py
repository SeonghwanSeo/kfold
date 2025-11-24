from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.modules.score_model import BaseScoreModel
from kfold.utils.geometry.random_augment import center_random_augmentation
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig


@STRUCTURE_MODULE.register()
class BaseStructureModule(ABC):
    """High-level diffusion framework for structure generation.
    You may want to implement diffusion bridge methods here.

    See Section 3.7, Algorithm18 (Sample Diffusion) of AlphaFold3 paper.

    NOTE: this is not a torch.nn.Module, as it may not have learnable parameters.
    """

    def __init__(self, cfg: BaseConfig, score_model: BaseScoreModel):
        self.cfg = cfg
        self.score_model = score_model

    # === EDM (Elucidating Diffusion Models) preconditioning coefficients === #
    # Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based "
    # Generative Models"
    # Check Boltz Implementation
    @abstractmethod
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection coefficient for EDM preconditioning."""

    @abstractmethod
    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output scaling coefficient for EDM preconditioning."""

    @abstractmethod
    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling coefficient for EDM preconditioning."""

    @abstractmethod
    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise level conditioning coefficient for EDM preconditioning."""

    # === Sampling and Interpolation Methods === #
    @abstractmethod
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
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def sample_noise_level(
        self,
        batch_size: int,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels (t_hat) during model training. Shape: (B, N)."""

    @abstractmethod
    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the noise schedule for diffusion sampling."""

    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.

        Parameters
        -----------
        f_input: FoldingInput
            Input features
        num_diffusion_samples:
            Number of diffusion samples
        label_coords: torch.Tensor
            Label coordinates. Shape: [B, N, La, 3]
            where N is the number of diffusion samples
        """

        # if the model is equivariance, skip augment
        random_augment = True

        # Sample apo coordinates
        apo_coords = self.sample_apo(f_input, num_diffusion_samples, random_augment)

        # If required, align to label coordinates
        return apo_coords

    @abstractmethod
    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        noise_coords : torch.Tensor
            The noisy coordinates. Shape (B, N, La, 3).
        label_coords : torch.Tensor
            The label coordinates. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            The dffusion noise levels (or sigmas of EDM). Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).
        """

    @abstractmethod
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
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3 paper.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, La, 3),
            where N is number of diffusion samples and La is number of atoms.
        t_hat : torch.Tensor | float
            Diffusion noise level (or sigmas of EDM). Shape (B, N,).
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, Lt, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, Lt, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, Lt, Lt, c_z).
        model_cache : optional
            Model cache for efficiency.

        Returns
        -------
        denoised_coords : torch.Tensor
            Denoised atom coordinates. Shape (B, N, Lt, 3).
        """

    def sample_apo(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        random_augment: bool = False,
    ) -> torch.Tensor:
        """Sample apo structures from input for model training/inference.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples to generate, by default 1.
        random_augment:
            Whether to apply random augmentation to apo coordinates.
            NOTE: the apo coordinates should be already randomly augmented per
            each chain during featurization. (See `do_augment_apo_structure`)

        Returns
        -------
        apo_coords : torch.Tensor
            Sampled apo coordinates. Shape (B, N, L, 3),
            where N is number of diffusion samples and L is the number of atoms.
        """

        all_apo_coords = f_input.atom.apo_coords  # [B, L, Napo, 3]
        all_apo_coords = all_apo_coords.permute(
            0, 2, 1, 3
        ).contiguous()  # [B, Napo, L, 3]
        all_apo_mask = f_input.atom.apo_mask  # [B, L, Napo, 3]
        all_apo_mask = all_apo_mask.permute(0, 2, 1).contiguous()  # [B, Napo, L]

        B, Napo, L = all_apo_mask.shape  # noqa
        assert Napo == 1
        # TODO(SeonghwanSeo): Currently only supports a single apo structure (Napo == 1).
        # Update this code to support multiple apo structures in the future.
        sampled_apo_coords = all_apo_coords.repeat(1, num_diffusion_samples, 1, 1)
        sampled_apo_mask = all_apo_mask.repeat(1, num_diffusion_samples, 1)

        if random_augment:
            sampled_apo_coords = center_random_augmentation(
                sampled_apo_coords,
                sampled_apo_mask,
                s_trans=0.0,  # Keep center to zero.
            )

        # Mask out to zero
        sampled_apo_coords = sampled_apo_coords * sampled_apo_mask[..., None]

        return sampled_apo_coords  # [B, N, L, 3]

    # === For model training === #
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

            # sample xt from label (Currently, there is only one holo structure per input)
            holo_coords = self.sample_holo(f_input, num_diffusion_samples)

            # sample x0 from prior
            prior_coords = self.sample_prior(f_input, num_diffusion_samples, holo_coords)

            noised_atom_coords = self.interpolate(prior_coords, holo_coords, t_hat, mask)
            noised_atom_coords = noised_atom_coords * mask[:, None, :, None]

        denoised_atom_coords = self.forward_model(
            x_noisy=noised_atom_coords,  # [B, N, La, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            model_cache=model_cache,
            prior_coords=prior_coords,  # [B, N, La, 3]
        )  # [B, N, La, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "loss_weights": loss_weights,
            "prior_atom_coords": prior_coords,
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "true_atom_coords": holo_coords,
        }

    @abstractmethod
    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat.
        See Section 3.7.1 Equation 6 of AlphaFold3 paper.
        """

    def sample_holo(
        self, f_input: FoldingInput, num_diffusion_samples: int = 1
    ) -> torch.Tensor:
        """Sample holo structures from input for model training.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples to generate, by default 1.

        Returns
        -------
        holo_coords : torch.Tensor
            Sampled holo coordinates. Shape (B, N, L, 3),
            where N is number of diffusion samples and L is the number of atoms.
        """

        holo_coords = f_input.atom.label_coords  # [B, L, Nholo, 3]
        holo_coords = holo_coords.permute(0, 2, 1, 3)  # [B, Nholo, L, 3]
        Nholo = holo_coords.shape[1]
        if Nholo != 1:
            raise NotImplementedError(
                "Multiple holo structures per input not supported yet."
            )

        if Nholo == 1:
            # repeat holo coords
            holo_coords = holo_coords.repeat(1, num_diffusion_samples, 1, 1)
        else:
            # sample holo indices
            raise NotImplementedError("sample not implemented")

        # Mask out unresolved atoms
        atom_mask = f_input.atom.resolved_mask  # (B, Latom)
        atom_mask = atom_mask.to(holo_coords.dtype)[:, None, :, None]
        holo_coords = holo_coords * atom_mask

        return holo_coords  # [B, N, L, 3]
