from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.modules.diffusion_module import BaseDiffusionModule
from kfold.utils.registry import DIFFUSION_MODULE, BaseConfig


@DIFFUSION_MODULE.register()
class BaseStructureModule(ABC):
    """High-level diffusion framework for structure generation.
    You may want to implement diffusion bridge methods here.

    See Section 3.7, Algorithm18 (Sample Diffusion) of AlphaFold3 paper.

    NOTE: this is not a torch.nn.Module, as it may not have learnable parameters.
    """

    def __init__(self, cfg: BaseConfig, score_model: BaseDiffusionModule):
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
        num_sampling_steps: int | None = None,
        num_diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
    ) -> torch.Tensor:
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def sample_sigma(
        self,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels (sigma) during model training."""

    @abstractmethod
    def get_sampling_schedule(
        self,
        num_sampling_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the noise schedule for diffusion sampling."""

    @abstractmethod
    def sample_prior(
        self, f_input: FoldingInput, num_diffusion_samples: int = 1
    ) -> torch.Tensor:
        """Sample from the prior distribution."""
        apo_coords = self.sample_apo(f_input, num_diffusion_samples)  # noqa
        # do apo perturbation according to prior distribution
        raise NotImplementedError("sample not implemented")

    @abstractmethod
    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        sigma: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Interpolate between apo (noised) and holo (label) coordinates.

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        noise_coords : torch.Tensor
            The noisy coordinates. Shape (B, L, 3).
        label_coords : torch.Tensor
            The label coordinates. Shape (B, L, 3).
        sigma : torch.Tensor
            The sigma values. Shape (B,).
        f_input : FoldingInput
            The FoldingInput object.

        L: number of atoms
        """

    @abstractmethod
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

    def sample_apo(
        self, f_input: FoldingInput, num_diffusion_samples: int = 1
    ) -> torch.Tensor:
        """Sample apo structures from input for model training/inference.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples to generate, by default 1.

        Returns
        -------
        apo_coords : torch.Tensor
            Sampled apo coordinates. Shape (num_diffusion_samples, L, 3),
            where L is the number of atoms.
        """

        apo_coords = f_input.atom.apo_coords  # [L, Napo, 3]
        apo_coords = apo_coords.permute(1, 0, 2)  # [Napo, L, 3]
        Napo = apo_coords.shape[0]

        if Napo == 1:
            apo_coords = apo_coords.repeat(num_diffusion_samples, 1, 1)
        else:
            # sample apo indices
            apo_indices = torch.randint(0, Napo, (num_diffusion_samples,))
            apo_coords = apo_coords[apo_indices]

        raise NotImplementedError("sample not implemented")

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
            Sampled holo coordinates. Shape (num_diffusion_samples, L, 3),
            where L is the number of atoms.
        """

        holo_coords = f_input.atom.apo_coords  # [L, Nholo, 3]
        holo_coords = holo_coords.permute(1, 0, 2)  # [Nholo, L, 3]
        Nholo = holo_coords.shape[0]

        if Nholo == 1:
            holo_coords = holo_coords.repeat(num_diffusion_samples, 1, 1)
        else:
            # sample holo indices
            holo_indices = torch.randint(0, Nholo, (num_diffusion_samples,))
            holo_coords = holo_coords[holo_indices]
        return holo_coords  # [num_diffusion_samples, L, 3]

    def sample_noise_structure(
        self, f_input: FoldingInput, num_diffusion_samples: int = 1
    ):
        """Sample noised structures from input for model training."""
        sigma = self.sample_sigma(num_diffusion_samples, device=f_input.device)
        prior_coords = self.sample_prior(f_input, num_diffusion_samples)
        holo_coords = self.sample_holo(f_input, num_diffusion_samples)
        noised_atom_coords = self.interpolate(prior_coords, holo_coords, sigma, f_input)
        return {
            "sigmas": sigma,
            "prior_atom_coords": prior_coords,
            "noised_atom_coords": noised_atom_coords,
            "label_atom_coords": holo_coords,
        }
