from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model import BaseScoreModel
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.misc import repeat_dim
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig


@STRUCTURE_MODULE.register()
class BaseStructureModule(ABC):
    """High-level flow-based framework for structure generation."""

    def __init__(self, cfg: BaseConfig, score_model: BaseScoreModel):
        self.cfg = cfg
        self.score_model = score_model

    # === Sampling and Interpolation Methods === #
    @abstractmethod
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
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def sample_noise_level(
        self,
        shape: tuple,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels (t_hat) during model training. Shape: (B, N)."""

    @abstractmethod
    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        """Get the noise schedule for diffusion sampling. Shape: (num_steps,)."""

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat. Shape: (B, N)."""
        return torch.ones_like(t_hat)

    def apply_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates. Shape (*, L, 3).
        mask : torch.Tensor
            Mask. Shape (*, L).

        Returns
        -------
        augmented_coords : torch.Tensor
            Augmented Coordinates. Shape (*, La, 3).
        """
        # Default: simple centering without augmentation
        coords = do_centering(coords, mask, mask_to_zero=True)
        return coords

    def sample_label(self, f_input: FoldingInput, num_samples: int = 1) -> torch.Tensor:
        """Sample label coordinates from input features.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.

        Parameters
        -----------
        f_input: FoldingInput
            Input features
        num_samples:
            Number of diffusion samples

        Returns
        -------
        label_coords: torch.Tensor
            Label coordinates. Shape: [B, N, La, 3]
        """
        # if the model is equivariance, skip augment
        random_augment = True

        holo_coords = self.sample_holo(f_input, num_samples, random_augment)

        return holo_coords

    @abstractmethod
    def sample_prior(
        self,
        f_input: FoldingInput,
        num_samples: int,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.

        Parameters
        -----------
        f_input: FoldingInput
            Input features
        num_samples:
            Number of diffusion samples
        label_coords: torch.Tensor
            Label coordinates. Shape: [B, N, La, 3]
            where N is the number of diffusion samples

        Returns
        -------
        prior_coords: torch.Tensor
            Prior coordinates. Shape: [B, N, La, 3]
        """

    @abstractmethod
    def interpolate(
        self,
        x_T: torch.Tensor,
        x_0: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        x_T : torch.Tensor
            The prior coordinates. Shape (B, N, La, 3).
        x_0 : torch.Tensor
            The label coordinates. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            The dffusion noise levels (or sigmas of EDM). Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).

        Returns
        -------
        interpolated_coords : torch.Tensor
            The interpolated coordinates. Shape (B, N, La, 3).
        """

    # === For model training === #
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
        """Forward pass for training. Returns denoised coordinates."""
        raise NotImplementedError("forward_train must be implemented in subclass")

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
        raise NotImplementedError("training_step must be implemented in subclass")

    # === Sampling holo/apo structures === #
    def sample_holo(
        self,
        f_input: FoldingInput,
        num_samples: int = 1,
        random_augment: bool = True,
    ) -> torch.Tensor:
        """Sample holo structures from input for model training.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_samples : int, optional
            Number of diffusion samples to generate, by default 1.
        random_augment: bool
            Whether to apply random augmentation to holo coordinates.

        Returns
        -------
        holo_coords : torch.Tensor
            Sampled holo coordinates. Shape (B, N, L, 3),
            where N is number of diffusion samples and L is the number of atoms.
        """
        holo_coords = f_input.atom.label_coords  # [B, L, 3]
        atom_mask = f_input.atom.resolved_mask  # [B, L]

        # repeat holo coords
        holo_coords = repeat_dim(holo_coords, num_samples, dim=-3)  # [B, N, L, 3]
        atom_mask = atom_mask.unsqueeze(-2)  # [B, 1, L]

        # Apply coordinate augmentation or centering
        if random_augment:
            holo_coords = self.apply_random_augmentation(holo_coords, atom_mask)

        return holo_coords  # [B, N, L, 3]


@STRUCTURE_MODULE.register()
class BaseEDM(BaseStructureModule):
    """High-level EDM framework for structure generation.

    See Section 3.7, Algorithm18 (Sample Diffusion) of AlphaFold3 paper.
    """

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

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat. Shape: (B, N)."""
        return 1 / self.c_out(t_hat) ** 2

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
        mask = f_input.atom.pad_mask  # [B, La]
        device = f_input.device

        t_hat = self.sample_noise_level((batch_size, num_samples), device)  # [B, N]

        # sample x0 from label
        x_gt = self.sample_label(f_input, num_samples)

        # sample xT from prior
        x_prior = self.sample_prior(f_input, num_samples, x_gt)

        # sample xt via interpolation
        x_t = self.interpolate(x_prior, x_gt, t_hat, mask)

        x_0_hat = self.forward_train(
            x_t=x_t,  # [B, N, La, 3]
            t_hat=t_hat,  # [B, N]
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
            "x_gt": x_gt,
            "loss_weights": loss_weights,
        }


@STRUCTURE_MODULE.register()
class BaseECSI(BaseEDM):
    """High-level ECSI framework for structure generation."""
