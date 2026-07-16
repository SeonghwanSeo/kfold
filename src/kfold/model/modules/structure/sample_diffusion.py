from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

from .score_model import DiffusionModule


@STRUCTURE_MODULE.register()
class BaseStructureModule(ABC):
    """High-level flow-based framework for structure generation."""

    class Config(BaseConfig): ...

    def __init__(self, cfg: BaseConfig, score_model: DiffusionModule):
        self.cfg = cfg
        self.score_model = score_model

    # === Sampling and Interpolation Methods === #
    @abstractmethod
    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        num_steps: int = 100,
        num_samples: int = 1,
        chunk_size: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def get_sampling_schedule(self, num_steps: int) -> list[float]:
        """Get the noise schedule for diffusion sampling. Shape: (num_steps,)."""

    # === For model training === #
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        raise NotImplementedError("training_step must be implemented in subclass")

    def _forward_train(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for training. Returns denoised coordinates."""
        raise NotImplementedError("forward_train must be implemented in subclass")

    @abstractmethod
    def sample_noise_level(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample noise levels (t_hat) during model training. Shape: (B, N)."""

    @abstractmethod
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
