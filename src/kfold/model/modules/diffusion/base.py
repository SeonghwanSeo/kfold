from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import DIFFUSION_MODULE


@DIFFUSION_MODULE.register()
class BaseDiffusionModule(torch.nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        input: FoldingInput,
        x_t: torch.Tensor,
        t_hat: torch.Tensor,
        s_input: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of diffusion module.
        See Algorithm 20 of AlphaFold3 paper for more details.

        Parameters
        ----------
        input : FoldingInput
            FoldingInput object containing model inputs.
        x_t : torch.Tensor
            Tensor of shape (B, N, 3) containing noised coordinates at step t.
        t_hat : torch.Tensor
            Tensor of shape (B,) containing time step.

        """
