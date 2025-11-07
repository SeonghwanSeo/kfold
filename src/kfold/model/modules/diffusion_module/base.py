from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import DIFFUSION_MODULE


@DIFFUSION_MODULE.register()
class BaseDiffusionModule(torch.nn.Module, ABC):
    """Base class for diffusion score model modules.
    See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        x_noisy: torch.Tensor,
        times: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache=None,
    ) -> torch.Tensor:
        """Forward pass of embedding module.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Tensor of shape (N_samples, L, 3) containing noisy atom positions.
        time : torch.Tensor
            Tensor of shape (N_samples,) containing diffusion times.
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Tensor of shape (L, c_s) containing input sequence embeddings.
        s_trunk : torch.Tensor
            Tensor of shape (L, c_s) containing trunk sequence embeddings.
        z_trunk : torch.Tensor
            Tensor of shape (L, L, c_z) containing trunk pairwise embeddings.
        times : torch.Tensor
            Tensor of shape (N_samples,) containing diffusion times.

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (L, C_s) containing sequence embeddings.
        """
