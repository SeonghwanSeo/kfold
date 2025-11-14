from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import SCORE_MODEL


@SCORE_MODEL.register()
class BaseScoreModel(torch.nn.Module, ABC):
    """Base class for diffusion score model modules.
    See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.is_compiled: bool = False

    def compile(self, compile: bool = True):
        """Compile the score model module."""
        if compile:
            self.do_compile()
            self.is_compiled = True

    def do_compile(self):
        """Compile the score model module."""
        self.forward = torch.compile(self.forward, dynamic=False, fullgraph=False)  # type: ignore

    @abstractmethod
    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass of the AF3 diffusion module.
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        x_noisy : torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        t_hat : torch.Tensor
            The diffusion noise level (or sigmas), shape [B, N].
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, c_z].


        Returns
        -------
        x_out : torch.Tensor
            The denoised atom positions, shape [B, N, La, 3].
        """
