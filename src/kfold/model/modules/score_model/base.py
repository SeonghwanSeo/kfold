from abc import ABC, abstractmethod

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import SCORE_MODEL


@SCORE_MODEL.register()
class BaseScoreModel(torch.nn.Module, ABC):
    """Base class for diffusion score model modules.
    See Section 3.7: Diffusion Module, Algorithm 20 of AlphaFold3
    """

    def __init__(self, cfg, kernel_config):
        super().__init__()
        self.cfg = cfg
        self.kernel_config = kernel_config
        self.is_compiled: bool = False

    def compile(self, compile: bool = True, mode: str = "default"):
        """Compile the score model module."""
        if compile:
            self.do_compile(mode)
            self.is_compiled = True

    def do_compile(self, mode: str = "default"):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        raise NotImplementedError("do_compile method is not implemented yet.")

    @abstractmethod
    def train_step(
        self,
        f_input: FoldingInput,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the AF3 diffusion module.
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.
        Notes: The scaling of x_noisy is handled outside this module.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        r_noisy : torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        c_noise : torch.Tensor
            The diffusion noise level (or sigmas), shape [B, N].
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm 21.)
            c_noise is computed outside of this class (See StructureModule).
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, c_z].


        Returns
        -------
        r_update : torch.Tensor
            The denoised atom positions, shape [B, N, La, 3].
        """
