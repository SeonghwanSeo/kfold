from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import TRUNK, BaseConfig

# TODO (seonghwanseo): we can define some common parameters across different transformer
# architectures, like seq_channel, token_channel, atom_channel, etc.


@TRUNK.register()
class BaseTrunk(torch.nn.Module, ABC):
    class Config(BaseConfig):
        channel_s: int = 384
        channel_z: int = 128
        num_blocks: int = 48

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.is_compiled: bool = False

    def compile(self, compile: bool = True):
        """Compile the trunk module."""
        if compile:
            self.do_compile()
            self.is_compiled = True

    def do_compile(self):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        raise NotImplementedError("do_compile method is not implemented yet.")

    @abstractmethod
    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_cycles: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3 Algorithm 1 Main Inference Loop: Line[6-14]

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_inits: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
        z_inits: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing initial pair representation
        f_input : FoldingInput
            The input features.
        num_cycles : int
            The number of recycling steps.

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
