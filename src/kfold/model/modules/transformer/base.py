from abc import ABC, abstractmethod

import torch

from kfold.data.model_input import FoldingInput
from kfold.utils.registry import STRUCT_REPR_MODULE, BaseConfig

# TODO (seonghwanseo): we can define some common parameters across different transformer
# architectures, like seq_channel, token_channel, atom_channel, etc.


class BaseTransformerConfig(BaseConfig):
    c_s: int = 384
    c_z: int = 128
    num_blocks: int = 48


@STRUCT_REPR_MODULE.register(config_cls=BaseTransformerConfig)
class BaseTransformer(torch.nn.Module, ABC):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        input: FoldingInput,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of transformer module.

        Parameters
        ----------
        input : FoldingInput
            FoldingInput object containing model inputs.
        s : torch.Tensor
            Tensor of shape (B, L, c_s) containing token single feature
        z : torch.Tensor
            Tensor of shape (B, L, L, c_z) containing token pair feature

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, N, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, N, N, c_z).
        """
