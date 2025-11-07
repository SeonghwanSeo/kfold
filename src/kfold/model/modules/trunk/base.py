from abc import ABC, abstractmethod

import torch

from kfold.utils.registry import TRANSFORMER_MODULE, BaseConfig

# TODO (seonghwanseo): we can define some common parameters across different transformer
# architectures, like seq_channel, token_channel, atom_channel, etc.


@TRANSFORMER_MODULE.register()
class BaseTransformer(torch.nn.Module, ABC):
    class Config(BaseConfig):
        channel_s: int = 384
        channel_z: int = 128
        num_blocks: int = 48

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of transformer module.

        Parameters
        ----------
        s : torch.Tensor
            Tensor of shape (B, L, c_s) containing token single feature
        z : torch.Tensor
            Tensor of shape (B, L, L, c_z) containing token pair feature
        mask : torch.Tensor
            The token mask of shape (B, L)
        pair_mask : torch.Tensor
            The pairwise mask of shape (B, L, L)

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
