import torch

from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.utils.registry import TRANSFORMER_MODULE

from .base import BaseTransformer


@TRANSFORMER_MODULE.register()
class PairformerModule(BaseTransformer):
    """Pairformer module."""

    class Config(BaseTransformer.Config):
        """Configuration for the Pairformer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False
        """

        channel_s: int = 384
        channel_z: int = 128
        num_blocks: int = 48
        num_heads: int = 16
        dropout: float = 0.25
        pairwise_head_width: int = 32
        pairwise_num_heads: int = 4
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        use_kernels: bool = False

    def __init__(self, cfg: Config, use_kernels: bool = False):
        """Initialize the Pairformer module."""
        super().__init__(cfg)

        self.trunk = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            pairwise_head_width=cfg.pairwise_head_width,
            pairwise_num_heads=cfg.pairwise_num_heads,
            activation_checkpointing=cfg.activation_checkpointing,
            offload_to_cpu=cfg.offload_to_cpu,
            use_kernels=cfg.use_kernels or use_kernels,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

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
        if not self.training:
            if z.shape[1] > 384:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        s_trunk, z_trunk = self.trunk(s, z, mask, pair_mask, chunk_size_tri_attn)
        return s_trunk, z_trunk
