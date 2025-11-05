import torch
import torch.nn as nn
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from kfold.model.layers.boltz.attention import AttentionPairBias
from kfold.model.layers.boltz.dropout import get_dropout_mask
from kfold.model.layers.boltz.transition import Transition
from kfold.model.layers.boltz.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from kfold.model.layers.boltz.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.utils.registry import TRANSFORMER_MODULE

from .base import BaseTransformerConfig


class PairformerConfig(BaseTransformerConfig):
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
    no_update_z : bool, optional
        Whether to update the pairwise embeddings, by default False
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
    no_update_s: bool = False
    no_update_z: bool = False
    offload_to_cpu: bool = False


@TRANSFORMER_MODULE.register(config_cls=PairformerConfig)
class PairformerModule(nn.Module):
    """Pairformer module."""

    def __init__(self, cfg: PairformerConfig, use_kernel: bool = False):
        """Initialize the Pairformer module."""
        super().__init__()
        self.channel_s: int = cfg.channel_s
        self.channel_z: int = cfg.channel_z
        self.num_blocks: int = cfg.num_blocks
        self.dropout: float = cfg.dropout
        self.num_heads: int = cfg.num_heads
        self.pairwise_head_width: int = cfg.pairwise_head_width
        self.pairwise_num_heads: int = cfg.pairwise_num_heads
        self.activation_checkpointing: bool = cfg.activation_checkpointing
        self.no_update_s: bool = cfg.no_update_s
        self.no_update_z: bool = cfg.no_update_z
        self.offload_to_cpu: bool = cfg.offload_to_cpu
        self.use_kernels: bool = use_kernel

        self.layers = nn.ModuleList()
        for i in range(cfg.num_blocks):
            no_update_z = False if i < self.num_blocks - 1 else cfg.no_update_z
            if cfg.activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        PairformerLayer(
                            self.channel_s,
                            self.channel_z,
                            self.num_heads,
                            self.dropout,
                            self.pairwise_head_width,
                            self.pairwise_num_heads,
                            no_update_s=self.no_update_s,
                            no_update_z=no_update_z,
                            use_kernels=self.use_kernels,
                        ),
                        offload_to_cpu=self.offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    PairformerLayer(
                        self.channel_s,
                        self.channel_z,
                        self.num_heads,
                        self.dropout,
                        self.pairwise_head_width,
                        self.pairwise_num_heads,
                        no_update_s=self.no_update_s,
                        no_update_z=no_update_z,
                        use_kernels=self.use_kernels,
                    )
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
            The sequence embeddings
        z : torch.Tensor
            The pairwise embeddings
        mask : torch.Tensor
            The token mask
        pair_mask : torch.Tensor
            The pairwise mask

        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        torch.Tensor
            The updated pairwise embeddings.

        """
        if not self.training:
            if z.shape[1] > 384:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(
                s,
                z,
                mask,
                pair_mask,
                chunk_size_tri_attn,
            )
        return s, z


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        channel_s: int,
        channel_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
        use_kernels: bool = False,
    ):
        """Initialize the Pairformer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False
        use_kernels : bool, optional
            Whether to use custom kernels, by default False

        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_heads: int = num_heads
        self.no_update_s: bool = no_update_s
        self.no_update_z: bool = no_update_z
        self.use_kernels: bool = use_kernels

        if not self.no_update_s:
            self.attention = AttentionPairBias(channel_s, channel_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        if not self.no_update_s:
            self.transition_s = Transition(channel_s, channel_s * 4)
        self.transition_z = Transition(channel_z, channel_z * 4)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=self.use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=self.use_kernels,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)

        return s, z
