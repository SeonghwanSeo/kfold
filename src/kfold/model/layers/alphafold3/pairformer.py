"""Section 3.6 Pairformer Stack of AlphaFold 3 paper."""

# started from code from https://github.com/jwohlwend/boltz, MIT License,

import torch
import torch.nn as nn
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from .dropout import get_dropout_mask
from .transformers import AttentionPairBias
from .transition import Transition
from .triangular_update import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairformerStack(nn.Module):
    """Pairformer stack.
    See Section 3.6 Algorithm 20 Pairformer Stack
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_blocks: int = 48,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        use_kernels: bool = False,
    ):
        """Initialize the Pairformer module."""
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_blocks: int = num_blocks
        self.dropout: float = dropout
        self.num_heads: int = num_heads
        self.pairwise_head_width: int = pairwise_head_width
        self.pairwise_num_heads: int = pairwise_num_heads
        self.activation_checkpointing: bool = activation_checkpointing
        self.offload_to_cpu: bool = offload_to_cpu
        self.use_kernels: bool = use_kernels

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if activation_checkpointing:
                self.blocks.append(
                    checkpoint_wrapper(
                        PairformerBlock(
                            self.channel_s,
                            self.channel_z,
                            self.num_heads,
                            self.dropout,
                            self.pairwise_head_width,
                            self.pairwise_num_heads,
                            use_kernels=self.use_kernels,
                        ),
                        offload_to_cpu=self.offload_to_cpu,
                    )
                )
            else:
                self.blocks.append(
                    PairformerBlock(
                        self.channel_s,
                        self.channel_z,
                        self.num_heads,
                        self.dropout,
                        self.pairwise_head_width,
                        self.pairwise_num_heads,
                        use_kernels=self.use_kernels,
                    )
                )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
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

        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        torch.Tensor
            The updated pairwise embeddings.

        """
        if self.training:
            assert chunk_size_tri_attn is None, (
                "During training, chunk_size_tri_attn must be None."
            )

        pair_mask = mask[:, :, None] * mask[:, None, :]

        # Line 1
        for block in self.blocks:
            # Line 2-8
            s, z = block(s, z, mask, pair_mask, chunk_size_tri_attn)
        # Line 10
        return s, z


class PairformerBlock(nn.Module):
    """Pairformer block.
    See Section 3.6 Algorithm 20 Pairformer Stack : Line [2-8]
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
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
        use_kernels : bool, optional
            Whether to use custom kernels, by default False

        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_heads: int = num_heads
        self.use_kernels: bool = use_kernels

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.attention = AttentionPairBias(
            channel_s, 0, channel_z, num_heads, use_s=False
        )

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
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """

        # Line 2
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        # Line 3
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        # Line 4
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=self.use_kernels,
        )

        # Line 5
        dropout = get_dropout_mask(z, self.dropout, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=self.use_kernels,
        )

        # Line 6
        z = z + self.transition_z(z)

        # Line 7
        s = s + self.attention(
            s,  # [B, L, C_s]
            None,
            z,  # [B, L, L, C_z]
            attn_mask=mask.unsqueeze(-2),  # [B, 1, L], broadcast to [B, L, L]
        )

        # Line 8
        s = s + self.transition_s(s)

        return s, z
