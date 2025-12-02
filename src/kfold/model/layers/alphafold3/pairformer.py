"""Section 3.6 Pairformer Stack of AlphaFold 3 paper."""

# started from code from https://github.com/jwohlwend/boltz, MIT License,

from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.primitives.dropout import get_dropout_mask
from kfold.model.layers.primitives.triangle_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from kfold.model.layers.primitives.triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.utils.checkpointing import checkpoint_blocks

from .transformers import AttentionPairBias
from .transition import Transition


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
        blocks_per_ckpt: int | None = None,
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

        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                PairformerBlock(
                    self.channel_s,
                    self.channel_z,
                    self.num_heads,
                    self.dropout,
                    self.pairwise_head_width,
                    self.pairwise_num_heads,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
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

        pair_mask = mask[..., None] & mask[..., None, :]

        blocks = [
            partial(
                b,
                single_mask=mask.float(),
                pair_mask=pair_mask.float(),
                chunk_size_tri_attn=chunk_size_tri_attn,
                use_cuequiv_mul=use_cuequiv_mul,
                use_cuequiv_attn=use_cuequiv_attn,
            )
            for b in self.blocks
        ]
        blocks_per_ckpt = self.blocks_per_ckpt

        if self.training and torch.is_grad_enabled():
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for block in blocks:
                s, z = block(s, z)

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
        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_heads: int = num_heads

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.attention = AttentionPairBias(
            channel_a=channel_s,
            channel_z=channel_z,
            num_heads=num_heads,
            channel_s=None,
            use_single_cond=False,
        )

        self.transition_s = Transition(channel_s, expansion_factor=4)
        self.transition_z = Transition(channel_z, expansion_factor=4)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """

        # Line 2
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_out(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        # Line 3
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_in(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        # Line 4
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn,
        )

        # Line 5
        dropout = get_dropout_mask(z, self.dropout, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn,
        )

        # Line 6
        z = z + self.transition_z(z)

        # Line 7
        s = s + self.attention(
            s,  # [B, L, C_s]
            None,
            z,  # [B, L, L, C_z]
            attn_mask=single_mask,  # [B, L]
        )

        # Line 8
        s = s + self.transition_s(s)

        return s, z
