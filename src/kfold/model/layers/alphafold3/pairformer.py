"""Section 3.6 Pairformer Stack of AlphaFold 3 paper."""

from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.utils.checkpointing import checkpoint_blocks

from .attention_pair_bias import SelfAttentionPairBias
from .transition import Transition


class PairformerStack(nn.Module):
    """Pairformer stack.
    See Section 3.6 Algorithm 20 Pairformer Stack
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        num_blocks: int = 48,
        dropout: float = 0.25,
        use_qk_norm: bool = False,
        blocks_per_ckpt: int | None = None,
    ):
        """Initialize the Pairformer module."""
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_heads_attn: int = num_heads_attn
        self.num_heads_tri_attn: int = num_heads_tri_attn
        self.dropout: float = dropout
        self.num_blocks: int = num_blocks

        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                PairformerBlock(
                    self.channel_s,
                    self.channel_z,
                    self.num_heads_attn,
                    self.num_heads_tri_attn,
                    self.dropout,
                    use_qk_norm,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
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
        chunk_size_tri_attn : int | None, optional
            The chunk size for triangle attention, by default None
        use_cuequiv_kernels : bool, optional
            Whether to use CuEQuiv kernels, by default False

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
                single_mask=mask,
                pair_mask=pair_mask,
                chunk_size_tri_attn=chunk_size_tri_attn,
                use_cuequiv_kernels=use_cuequiv_kernels,
            )
            for b in self.blocks
        ]
        s, z = checkpoint_blocks(
            blocks,
            (s, z),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )

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
        num_heads_attn: int = 16,
        num_heads_tri_attn: int = 4,
        dropout: float = 0.25,
        use_qk_norm: bool = False,
    ):
        """Initialize the Pairformer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_heads_attn : int, optional
            The number of attention heads, by default 16
        num_heads_tri_attn : int, optional
            The number of triangle attention heads, by default 4
        dropout : float, optional
            The dropout rate, by default 0.25
        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.dropout: float = dropout

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            channel_z, num_heads_tri_attn, inf=1e9
        )

        self.attention = SelfAttentionPairBias(
            channel_a=channel_s,
            channel_z=channel_z,
            num_heads=num_heads_attn,
            channel_s=None,
            qk_norm=use_qk_norm,
        )

        self.transition_s = Transition(channel_s, expansion_factor=4)
        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_rowwise = DropoutRowwise(dropout)
        self.dropout_columnwise = DropoutColumnwise(dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn: int | None = None,
        use_cuequiv_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """

        # Line 2
        z = z + self.dropout_rowwise(
            self.tri_mul_out(
                z,
                pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )

        # Line 3
        z = z + self.dropout_rowwise(
            self.tri_mul_in(
                z,
                mask=pair_mask,
                use_kernels=use_cuequiv_kernels,
            )
        )

        # Line 4
        z = z + self.dropout_rowwise(
            self.tri_att_start(
                z,
                mask=pair_mask,
                chunk_size=chunk_size_tri_attn,
                use_kernels=use_cuequiv_kernels,
            )
        )

        # Line 5
        z = z + self.dropout_columnwise(
            self.tri_att_end(
                z,
                mask=pair_mask,
                chunk_size=chunk_size_tri_attn,
                use_kernels=use_cuequiv_kernels,
            )
        )

        # Line 6
        z = z + self.transition_z(z)

        # Line 7
        s = s + self.attention(
            a=s,  # [B, L, C_s]
            s=None,
            z=z,  # [B, L, L, C_z]
            mask=single_mask,  # [B, L]
            use_kernels=use_cuequiv_kernels,
        )

        # Line 8
        s = s + self.transition_s(s)

        return s, z
